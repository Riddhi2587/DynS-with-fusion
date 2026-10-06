"""
Train QPPMLP on TREC DL 2019 or 2020 to classify, per query, which ranker
produces the best (highest nDCG@100) ranked list. Saves a checkpoint after
every epoch (no dev-set checkpoint selection) - run evaluate.py against each
saved checkpoint afterward to score them on your eval set.

Example:
    python train.py \
        --train_run data/dl19_runs.txt \
        --train_qrels data/dl19-passage.qrels \
        --train_queries data/dl19-queries.tsv \
        --lexical_cache dl19_lexical.pkl \
        --embedding_cache dl19_embedding.pkl \
        --query_type_cache dl19_query_type.pkl \
        --entity_count_cache dl19_entity_count.pkl \
        --scs_pmi_cache dl19_scs_pmi.pkl \
        --doc_feature_cache dl19_doc_features.pkl \
        --score_norm per_ranker

All six feature caches are required (whichever ones --features selects) -
Lucene is never touched here; build them once, offline, with
build_feature_cache.py/build_embedding_cache.py/build_query_type_cache.py/
build_entity_count_cache.py/build_scs_pmi_cache.py/build_doc_feature_cache.py
(see guide_docs/FEATURE_CACHE_GUIDE.md).
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import QPPDataset, build_feature_sanity_report
from feature_cache import load_feature_cache
from features import ALL_FEATURE_BLOCKS, validate_feature_blocks
from losses import listmle_loss
from model import DEFAULT_HIDDEN_DIMS, QPPMLP
from timing_utils import SUMMARY_HEADER, Timer, append_rows_csv, summary_row, write_env_json

# Short tag abbreviations for each feature block, in ALL_FEATURE_BLOCKS'
# canonical order - used by make_run_tag so run directories/tags stay
# distinguishable across --features configs.
_FEATURE_TAG_ABBREV = {
    "lexical": "lex", "embedding": "emb", "query_type": "qt", "entity_count": "ent",
    "scs_pmi": "scspmi", "doc_feats": "doc",
}


def make_run_tag(args):
    lr = f"{args.lr:.0e}".replace("-0", "-").replace("+0", "")
    do = str(args.dropout).replace("0.", "").replace(".", "")
    scale = f"_s{args.score_scale:g}" if args.score_scale is not None else ""
    norm = f"_{args.score_norm}"
    bs = f"_bs{args.batch_size}"
    hd = "-".join(str(h) for h in args.hidden_dims)
    loss_tag = "ce" if args.loss_type == "cross_entropy" else "mle"
    feat_tag = "-".join(_FEATURE_TAG_ABBREV[b] for b in ALL_FEATURE_BLOCKS if b in args.features)
    return f"h{hd}_lr{lr}_do{do}_e{args.epochs}{bs}_{loss_tag}{scale}{norm}_feat-{feat_tag}"


def update_master_index(index_path, tag, args, notes):
    fieldnames = ["run", "hidden_dims", "num_layers", "features",
                  "lr", "dropout", "epochs", "batch_size", "score_norm", "loss_type", "notes"]
    row = {
        "run": tag, "hidden_dims": "-".join(str(h) for h in args.hidden_dims),
        "num_layers": len(args.hidden_dims),
        "features": "+".join(args.features),
        "lr": args.lr, "dropout": args.dropout, "epochs": args.epochs,
        "batch_size": args.batch_size,
        "score_norm": args.score_norm,
        "loss_type": args.loss_type,
        "notes": notes or "",
    }
    write_header = not os.path.exists(index_path)
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_queries(path: str):
    queries = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def train_epoch(model, loader, optimizer, device, loss_type="cross_entropy"):
    model.train()
    criterion = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else None
    total = 0.0
    for doc_feats, list_feats, query_feats, pad_mask, ranker_mask, labels in loader:
        doc_feats = doc_feats.to(device)
        list_feats = list_feats.to(device)
        query_feats = query_feats.to(device)
        pad_mask = pad_mask.to(device)
        ranker_mask = ranker_mask.to(device)
        labels = labels.to(device)

        logits = model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)
        if loss_type == "cross_entropy":
            # The best ranker per query is the one with max true nDCG@100 among
            # the rankers that actually have data for that query.
            target_ndcg = labels.masked_fill(~ranker_mask, float("-inf"))
            target_ranker = target_ndcg.argmax(dim=-1)
            loss = criterion(logits, target_ranker)
        else:  # listmle
            loss = listmle_loss(logits, labels, ranker_mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_run", required=True, nargs="+")
    parser.add_argument("--train_qrels", required=True)
    parser.add_argument("--train_queries", required=True)
    parser.add_argument(
        "--train_metrics_csv", required=True,
        help="Path to a precomputed per-(ranker, qid) metrics CSV (see "
             "build_metrics_csv.py and guide_docs/PRECOMPUTED_METRICS_GUIDE.md). "
             "Required - there is no pytrec_eval fallback.",
    )
    parser.add_argument(
        "--features", nargs="+", default=list(ALL_FEATURE_BLOCKS), choices=list(ALL_FEATURE_BLOCKS),
        help="Which input feature blocks to include, any subset of "
             f"{list(ALL_FEATURE_BLOCKS)}: lexical (5-dim IDF), embedding "
             "(raw query representation, variable width by source - reduced "
             "to EMBEDDING_DIM dims by a trained nn.Linear in the model), "
             "query_type (1-dim "
             "keyword-vs-natural-language flag), entity_count (1-dim "
             "precomputed named-entity count for the query - see "
             "build_entity_count_cache.py), scs_pmi (3-dim precomputed "
             "scs/avg_pmi/max_pmi triple for the query - see "
             "build_scs_pmi_cache.py), doc_feats (per-ranker "
             "doc-list features - LIST_FEATURE_DIM once-per-ranker stats + "
             "DOC_FEATURE_DIM per doc * top_k). At least one required. "
             "Default: all six (the original full model).",
    )
    parser.add_argument(
        "--lexical_cache", default=None,
        help="Path to a precomputed lexical/IDF feature cache (see "
             "build_feature_cache.py). Required if 'lexical' is in "
             "--features - no live fallback (Lucene is never touched here; "
             "see QPPDataset's require_caches).",
    )
    parser.add_argument(
        "--embedding_cache", default=None,
        help="Path to a precomputed raw embedding feature cache (see "
             "build_embedding_cache.py, built from a raw source like "
             "data/cache/bert-query-embeddings/cls/*.cls.pkl - width is "
             "inferred, not fixed). Required if 'embedding' is in "
             "--features - no live fallback.",
    )
    parser.add_argument(
        "--query_type_cache", default=None,
        help="Path to a precomputed query_type feature cache (see "
             "build_query_type_cache.py). Required if 'query_type' is in "
             "--features - no live fallback.",
    )
    parser.add_argument(
        "--entity_count_cache", default=None,
        help="Path to a precomputed entity_count feature cache (see "
             "build_entity_count_cache.py) - the 1-dim precomputed "
             "named-entity count per query. Required if 'entity_count' is "
             "in --features - no live fallback.",
    )
    parser.add_argument(
        "--scs_pmi_cache", default=None,
        help="Path to a precomputed scs_pmi feature cache (see "
             "build_scs_pmi_cache.py) - the 3-dim precomputed "
             "(scs, avg_pmi, max_pmi) triple per query. Required if "
             "'scs_pmi' is in --features - no live fallback.",
    )
    parser.add_argument(
        "--doc_feature_cache", default=None,
        help="Path to a precomputed doc-content feature cache (see "
             "build_doc_feature_cache.py) - the 8-dim Lucene-derived "
             "per-(qid, doc_id) term features (features.DOC_TERM_FEATURE_NAMES). "
             "Required if 'doc_feats' is in --features - no live fallback. "
             "`score` itself always comes from the run file, never cached.",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS,
        help="Sizes of the MLP's hidden layers, in order (number of layers "
             f"n1 = len(hidden_dims)). Default: {DEFAULT_HIDDEN_DIMS}.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--score_scale", type=float, default=None)
    parser.add_argument(
        "--loss_type",
        default="cross_entropy",
        choices=["cross_entropy", "listmle"],
        help="cross_entropy: top-1 best-ranker classification. "
             "listmle: listwise ranking loss over all rankers' graded NDCG labels.",
    )
    parser.add_argument(
        "--score_norm",
        default="global",
        choices=["global", "per_query", "per_ranker"],
        help="How to normalize the score-derived doc features. "
             "global: one mean/std pooled over all rankers. "
             "per_query: within-list z-score + scale-free shape features. "
             "per_ranker: standardize score features per ranker (needs shared rankers).",
    )
    parser.add_argument(
        "--standardize_embedding", action="store_true",
        help="Z-score the raw embedding block (before it's projected by "
             "embedding_proj) using real computed statistics, like every "
             "other query feature. Default: off, matching the original "
             "PCA-embedding behavior of never standardizing it (it's already "
             "L2-normalized per query at feature-construction time). Only "
             "affects training (fit_standardization) - evaluate.py has no "
             "matching flag since it loads already-fitted buffers from the "
             "checkpoint and never refits them.",
    )
    parser.add_argument(
        "--no_embedding_reduction", action="store_true",
        help="Skip the learned embedding_proj reduction to EMBEDDING_DIM "
             "(32) dims and feed the full raw embedding width straight into "
             "the MLP instead. Default: off (reduce, matching original "
             "behavior). Changes the model's architecture, so evaluate.py "
             "needs the matching --no_embedding_reduction flag to reload a "
             "checkpoint trained with this set.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--results_path", default=None)
    parser.add_argument(
        "--feature_report_path", default=None,
        help="Where to write the feature sanity-check log (JSON): per-feature "
             "min/max/mean/nonzero and tensor shapes. Defaults to "
             "'<save_path>.feature_report.json', or 'feature_report_<tag>.json' "
             "under --run_dir if that's set.",
    )
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--dataset_name", default=None,
        help="Label for the 'dataset' column of the timing CSV (e.g. "
             "msmarco-dev, llm-judged). Defaults to the --train_queries filename.",
    )
    parser.add_argument(
        "--timing_csv", default=None,
        help="Append this run's step timings (cache_load, dataset_build, "
             "fit_standardization, train_total, ...) to this CSV, same schema as "
             "time_feature_computation.py's train_timing_summary.csv. Defaults to "
             "<run_dir>/train_timing_summary.csv if --run_dir is set, else no CSV "
             "(a 'timing' block is still added to the results JSON).",
    )
    args = parser.parse_args()

    try:
        validate_feature_blocks(args.features)
    except ValueError as e:
        parser.error(str(e))
    if "lexical" in args.features and args.lexical_cache is None:
        parser.error("--lexical_cache is required when 'lexical' is included in --features")
    if "embedding" in args.features and args.embedding_cache is None:
        parser.error("--embedding_cache is required when 'embedding' is included in --features")
    if "query_type" in args.features and args.query_type_cache is None:
        parser.error("--query_type_cache is required when 'query_type' is included in --features")
    if "entity_count" in args.features and args.entity_count_cache is None:
        parser.error("--entity_count_cache is required when 'entity_count' is included in --features")
    if "scs_pmi" in args.features and args.scs_pmi_cache is None:
        parser.error("--scs_pmi_cache is required when 'scs_pmi' is included in --features")
    if "doc_feats" in args.features and args.doc_feature_cache is None:
        parser.error("--doc_feature_cache is required when 'doc_feats' is included in --features")

    tag = make_run_tag(args)
    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        if args.results_path is None:
            args.results_path = os.path.join(args.run_dir, f"results_{tag}.json")
        if args.save_path is None:
            args.save_path = os.path.join(args.run_dir, f"model_{tag}.pt")
        if args.feature_report_path is None:
            args.feature_report_path = os.path.join(args.run_dir, f"feature_report_{tag}.json")
    if args.save_path is None:
        args.save_path = "model_best.pt"
    if args.feature_report_path is None:
        args.feature_report_path = args.save_path + ".feature_report.json"

    print(f"Run tag : {tag}")
    print(f"Model   : {args.save_path} (one checkpoint per epoch: <name>_epoch<N>.pt)")
    if args.results_path:
        print(f"Results : {args.results_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    wall_start = time.perf_counter()

    # Lucene is never touched here: index_stats always stays None, and
    # require_caches=True below forbids QPPDataset from falling back to it
    # even if it weren't - every selected block must come from a cache.
    index_stats = None

    with Timer() as t_cache_load:
        lexical_cache = load_feature_cache(args.lexical_cache) if "lexical" in args.features else None
        embedding_cache = load_feature_cache(args.embedding_cache) if "embedding" in args.features else None
        query_type_cache = load_feature_cache(args.query_type_cache) if "query_type" in args.features else None
        entity_count_cache = load_feature_cache(args.entity_count_cache) if "entity_count" in args.features else None
        scs_pmi_cache = load_feature_cache(args.scs_pmi_cache) if "scs_pmi" in args.features else None
        doc_feature_cache = load_feature_cache(args.doc_feature_cache) if "doc_feats" in args.features else None

    train_queries = load_queries(args.train_queries)

    print(f"Building dataset (features={args.features}, score_norm={args.score_norm})...")
    with Timer() as t_dataset_build:
        train_ds = QPPDataset(
            args.train_run, args.train_qrels, train_queries, index_stats,
            args.top_k, args.score_scale, args.score_norm,
            metrics_csv=args.train_metrics_csv,
            lexical_cache=lexical_cache,
            embedding_cache=embedding_cache,
            query_type_cache=query_type_cache,
            entity_count_cache=entity_count_cache,
            scs_pmi_cache=scs_pmi_cache,
            doc_feature_cache=doc_feature_cache,
            require_caches=True,
            feature_blocks=args.features,
        )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Rankers ({train_ds.num_rankers}): {train_ds.ranker_to_id}")
    print(f"  Dims: doc_feature_dim={train_ds.doc_feature_dim} "
          f"list_feature_dim={train_ds.list_feature_dim} "
          f"query_feature_dim={train_ds.query_feature_dim}")

    print("Building feature sanity-check report...")
    feature_report = build_feature_sanity_report(train_ds)
    with open(args.feature_report_path, "w") as f:
        json.dump(feature_report, f, indent=2)
    dead = {k: v["dead_term_features"] for k, v in feature_report["branches"].items() if v["dead_term_features"]}
    if dead:
        print(f"  WARNING: constant-zero term-based features detected: {dead}")
    print(f"  Feature report saved to {args.feature_report_path}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = QPPMLP(
        doc_feature_dim=train_ds.doc_feature_dim,
        list_feature_dim=train_ds.list_feature_dim,
        query_feature_dim=train_ds.query_feature_dim,
        embedding_slice=train_ds.embedding_slice,
        top_k=args.top_k,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        score_norm=args.score_norm,
        num_rankers=train_ds.num_rankers,
        standardize_embedding=args.standardize_embedding,
        reduce_embedding=not args.no_embedding_reduction,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fit standardization on the TRAINING set only, before training. Buffers
    # default to identity, so this step is required or standardization silently
    # does nothing. For per_ranker this also fits the per-ranker score stats.
    print("Fitting feature standardization on training set...")
    fit_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    with Timer(device) as t_fit_standardization:
        model.fit_standardization(fit_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    results = {
        "config": {
            "run_tag": tag,
            "notes": args.notes,
            "features": args.features,
            "hidden_dims": args.hidden_dims,
            "num_layers": len(args.hidden_dims),
            "dropout": args.dropout,
            "loss_type": args.loss_type,
            "score_scale": args.score_scale,
            "score_norm": args.score_norm,
            "standardize_embedding": args.standardize_embedding,
            "embedding_reduced": not args.no_embedding_reduction,
            "num_rankers": train_ds.num_rankers,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "feature_report_path": args.feature_report_path,
            "seed": args.seed,
            "train_run": args.train_run,
            "train_qrels": args.train_qrels,
            "train_metrics_csv": args.train_metrics_csv,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "epochs": [],
        "timing": {
            "device": str(device),
            "cache_load_s": round(t_cache_load.seconds, 6),
            "dataset_build_s": round(t_dataset_build.seconds, 6),
            "fit_standardization_s": round(t_fit_standardization.seconds, 6),
            "epoch_train_s": [],
            "checkpoint_io_s": [],
        },
    }
    timing = results["timing"]

    save_stem, save_ext = os.path.splitext(args.save_path)

    for epoch in range(1, args.epochs + 1):
        # Epoch time = the train_epoch call only (DataLoader included, since it's
        # part of real training cost); checkpoint/sidecar/results IO is timed
        # separately below so it never inflates the training number.
        with Timer(device) as t_epoch:
            train_loss = train_epoch(model, train_loader, optimizer, device, loss_type=args.loss_type)
        scheduler.step()

        # Every epoch gets its own checkpoint - no dev-set comparison, so there's
        # no notion of a single "best" one here. Score them against your eval set
        # afterward (e.g. with evaluate.py, once per checkpoint).
        epoch_save_path = f"{save_stem}_epoch{epoch}{save_ext}"
        with Timer() as t_ckpt:
            torch.save(model.state_dict(), epoch_save_path)

            # Ranker->id map is the same every epoch (built once from train_ds), but
            # is saved alongside each checkpoint so evaluate.py's auto-detection
            # (<model_path>.rankers.json) keeps working per-checkpoint unchanged.
            with open(epoch_save_path + ".rankers.json", "w") as f:
                json.dump(train_ds.ranker_to_id, f, indent=2)

            # Feature-block config is also fixed for the whole run, saved per
            # checkpoint so evaluate.py's auto-detection (<model_path>.features.json)
            # reconstructs the exact same QPPMLP architecture without needing
            # --features passed explicitly at eval time.
            with open(epoch_save_path + ".features.json", "w") as f:
                json.dump(args.features, f, indent=2)

        timing["epoch_train_s"].append(round(t_epoch.seconds, 6))
        timing["checkpoint_io_s"].append(round(t_ckpt.seconds, 6))

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "save_path": epoch_save_path,
        })

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}  "
              f"({t_epoch.seconds:.1f}s) -> Saved {epoch_save_path}")

        if args.results_path:
            with open(args.results_path, "w") as f:
                json.dump(results, f, indent=2)

    timing["train_total_s"] = round(sum(timing["epoch_train_s"]), 6)
    timing["checkpoint_io_total_s"] = round(sum(timing["checkpoint_io_s"]), 6)
    timing["wall_total_s"] = round(time.perf_counter() - wall_start, 6)
    print(f"\nTiming: cache_load={timing['cache_load_s']:.1f}s "
          f"dataset_build={timing['dataset_build_s']:.1f}s "
          f"fit_standardization={timing['fit_standardization_s']:.1f}s "
          f"train_total={timing['train_total_s']:.1f}s "
          f"wall_total={timing['wall_total_s']:.1f}s")

    timing_csv = args.timing_csv or (
        os.path.join(args.run_dir, "train_timing_summary.csv") if args.run_dir else None
    )
    if timing_csv:
        ds_name = args.dataset_name or os.path.basename(args.train_queries)
        n_epochs = len(timing["epoch_train_s"])
        append_rows_csv(timing_csv, SUMMARY_HEADER, [
            summary_row(ds_name, "cache_load", timing["cache_load_s"], None, "cpu"),
            summary_row(ds_name, "dataset_build", timing["dataset_build_s"], len(train_ds), "cpu"),
            summary_row(ds_name, "fit_standardization", timing["fit_standardization_s"], None, device),
            summary_row(ds_name, "train_total", timing["train_total_s"], n_epochs, device),
            summary_row(ds_name, "checkpoint_io", timing["checkpoint_io_total_s"], n_epochs, "cpu"),
            summary_row(ds_name, "train_wall_total", timing["wall_total_s"], None, device),
        ])
        write_env_json(timing_csv + ".env.json", device, {"note": "shared by all rows"})
        print(f"Timing  : {timing_csv}")

    if args.results_path:
        with open(args.results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nTraining complete: saved {args.epochs} checkpoints "
          f"(pattern: {save_stem}_epoch<N>{save_ext})")
    if args.results_path:
        print(f"Results saved to {args.results_path}")
    if args.run_dir:
        index_path = os.path.join(args.run_dir, "runs_index.csv")
        update_master_index(index_path, tag, args, args.notes)
        print(f"Index   : {index_path}")


if __name__ == "__main__":
    main()
