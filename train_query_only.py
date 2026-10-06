"""
Train QueryOnlyMLP: the query-features-only ablation of QPPMLP (see
model.py). Saves a checkpoint after every epoch (no dev-set checkpoint
selection) - run evaluate_query_only.py against each saved checkpoint
afterward to score them on your eval set.

Example:
    python train_query_only.py \
        --index /path/to/msmarco-passage-index \
        --train_run data/dl19_runs.txt \
        --train_qrels data/dl19-passage.qrels \
        --train_queries data/dl19-queries.tsv
"""

import argparse
import csv
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import QPPQueryOnlyDataset, build_query_only_feature_sanity_report
from feature_cache import load_feature_cache, load_query_embeddings
from features import IndexStats, QueryTypeClassifier
from losses import listmle_loss
from model import DEFAULT_HIDDEN_DIMS, QueryOnlyMLP
from train import load_queries


def make_run_tag(args):
    lr = f"{args.lr:.0e}".replace("-0", "-").replace("+0", "")
    do = str(args.dropout).replace("0.", "").replace(".", "")
    bs = f"_bs{args.batch_size}"
    hd = "-".join(str(h) for h in args.hidden_dims)
    loss_tag = "ce" if args.loss_type == "cross_entropy" else "mle"
    return f"qonly_h{hd}_lr{lr}_do{do}_e{args.epochs}{bs}_{loss_tag}"


def update_master_index(index_path, tag, args, notes):
    # Separate file/schema from train.py's runs_index.csv (no score_norm/
    # score_scale/top_k columns here), so query-only and full-model runs
    # don't collide in one CSV with mismatched fieldnames.
    fieldnames = ["run", "hidden_dims", "num_layers",
                  "lr", "dropout", "epochs", "batch_size", "loss_type", "notes"]
    row = {
        "run": tag, "hidden_dims": "-".join(str(h) for h in args.hidden_dims),
        "num_layers": len(args.hidden_dims),
        "lr": args.lr, "dropout": args.dropout, "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss_type": args.loss_type,
        "notes": notes or "",
    }
    write_header = not os.path.exists(index_path)
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_epoch(model, loader, optimizer, device, loss_type="cross_entropy"):
    model.train()
    criterion = nn.CrossEntropyLoss() if loss_type == "cross_entropy" else None
    total = 0.0
    for query_feats, labels in loader:
        query_feats = query_feats.to(device)
        labels = labels.to(device)

        logits = model(query_feats)
        if loss_type == "cross_entropy":
            # The best ranker per query is the one with max true nDCG@100.
            target_ranker = labels.argmax(dim=-1)
            loss = criterion(logits, target_ranker)
        else:  # listmle
            ranker_mask = torch.ones_like(labels, dtype=torch.bool)
            loss = listmle_loss(logits, labels, ranker_mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", default=None,
        help="Path to the Lucene index. Required as the --lexical_cache miss "
             "fallback; omit if --lexical_cache fully covers the training "
             "set's queries.",
    )
    parser.add_argument(
        "--query_embeddings", default=None,
        help="Path to the training dataset's precomputed raw query-embeddings "
             "pkl ({qid: vector}, see e.g. "
             "data/cache/bert-query-embeddings/cls/*.cls.pkl - width is "
             "inferred, not fixed). Required as the --embedding_cache miss "
             "fallback; omit if --embedding_cache fully covers the training "
             "set's queries.",
    )
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
        "--lexical_cache", default=None,
        help="Path to a precomputed lexical/IDF feature cache (see "
             "build_feature_cache.py) - the 5-dim lexical query features are "
             "looked up from it instead of being recomputed via Lucene.",
    )
    parser.add_argument(
        "--embedding_cache", default=None,
        help="Path to a precomputed raw embedding feature cache (see "
             "build_embedding_cache.py) - the raw query representation "
             "(variable width by source) is looked up from it instead of "
             "being recomputed from --query_embeddings.",
    )
    parser.add_argument(
        "--query_type_cache", default=None,
        help="Path to a precomputed query_type feature cache (see "
             "build_query_type_cache.py) - the 1-dim query_type flag is "
             "looked up from it instead of being recomputed by the "
             "classifier.",
    )
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS,
        help="Sizes of the MLP's hidden layers, in order (number of layers "
             f"n1 = len(hidden_dims)). Default: {DEFAULT_HIDDEN_DIMS}.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--loss_type",
        default="cross_entropy",
        choices=["cross_entropy", "listmle"],
        help="cross_entropy: top-1 best-ranker classification. "
             "listmle: listwise ranking loss over all rankers' graded NDCG labels.",
    )
    parser.add_argument(
        "--standardize_embedding", action="store_true",
        help="Z-score the raw embedding block (before it's projected by "
             "embedding_proj) using real computed statistics, like every "
             "other query feature. Default: off, matching the original "
             "PCA-embedding behavior of never standardizing it. Only affects "
             "training (fit_standardization) - evaluate_query_only.py has no "
             "matching flag since it loads already-fitted buffers from the "
             "checkpoint and never refits them.",
    )
    parser.add_argument(
        "--no_embedding_reduction", action="store_true",
        help="Skip the learned embedding_proj reduction to EMBEDDING_DIM "
             "(32) dims and feed the full raw embedding width straight into "
             "the MLP instead. Default: off (reduce, matching original "
             "behavior). Changes the model's architecture, so "
             "evaluate_query_only.py needs the matching "
             "--no_embedding_reduction flag to reload a checkpoint trained "
             "with this set.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--results_path", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--feature_report_path", default=None,
        help="Where to write the query-feature sanity-check log (JSON): "
             "per-feature min/max/mean/nonzero and tensor shape. Defaults to "
             "'<save_path>.feature_report.json', or 'feature_report_<tag>.json' "
             "under --run_dir if that's set. Pass an empty string to skip.",
    )
    args = parser.parse_args()

    if args.index is None and args.lexical_cache is None:
        parser.error("--index or --lexical_cache (or both) must be provided")
    if args.query_embeddings is None and args.embedding_cache is None:
        parser.error("--query_embeddings or --embedding_cache (or both) must be provided")

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
        args.save_path = "model_query_only_best.pt"
    if args.feature_report_path is None:
        args.feature_report_path = args.save_path + ".feature_report.json"

    print(f"Run tag : {tag}")
    print(f"Model   : {args.save_path} (one checkpoint per epoch: <name>_epoch<N>.pt)")
    if args.results_path:
        print(f"Results : {args.results_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.index:
        print("Loading index...")
        index_stats = IndexStats(args.index)
    else:
        index_stats = None
        print("No --index given; relying entirely on --lexical_cache "
              "(a cache miss will raise).")

    if args.query_embeddings:
        print("Loading query embeddings...")
        embedding_lookup = load_query_embeddings(args.query_embeddings)
    else:
        embedding_lookup = None
        print("No --query_embeddings given; relying entirely on "
              "--embedding_cache (a cache miss will raise).")

    # Cheap/lazy to construct (the HF pipeline only loads on first .classify()
    # call), so always available as the --query_type_cache miss fallback.
    query_type_classifier = QueryTypeClassifier()

    lexical_cache = load_feature_cache(args.lexical_cache) if args.lexical_cache else None
    embedding_cache = load_feature_cache(args.embedding_cache) if args.embedding_cache else None
    query_type_cache = load_feature_cache(args.query_type_cache) if args.query_type_cache else None

    train_queries = load_queries(args.train_queries)

    print("Building dataset (query features only)...")
    train_ds = QPPQueryOnlyDataset(
        args.train_run, args.train_qrels, train_queries, index_stats,
        lexical_cache=lexical_cache,
        embedding_cache=embedding_cache,
        query_type_cache=query_type_cache,
        embedding_lookup=embedding_lookup,
        query_type_classifier=query_type_classifier,
        metrics_csv=args.train_metrics_csv,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Rankers ({train_ds.num_rankers}): {train_ds.ranker_to_id}")

    if args.feature_report_path:
        feature_report = build_query_only_feature_sanity_report(train_ds)
        with open(args.feature_report_path, "w") as f:
            json.dump(feature_report, f, indent=2)
        print(f"  Feature report saved to {args.feature_report_path}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = QueryOnlyMLP(
        query_feature_dim=train_ds.query_feature_dim,
        embedding_slice=train_ds.embedding_slice,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        num_rankers=train_ds.num_rankers,
        standardize_embedding=args.standardize_embedding,
        reduce_embedding=not args.no_embedding_reduction,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fit standardization on the TRAINING set only, before training. Buffers
    # default to identity, so this step is required or standardization silently
    # does nothing.
    print("Fitting feature standardization on training set...")
    fit_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    model.fit_standardization(fit_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    results = {
        "config": {
            "run_tag": tag,
            "notes": args.notes,
            "ablation": "query_only",
            "hidden_dims": args.hidden_dims,
            "num_layers": len(args.hidden_dims),
            "dropout": args.dropout,
            "loss_type": args.loss_type,
            "standardize_embedding": args.standardize_embedding,
            "embedding_reduced": not args.no_embedding_reduction,
            "num_rankers": train_ds.num_rankers,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "train_run": args.train_run,
            "train_qrels": args.train_qrels,
            "train_metrics_csv": args.train_metrics_csv,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "epochs": [],
    }

    save_stem, save_ext = os.path.splitext(args.save_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_type=args.loss_type)
        scheduler.step()

        # Every epoch gets its own checkpoint - no dev-set comparison, so there's
        # no notion of a single "best" one here. Score them against your eval set
        # afterward (e.g. with evaluate_query_only.py, once per checkpoint).
        epoch_save_path = f"{save_stem}_epoch{epoch}{save_ext}"
        torch.save(model.state_dict(), epoch_save_path)

        # Ranker->id map is the same every epoch (built once from train_ds), but
        # is saved alongside each checkpoint so evaluate_query_only.py's
        # auto-detection (<model_path>.rankers.json) keeps working per-checkpoint.
        with open(epoch_save_path + ".rankers.json", "w") as f:
            json.dump(train_ds.ranker_to_id, f, indent=2)

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "save_path": epoch_save_path,
        })

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}  -> Saved {epoch_save_path}")

        if args.results_path:
            with open(args.results_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nTraining complete: saved {args.epochs} checkpoints "
          f"(pattern: {save_stem}_epoch<N>{save_ext})")
    if args.results_path:
        print(f"Results saved to {args.results_path}")
    if args.run_dir:
        index_path = os.path.join(args.run_dir, "runs_index_query_only.csv")
        update_master_index(index_path, tag, args, args.notes)
        print(f"Index   : {index_path}")


if __name__ == "__main__":
    main()
