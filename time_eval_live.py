"""
Per-query runtime of the FULL live inference path for a trained QPPMLP: the
lexical/IDF features, the MiniLM query embedding and the query_type flag are all
computed live (no feature caches), models pre-loaded and warmed up, then the
assembled features go through the MLP. One row per qid, batch size 1.

Columns of eval_timing_<dataset>.csv:
    dataset, qid, lexical_s, embedding_s, query_type_s, assembly_s, mlp_s,
    total_s, device
  lexical_s     lowercase/split + features.build_query_features (Lucene IDF;
                the IDF memo is cleared per query unless --keep_idf_cache)
  embedding_s   MiniLM encode of the single query + L2-normalize
                (features.build_embedding_feature)
  query_type_s  features.build_query_type_feature (bert-mini classifier)
  assembly_s    concat lexical|embedding|query_type + the zero-width doc/list
                tensors and masks QPPDataset would yield + move to device
  mlp_s         QPPMLP forward + softmax
A block the checkpoint wasn't trained with is skipped and its column left blank.
--num_feature_samples N (default 10) additionally saves the live lexical/embedding/
query_type values of N random queries - the exact vectors fed to the MLP - to
feature_samples_eval_<dataset>.json for sanity-checking (see feature_samples.py);
they're read from the timed run and described after timing, so timings are unaffected.
Only the lexical/embedding/query_type blocks have a live path here, so the
checkpoint's features must be a subset of those.

Example:
    python time_eval_live.py \
        --dataset dl19 --queries data/dl19-queries.tsv \
        --index /path/to/msmarco-passage-index \
        --model_path runs/model_..._epoch40.pt \
        --hidden_dims 128 128 --no_embedding_reduction \
        --output_dir timing_results
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from embedding_live import MiniLMEncoder, default_device, make_query_type_classifier
from feature_samples import describe_query, pick_sample_qids, write_feature_samples
from features import (
    ALL_FEATURE_BLOCKS,
    build_embedding_feature,
    build_query_features,
    build_query_type_feature,
    resolve_doc_feature_layout,
    resolve_query_feature_layout,
    validate_feature_blocks,
)
from model import DEFAULT_HIDDEN_DIMS, QPPMLP
from timing_utils import Timer, append_rows_csv, device_str, write_env_json
from train import load_queries

LIVE_BLOCKS = ("lexical", "embedding", "query_type")
CSV_HEADER = [
    "dataset", "qid", "lexical_s", "embedding_s", "query_type_s",
    "assembly_s", "mlp_s", "total_s", "device",
]
WARMUP_QUERY = "warm up query"


def _clear_idf_cache(index_stats) -> None:
    cache = getattr(index_stats, "_idf_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def resolve_live_blocks(features) -> tuple:
    """Canonical-order tuple of the checkpoint's blocks; errors unless they are
    all live-computable here."""
    validate_feature_blocks(features)
    blocks = tuple(b for b in ALL_FEATURE_BLOCKS if b in features)
    unsupported = [b for b in blocks if b not in LIVE_BLOCKS]
    if unsupported:
        raise ValueError(
            f"Checkpoint uses feature block(s) {unsupported} that have no live "
            f"computation path; time_eval_live.py supports only {list(LIVE_BLOCKS)}."
        )
    return blocks


def _per_query_features(qid, raw, blocks, index_stats, encoder, classifier, keep_idf_cache, timings):
    """Compute (and time) each selected block for one query. Returns
    {block: vector}; per-block seconds go into `timings`."""
    feats: Dict[str, np.ndarray] = {}
    if "lexical" in blocks:
        if not keep_idf_cache:
            _clear_idf_cache(index_stats)
        with Timer() as t:  # Lucene lookups are CPU-only
            terms = raw.lower().split()
            feats["lexical"] = build_query_features(terms, index_stats)
        timings["lexical_s"] = t.seconds
    if "embedding" in blocks:
        with Timer(encoder.device) as t:
            vec = encoder.encode([raw], batch_size=1)[0]
            feats["embedding"] = build_embedding_feature(qid, {qid: vec})
        timings["embedding_s"] = t.seconds
    if "query_type" in blocks:
        with Timer(classifier_device(classifier)) as t:
            feats["query_type"] = build_query_type_feature(raw, classifier)
        timings["query_type_s"] = t.seconds
    return feats


def classifier_device(classifier):
    """The device a classifier's pipeline runs on (for CUDA sync), if it has one."""
    pipe = getattr(classifier, "_pipeline_fn", None)
    return getattr(pipe, "device", None)


def time_queries(
    dataset: str,
    queries: Dict[str, str],
    blocks: tuple,
    index_stats,
    encoder,
    classifier,
    model: QPPMLP,
    device,
    num_rankers: int,
    top_k: int,
    keep_idf_cache: bool = False,
    warmup: int = 3,
    keep_vectors_for: int = 0,
    keep_qids=None,
):
    """Times every query in `queries`. Returns (rows, live_vectors) where rows
    are CSV_HEADER dicts and live_vectors holds {block: vector} - the exact
    vectors that fed the model - for the first `keep_vectors_for` queries (the
    optional cache cross-check) plus every qid in `keep_qids` (the saved feature
    samples). Keeping them is a dict store outside every timed region."""
    keep_qids = set(keep_qids or ())
    dev = device_str(device)
    doc_dim, list_dim, _, _ = resolve_doc_feature_layout(blocks)

    def run_one(qid, raw, timings):
        feats = _per_query_features(
            qid, raw, blocks, index_stats, encoder, classifier, keep_idf_cache, timings,
        )
        with Timer(device) as t_asm:
            query_feats = np.concatenate([feats[b] for b in blocks]).astype(np.float32)
            doc_np = np.zeros((num_rankers, top_k, doc_dim), dtype=np.float32)
            list_np = np.zeros((num_rankers, list_dim), dtype=np.float32)
            pad_np = np.ones((num_rankers, top_k), dtype=bool)
            rmask_np = np.ones(num_rankers, dtype=bool)
            doc_t = torch.from_numpy(doc_np).unsqueeze(0).to(device)
            list_t = torch.from_numpy(list_np).unsqueeze(0).to(device)
            q_t = torch.from_numpy(query_feats).unsqueeze(0).to(device)
            pad_t = torch.from_numpy(pad_np).unsqueeze(0).to(device)
            rmask_t = torch.from_numpy(rmask_np).unsqueeze(0).to(device)
        timings["assembly_s"] = t_asm.seconds
        with Timer(device) as t_mlp:
            with torch.no_grad():
                logits = model(doc_t, list_t, q_t, pad_t, rmask_t)
                torch.softmax(logits, dim=-1)
        timings["mlp_s"] = t_mlp.seconds
        return feats

    model.eval()
    first = next(iter(queries.items()))
    for _ in range(warmup):  # warm-up: lazy CUDA init, allocator, first-call overheads
        run_one(first[0], first[1], {})

    rows: List[Dict] = []
    live_vectors: Dict[str, Dict[str, np.ndarray]] = {}
    for i, (qid, raw) in enumerate(queries.items()):
        timings: Dict[str, float] = {}
        feats = run_one(qid, raw, timings)
        if i < keep_vectors_for or qid in keep_qids:
            live_vectors[qid] = feats
        total = sum(timings.values())
        fmt = lambda k: f"{timings[k]:.6f}" if k in timings else ""
        rows.append({
            "dataset": dataset, "qid": qid,
            "lexical_s": fmt("lexical_s"), "embedding_s": fmt("embedding_s"),
            "query_type_s": fmt("query_type_s"), "assembly_s": fmt("assembly_s"),
            "mlp_s": fmt("mlp_s"), "total_s": f"{total:.6f}", "device": dev,
        })
    return rows, live_vectors


def build_eval_samples(queries, live_vectors, sample_qids, index_stats) -> List[Dict]:
    """feature_samples.describe_query for each sampled qid, from the live vectors
    time_queries kept. Run after timing (it does IDF lookups)."""
    samples = []
    for qid in sample_qids:
        feats = live_vectors[qid]
        samples.append(describe_query(
            qid, queries[qid], feats.get("lexical"), feats.get("embedding"),
            feats.get("query_type"), index_stats,
        ))
    return samples


def check_against_caches(live_vectors, cache_paths: Dict[str, str], atol: float = 1e-4) -> int:
    """Correctness cross-check (NOT timed): live vectors vs the cached ones for the
    qids we kept. Returns the number of mismatching (qid, block) pairs."""
    from feature_cache import load_feature_cache

    mismatches = 0
    for block, path in cache_paths.items():
        cache = load_feature_cache(path)
        for qid, feats in live_vectors.items():
            if block not in feats:
                continue
            cached = cache.query_feats.get(qid)
            if cached is None:
                print(f"  [check] {block}: qid {qid} missing from cache")
                mismatches += 1
            elif not np.allclose(feats[block], cached, atol=atol):
                print(f"  [check] {block}: qid {qid} differs "
                      f"(max abs diff {np.max(np.abs(feats[block] - cached)):.3e})")
                mismatches += 1
    print(f"  [check] {len(live_vectors)} qids compared, {mismatches} mismatching (qid, block) pairs")
    return mismatches


def build_model(blocks, embedding_dim, num_rankers, args, device) -> QPPMLP:
    query_dim, _, emb_slice = resolve_query_feature_layout(blocks, embedding_dim=embedding_dim)
    doc_dim, list_dim, _, _ = resolve_doc_feature_layout(blocks)
    model = QPPMLP(
        doc_feature_dim=doc_dim,
        list_feature_dim=list_dim,
        query_feature_dim=query_dim,
        embedding_slice=emb_slice,
        top_k=args.top_k,
        hidden_dims=args.hidden_dims,
        score_norm=args.score_norm,
        num_rankers=num_rankers,
        reduce_embedding=not args.no_embedding_reduction,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Label for the CSV (e.g. dl19).")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--index", default=None,
                        help="Lucene index for THIS dataset (required if 'lexical' is in the checkpoint's features).")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--features", nargs="+", default=None, choices=list(ALL_FEATURE_BLOCKS),
                        help="Defaults to <model_path>.features.json (written by train.py).")
    parser.add_argument("--ranker_map", default=None,
                        help="Defaults to <model_path>.rankers.json (written by train.py).")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--score_norm", default="global", choices=["global", "per_query", "per_ranker"])
    parser.add_argument("--no_embedding_reduction", action="store_true")
    parser.add_argument("--output_dir", default="timing_results")
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_queries", type=int, default=None, help="Time only the first N queries (smoke test).")
    parser.add_argument("--keep_idf_cache", action="store_true",
                        help="Don't clear the IDF memo before each query.")
    parser.add_argument("--check_against_cache", nargs="+", metavar="BLOCK=PKL", default=None,
                        help="Cross-check the first --check_n live vectors against caches, e.g. "
                             "lexical=dl19_lexical.pkl embedding=dl19_embedding.pkl "
                             "query_type=dl19_query_type.pkl. Not timed.")
    parser.add_argument("--check_n", type=int, default=20)
    parser.add_argument(
        "--num_feature_samples", type=int, default=10,
        help="Save the live lexical/embedding/query_type values (the exact vectors fed to "
             "the MLP) of this many random queries to "
             "<output_dir>/feature_samples_eval_<dataset>.json for sanity-checking "
             "(default 10; 0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for picking the --num_feature_samples queries (default 42).")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()

    features_path = args.model_path + ".features.json"
    if args.features is not None:
        features = args.features
    elif os.path.exists(features_path):
        with open(features_path) as f:
            features = json.load(f)
    else:
        parser.error("No --features given and no <model_path>.features.json found.")
    try:
        blocks = resolve_live_blocks(features)
    except ValueError as e:
        parser.error(str(e))

    ranker_path = args.ranker_map or args.model_path + ".rankers.json"
    if not os.path.exists(ranker_path):
        parser.error(f"Ranker map not found: {ranker_path} (pass --ranker_map).")
    with open(ranker_path) as f:
        num_rankers = len(json.load(f))

    if "lexical" in blocks and not args.index:
        parser.error("--index is required when 'lexical' is in the checkpoint's features.")

    queries = load_queries(args.queries)
    if args.max_queries:
        queries = dict(list(queries.items())[: args.max_queries])
    if not queries:
        parser.error(f"No queries loaded from {args.queries}.")

    # ---- Pre-load everything (timed only as metadata, never per-query) ----
    load_s: Dict[str, float] = {}
    index_stats = encoder = classifier = None
    if "lexical" in blocks:
        from features import IndexStats

        with Timer() as t:
            index_stats = IndexStats(args.index)
        load_s["index_load_s"] = t.seconds
    embedding_dim = None
    if "embedding" in blocks:
        with Timer(device) as t:
            encoder = MiniLMEncoder(device=device).load()
        load_s["minilm_load_s"] = t.seconds
        embedding_dim = int(encoder.encode([WARMUP_QUERY]).shape[1])
    if "query_type" in blocks:
        with Timer(device) as t:
            classifier = make_query_type_classifier(device)
        load_s["query_type_load_s"] = t.seconds
    with Timer(device) as t:
        model = build_model(blocks, embedding_dim, num_rankers, args, device)
    load_s["mlp_load_s"] = t.seconds

    print(f"[{args.dataset}] timing {len(queries)} queries live "
          f"(blocks={list(blocks)}, device={device})...")
    sample_qids = pick_sample_qids(queries, args.num_feature_samples, args.seed) \
        if args.num_feature_samples > 0 else []
    rows, live_vectors = time_queries(
        args.dataset, queries, blocks, index_stats, encoder, classifier, model,
        device, num_rankers, args.top_k, keep_idf_cache=args.keep_idf_cache,
        warmup=args.warmup,
        keep_vectors_for=args.check_n if args.check_against_cache else 0,
        keep_qids=sample_qids,
    )

    out_csv = os.path.join(args.output_dir, f"eval_timing_{args.dataset}.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)  # one fresh CSV per run, not appended across runs
    append_rows_csv(out_csv, CSV_HEADER, rows)
    write_env_json(os.path.splitext(out_csv)[0] + ".env.json", device, {
        "dataset": args.dataset, "model_path": args.model_path, "blocks": list(blocks),
        "n_queries": len(rows), "batch_size": 1, "warmup_runs": args.warmup,
        "idf_cache_cleared_per_query": not args.keep_idf_cache,
        "model_load_seconds": {k: round(v, 4) for k, v in load_s.items()},
    })

    total = np.array([float(r["total_s"]) for r in rows])
    print(f"  total_s per query: mean={total.mean():.4f} median={np.median(total):.4f} "
          f"p95={np.percentile(total, 95):.4f}")
    print(f"CSV -> {out_csv}")

    # After the whole timing loop: describe_query does IDF lookups.
    if sample_qids:
        samples = build_eval_samples(queries, live_vectors, sample_qids, index_stats)
        samples_path = os.path.join(args.output_dir, f"feature_samples_eval_{args.dataset}.json")
        write_feature_samples(samples_path, "eval", args.dataset, args.seed, samples, {
            "blocks": list(blocks), "model_path": args.model_path, "n_queries": len(queries),
        })
        print(f"  saved {len(samples)} feature samples -> {samples_path}")

    if args.check_against_cache:
        cache_paths = dict(spec.split("=", 1) for spec in args.check_against_cache)
        first_n = set(list(queries)[: args.check_n])  # not the extra feature-sample qids
        check_against_caches(
            {q: v for q, v in live_vectors.items() if q in first_n}, cache_paths,
        )


if __name__ == "__main__":
    main()
