"""
Time how long it takes to COMPUTE the three pre-retrieval feature blocks
(lexical/IDF, query_type, MiniLM embedding) for a dataset - i.e. what building
its feature caches costs. This is a timing-only run: it does not touch any
existing cache (train.py keeps consuming the already-built ones), and writes
cache pickles only if --save_caches_dir is given.

Appends rows to <output_dir>/train_timing_summary.csv (dataset, step, n_items,
seconds, mean_per_item_s, device) - the same schema train.py uses for its own
timing rows. Steps:
    index_load / query_type_model_load / minilm_model_load   model load (once)
    run_parse                     parsing the run files (excluded from lexical)
    lexical                       sum of per-qid tokenize + build_query_features
    query_type                    build_query_type_cache path (one query at a time,
                                  exactly like build_query_type_cache.py)
    query_type_batched            same classifier, batched (--batch_size)
    embedding_encode              MiniLM encode over every qid (--batch_size)
    embedding_normalize           build_embedding_cache (L2-normalize) step
    embedding_total               encode + normalize

--num_queries NAME=N times a random sample of N queries for that dataset instead
of all of them (seeded by --seed; the sampled qids are written to
<output_dir>/sampled_qids_<NAME>.txt). The sample is drawn from qids that have
non-empty text and appear in a run file, so lexical, query_type and embedding
are all timed on the SAME N queries.

--num_feature_samples N (default 10) also saves the computed lexical/embedding/
query_type values of N random timed queries to
<output_dir>/feature_samples_train_<NAME>.json so the computation can be checked
by eye (see feature_samples.py); 0 disables. Taken from the vectors already
computed, after timing, so it doesn't affect the timings.

Example (100 random llm-judged queries, 300 random msmarco-dev queries):
    python time_feature_computation.py \
        --dataset msmarco-dev data/msmarco-dev.queries data/msmarco-dev-runs/*.res \
        --dataset llm-judged data/llm-judged.queries data/llm-judged-runs/*.res \
        --index /path/to/msmarco-passage-index \
        --num_queries llm-judged=100 msmarco-dev=300 --seed 42 \
        --output_dir timing_results
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np

from dataset import load_run
from embedding_live import MiniLMEncoder, default_device, make_query_type_classifier
from feature_cache import (
    FeatureCache,
    build_embedding_cache,
    build_query_type_cache,
    save_feature_cache,
)
from feature_samples import describe_query, pick_sample_qids, write_feature_samples
from features import build_query_features
from timing_utils import (
    SUMMARY_HEADER,
    Timer,
    append_rows_csv,
    summary_row,
    write_env_json,
)
from train import load_queries

ALL_STEPS = ("lexical", "query_type", "embedding")
WARMUP_QUERY = "warm up query"


def clear_idf_cache(index_stats) -> None:
    """Cold-start the IDF memo so lexical time doesn't depend on earlier
    queries/datasets. (Lucene/JVM-internal caches can't be cleared from here.)"""
    cache = getattr(index_stats, "_idf_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def time_lexical(
    queries: Dict[str, str], runs, index_stats,
) -> "tuple[Dict[str, np.ndarray], float, int]":
    """Per-qid tokenize + build_query_features, timed individually and summed
    (CPU-only work, so no CUDA sync). Same qid selection as
    feature_cache.build_feature_cache: qids present in some run file and in
    `queries`, skipping empty queries. Returns (query_feats, total_seconds,
    n_qids)."""
    query_feats: Dict[str, np.ndarray] = {}
    total = 0.0
    for run_data in runs.values():
        for qid in run_data:
            if qid in query_feats or qid not in queries:
                continue
            raw = queries[qid]
            with Timer() as t:
                query_terms = raw.lower().split()
                vec = build_query_features(query_terms, index_stats) if query_terms else None
            if vec is None:
                continue
            total += t.seconds
            query_feats[qid] = vec
    return query_feats, total, len(query_feats)


def time_dataset(
    name: str,
    queries: Dict[str, str],
    runs,
    index_stats,
    classifier,
    encoder,
    device,
    batch_size: int = 32,
    keep_idf_cache: bool = False,
    steps=ALL_STEPS,
):
    """Times the selected steps for one dataset. Returns (rows, caches) where
    rows are SUMMARY_HEADER dicts and caches is {"lexical"|"query_type"|
    "embedding": FeatureCache} for whatever was computed."""
    rows: List[Dict] = []
    caches: Dict[str, FeatureCache] = {}
    qids = list(queries.keys())
    texts = [queries[q] for q in qids]

    if "lexical" in steps:
        if not keep_idf_cache:
            clear_idf_cache(index_stats)
        feats, seconds, n = time_lexical(queries, runs, index_stats)
        rows.append(summary_row(name, "lexical", seconds, n, "cpu"))  # Lucene is CPU-only
        caches["lexical"] = FeatureCache(query_feats=feats, meta={"num_qids": n})

    if "query_type" in steps:
        with Timer(device) as t:
            qt_cache = build_query_type_cache(queries, classifier)
        rows.append(summary_row(name, "query_type", t.seconds, len(qids), device))
        caches["query_type"] = qt_cache

        with Timer(device) as t:
            for start in range(0, len(texts), batch_size):
                classifier.batch_classify(texts[start:start + batch_size])
        rows.append(summary_row(name, "query_type_batched", t.seconds, len(qids), device))

    if "embedding" in steps:
        with Timer(device) as t_enc:
            vecs = encoder.encode(texts, batch_size=batch_size)
        lookup = {qid: vec for qid, vec in zip(qids, vecs)}
        with Timer(device) as t_norm:
            emb_cache = build_embedding_cache(queries, lookup)
        rows.append(summary_row(name, "embedding_encode", t_enc.seconds, len(qids), device))
        rows.append(summary_row(name, "embedding_normalize", t_norm.seconds, len(qids), device))
        rows.append(summary_row(
            name, "embedding_total", t_enc.seconds + t_norm.seconds, len(qids), device,
        ))
        caches["embedding"] = emb_cache

    return rows, caches


def sample_queries(
    name: str, queries: Dict[str, str], runs, n: int, seed: int,
) -> "tuple[Dict[str, str], int]":
    """Random sample of n queries, returned in sample order along with the pool
    size. The pool is qids with non-empty text that appear in at least one run
    file - the same qids the lexical step can cover - so every step is timed on
    identical queries. Sorted before sampling and seeded per call, so the result
    doesn't depend on file/dict order or on which other datasets are in the run.
    Raises ValueError if the pool has fewer than n qids."""
    in_runs = set()
    for run_data in runs.values():
        in_runs.update(run_data)
    pool = sorted(qid for qid, text in queries.items() if qid in in_runs and text.split())
    if len(pool) < n:
        raise ValueError(
            f"Dataset {name!r}: asked for {n} random queries but only {len(pool)} "
            f"queries have non-empty text and appear in a run file."
        )
    chosen = random.Random(seed).sample(pool, n)
    return {qid: queries[qid] for qid in chosen}, len(pool)


def build_train_samples(
    queries: Dict[str, str], caches: Dict[str, FeatureCache], index_stats, n: int, seed: int,
) -> List[Dict]:
    """Descriptions (feature_samples.describe_query) of up to n random queries,
    using the vectors time_dataset already computed - nothing is recomputed. Draws
    from qids present in every cache that was computed. Call after all timing for
    the dataset is finished, since describe_query does IDF lookups."""
    if not caches:
        return []
    qid_sets = [set(c.query_feats) for c in caches.values()]
    eligible = set.intersection(*qid_sets) & set(queries)
    samples = []
    for qid in pick_sample_qids(eligible, n, seed):
        vec = lambda kind: caches[kind].query_feats[qid] if kind in caches else None
        samples.append(describe_query(
            qid, queries[qid], vec("lexical"), vec("embedding"), vec("query_type"), index_stats,
        ))
    return samples


def parse_num_queries(specs: Optional[List[str]], dataset_names) -> Dict[str, int]:
    """Parse --num_queries NAME=N [NAME=N ...] into {name: N}. Raises ValueError on
    a malformed entry, a non-positive/non-integer N, a duplicate name, or a name
    that isn't one of the datasets being timed."""
    out: Dict[str, int] = {}
    for spec in specs or []:
        name, sep, value = spec.partition("=")
        if not sep or not name:
            raise ValueError(f"--num_queries expects NAME=N, got {spec!r}.")
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"--num_queries {spec!r}: {value!r} is not an integer.") from None
        if n <= 0:
            raise ValueError(f"--num_queries {spec!r}: N must be positive.")
        if name not in dataset_names:
            raise ValueError(
                f"--num_queries names unknown dataset {name!r}; datasets are {sorted(dataset_names)}."
            )
        if name in out:
            raise ValueError(f"--num_queries lists dataset {name!r} more than once.")
        out[name] = n
    return out


def _parse_datasets(args) -> List[dict]:
    datasets: List[dict] = []
    for spec in args.dataset or []:
        if len(spec) < 3:
            raise SystemExit("--dataset needs: NAME QUERIES RUN [RUN ...]")
        datasets.append({"name": spec[0], "queries": spec[1], "runs": spec[2:]})
    if args.datasets_config:
        with open(args.datasets_config) as f:
            datasets.extend(json.load(f))
    if not datasets:
        raise SystemExit("Give at least one --dataset or a --datasets_config.")
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", action="append", nargs="+", metavar="ARG",
        help="NAME QUERIES RUN [RUN ...] (repeatable). Alternatively use "
             "--datasets_config.",
    )
    parser.add_argument(
        "--datasets_config", default=None,
        help="JSON list of {\"name\", \"queries\", \"runs\": [...], optional \"index\", "
             "optional \"num_queries\"}.",
    )
    parser.add_argument(
        "--num_queries", nargs="+", metavar="NAME=N", default=None,
        help="Time a random sample of N queries for the named dataset(s) instead of "
             "all of them, e.g. llm-judged=100 msmarco-dev=300. Overrides a "
             "dataset's \"num_queries\" in --datasets_config; datasets not listed "
             "use all their queries.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for --num_queries sampling and for picking the "
                             "--num_feature_samples queries (default 42).")
    parser.add_argument(
        "--num_feature_samples", type=int, default=10,
        help="Save the computed lexical/embedding/query_type values of this many random "
             "timed queries per dataset to <output_dir>/feature_samples_train_<NAME>.json "
             "for sanity-checking (default 10; 0 disables).",
    )
    parser.add_argument("--index", default=None,
                        help="Default Lucene index (needed for the lexical step); a "
                             "dataset's own \"index\" in --datasets_config overrides it.")
    parser.add_argument("--steps", nargs="+", default=list(ALL_STEPS), choices=list(ALL_STEPS))
    parser.add_argument("--output_dir", default="timing_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu.")
    parser.add_argument("--keep_idf_cache", action="store_true",
                        help="Don't clear the IDF memo before each dataset's lexical step.")
    parser.add_argument("--save_caches_dir", default=None,
                        help="OPT-IN: also write the computed caches here as "
                             "<name>_{lexical,query_type,embedding}.pkl. Off by default.")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device) if args.device else default_device()
    datasets = _parse_datasets(args)
    steps = tuple(args.steps)
    summary_csv = os.path.join(args.output_dir, "train_timing_summary.csv")

    try:
        cli_n = parse_num_queries(args.num_queries, {ds["name"] for ds in datasets})
    except ValueError as e:
        raise SystemExit(str(e))
    n_by_dataset = {
        ds["name"]: cli_n.get(ds["name"], ds.get("num_queries")) for ds in datasets
    }
    if args.save_caches_dir and any(n_by_dataset.values()):
        raise SystemExit(
            "--save_caches_dir can't be combined with --num_queries: a cache built "
            "from a random subset would look like a real cache but be partial. "
            "Build training caches with the build_*_cache.py scripts."
        )

    load_rows: List[Dict] = []
    classifier = encoder = None
    if "query_type" in steps:
        with Timer(device) as t:
            classifier = make_query_type_classifier(device)
        load_rows.append(summary_row("_shared", "query_type_model_load", t.seconds, None, device))
        classifier.batch_classify([WARMUP_QUERY])
    if "embedding" in steps:
        with Timer(device) as t:
            encoder = MiniLMEncoder(device=device).load()
        load_rows.append(summary_row("_shared", "minilm_model_load", t.seconds, None, device))
        encoder.encode([WARMUP_QUERY])
    append_rows_csv(summary_csv, SUMMARY_HEADER, load_rows)

    index_cache: Dict[str, object] = {}
    sampling: Dict[str, dict] = {}
    for ds in datasets:
        name = ds["name"]
        queries = load_queries(ds["queries"])
        index_stats = None
        rows: List[Dict] = []
        if "lexical" in steps:
            from features import IndexStats

            index_path = ds.get("index") or args.index
            if not index_path:
                raise SystemExit(f"Dataset {name!r}: the lexical step needs --index (or a per-dataset 'index').")
            if index_path not in index_cache:
                with Timer() as t:
                    index_cache[index_path] = IndexStats(index_path)
                rows.append(summary_row(name, "index_load", t.seconds, None, "cpu"))
            index_stats = index_cache[index_path]
        # Runs are always parsed: the lexical step needs them, and sampling draws
        # from qids that appear in them. run_parse is only reported when lexical
        # runs, since that's the only step it's part of.
        with Timer() as t:
            runs = load_run(ds["runs"])
        if "lexical" in steps:
            rows.append(summary_row(name, "run_parse", t.seconds, len(ds["runs"]), "cpu"))

        n = n_by_dataset[name]
        if n:
            try:
                queries, pool_size = sample_queries(name, queries, runs, n, args.seed)
            except ValueError as e:
                raise SystemExit(str(e))
            os.makedirs(args.output_dir, exist_ok=True)
            qids_path = os.path.join(args.output_dir, f"sampled_qids_{name}.txt")
            with open(qids_path, "w") as f:
                f.write("\n".join(queries) + "\n")
            sampling[name] = {"num_queries": n, "pool_size": pool_size, "sampled_qids_file": qids_path}
            print(f"[{name}] sampled {n} of {pool_size} eligible queries (seed={args.seed}) -> {qids_path}")

        print(f"[{name}] timing {steps} over {len(queries)} queries on {device}...")
        ds_rows, caches = time_dataset(
            name, queries, runs, index_stats, classifier, encoder, device,
            batch_size=args.batch_size, keep_idf_cache=args.keep_idf_cache, steps=steps,
        )
        rows.extend(ds_rows)
        append_rows_csv(summary_csv, SUMMARY_HEADER, rows)
        for r in ds_rows:
            print(f"  {r['step']:<22} {r['seconds']:>12}s  n={r['n_items']}  "
                  f"mean={r['mean_per_item_s']}s  [{r['device']}]")

        # After ALL timing for this dataset: describe_query does IDF lookups.
        if args.num_feature_samples > 0:
            samples = build_train_samples(
                queries, caches, index_stats, args.num_feature_samples, args.seed,
            )
            samples_path = os.path.join(args.output_dir, f"feature_samples_train_{name}.json")
            write_feature_samples(samples_path, "train", name, args.seed, samples, {
                "blocks": sorted(caches),
                "num_timed_queries": len(queries),
            })
            print(f"  saved {len(samples)} feature samples -> {samples_path}")

        if args.save_caches_dir:
            os.makedirs(args.save_caches_dir, exist_ok=True)
            for kind, cache in caches.items():
                out = os.path.join(args.save_caches_dir, f"{name}_{kind}.pkl")
                save_feature_cache(cache, out)
                print(f"  saved {kind} cache -> {out}")

    write_env_json(summary_csv + ".env.json", device, {
        "batch_size": args.batch_size,
        "idf_cache_cleared_per_dataset": not args.keep_idf_cache,
        "steps": list(steps),
        "seed": args.seed,
        "sampling": sampling,  # {} when every dataset used all its queries
    })
    print(f"Summary -> {summary_csv}")


if __name__ == "__main__":
    main()
