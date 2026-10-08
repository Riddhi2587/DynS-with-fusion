import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from build_caches.feature_cache import FeatureCache
from features import (
    LEXICAL_FEATURE_DIM,
    LEXICAL_FEATURE_NAMES,
    QUERY_TYPE_FEATURE_DIM,
    QUERY_TYPE_FEATURE_NAMES,
    IndexStats,
    QueryTypeClassifier,
    build_embedding_feature,
    build_query_features,
    build_query_type_feature,
    embedding_feature_names,
)

def _canonical_run_name(path: str) -> str:
    """Strip year tokens so train/eval stems match (e.g. BM25.2019.100 → BM25.100)."""
    stem = Path(path).stem
    stem = re.sub(r'\.(20\d{2})', '', stem)       # .2019 / .2020
    stem = re.sub(r'_(1\d|2\d)(?=\.|$)', '', stem)  # _19 / _20
    return stem


def load_run(run_paths: List[str]) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Parse one or more TREC-format run files.

    Returns:
        {run_name: {qid: [(docid, score), ...]}} sorted by score descending.
    """
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run_path in run_paths:
        run_name = _canonical_run_name(run_path)
        with open(run_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                qid, _, docid, _rank, score, _ = parts[:6]
                runs[run_name][qid].append((docid, float(score)))

    for run_name in runs:
        for qid in runs[run_name]:
            runs[run_name][qid].sort(key=lambda x: -x[1])

    return dict(runs)


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Parse a TREC qrels file. Returns {qid: {docid: relevance}}."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            qrels[qid][docid] = int(rel)
    return dict(qrels)


@lru_cache(maxsize=None)
def load_precomputed_metrics(csv_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load a precomputed per-(ranker, qid) metrics CSV (columns: qid, ranker,
    plus one column per pytrec_eval metric key, e.g. ndcg_cut_100) into a
    {(ranker, qid): {metric_key: value}} lookup. Cached per path so repeated
    calls (once per metric, once per checkpoint) only hit disk once.

    See guide_docs/PRECOMPUTED_METRICS_GUIDE.md."""
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        metric_cols = [c for c in reader.fieldnames if c not in ("qid", "ranker")]
        for row in reader:
            lookup[(row["ranker"], row["qid"])] = {c: float(row[c]) for c in metric_cols}
    if not lookup:
        raise ValueError(f"No rows loaded from metrics CSV: {csv_path}")
    return lookup


def _require_metric_column(
    lookup: Dict[Tuple[str, str], Dict[str, float]], metric_key: str, csv_path: str
) -> None:
    """Raise a clear error if metric_key is missing from a loaded metrics
    CSV's columns entirely - almost always a wrong CSV path/wrong dataset,
    unlike a single missing (ranker, qid) row (silently resolves to 0.0,
    per guide_docs/PRECOMPUTED_METRICS_GUIDE.md's documented behavior)."""
    available = next(iter(lookup.values()))
    if metric_key not in available:
        raise ValueError(
            f"Metrics CSV {csv_path!r} has no {metric_key!r} column. "
            f"Available columns: {sorted(available.keys())}."
        )


def build_ranker_to_id(
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    ranker_to_id: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Ranker (run) name -> integer id, consistent across train/eval.

    Built from sorted run names unless an explicit map is injected (used at
    eval time to reuse the training map so that a ranker's column index
    means the same ranker in both). Raises if `runs` contains a ranker not
    present in an explicitly provided map, since a silently-added column
    would desync labels/predictions from the ranker they belong to.
    """
    if ranker_to_id is None:
        return {rn: i for i, rn in enumerate(sorted(runs.keys()))}
    ranker_to_id = dict(ranker_to_id)
    for rn in runs:
        if rn not in ranker_to_id:
            raise ValueError(
                f"Ranker '{rn}' is not in the provided ranker_to_id map. "
                f"Train and eval must share the same rankers (same run-file "
                f"names) for ranker columns to line up."
            )
    return ranker_to_id


class QPPQueryOnlyDataset(Dataset):
    """
    One sample = one query: the query feature vector u_q plus per-ranker
    nDCG@100 labels, matching how QueryOnlyMLP (model.py) consumes them.

    Every query in this data has all `num_rankers` rankers present, so
    there is no per-ranker "absent for this query" case to track - no
    ranker_mask.

    Returns per __getitem__:
        query_feats : (query_feature_dim,)   float32
        labels      : (num_rankers,)         float32  nDCG@100

    ranker_to_id: ranker (run) name -> column id (see build_ranker_to_id);
    must match across train/eval so ranker columns mean the same ranker in
    both.

    lexical_cache, embedding_cache, query_type_cache:
        Three SEPARATE, independently optional `feature_cache.FeatureCache`
        instances (see `feature_cache.py`) - the 5-dim lexical/IDF vector,
        the raw (variable-width, e.g. 768/384) query representation, and the
        1-dim query_type flag are looked up from whichever of these caches
        has them, and concatenated (in that order) into `query_feats` per
        query. Each is independent: e.g. you can have a real precomputed
        `lexical_cache` (no Lucene needed at dataset-build time) while
        computing embedding/query_type live, or any other mix. The
        embedding block's raw width is inferred once (from
        `embedding_cache.meta["embedding_dim"]`, or from `embedding_lookup`)
        and exposed as `self.embedding_slice`/`self.query_feature_dim`/
        `self.query_feature_names` - never hardcoded.

    index_stats, embedding_lookup, query_type_classifier:
        The live-computation fallback for a cache miss, one per piece:
        `index_stats` (features.build_query_features) backs a
        `lexical_cache` miss, `embedding_lookup` (a {qid: raw embedding
        vector} dict, variable width, see `feature_cache.load_query_embeddings`,
        via features.build_embedding_feature) backs an `embedding_cache`
        miss, and `query_type_classifier` (a `features.QueryTypeClassifier`,
        via features.build_query_type_feature) backs a `query_type_cache`
        miss. Each is optional when its corresponding cache fully covers
        this dataset's queries; if a miss occurs on a piece with no
        fallback available, a RuntimeError is raised naming the missing qid
        rather than silently failing.

    metrics_csv:
        Path to a precomputed per-(ranker, qid) metrics CSV (see
        `load_precomputed_metrics` and guide_docs/PRECOMPUTED_METRICS_GUIDE.md)
        - the nDCG@100 label is looked up from it. Required: there is no
        pytrec_eval fallback (supply a precomputed one).
    """

    def __init__(
        self,
        run_paths: List[str],
        qrels_path: str,
        queries: Dict[str, str],
        index_stats: Optional[IndexStats] = None,
        ranker_to_id: Optional[Dict[str, int]] = None,
        lexical_cache: Optional[FeatureCache] = None,
        embedding_cache: Optional[FeatureCache] = None,
        query_type_cache: Optional[FeatureCache] = None,
        embedding_lookup: Optional[Dict[str, np.ndarray]] = None,
        query_type_classifier: Optional[QueryTypeClassifier] = None,
        metrics_csv: Optional[str] = None,
    ):
        if metrics_csv is None:
            raise ValueError(
                "QPPQueryOnlyDataset requires metrics_csv (a precomputed "
                "per-(ranker, qid) metrics CSV) - there is no pytrec_eval "
                "fallback. Supply a precomputed one."
            )
        self.samples: List[dict] = []
        groups: Dict[str, dict] = {}
        # Per-piece cache hit tracking (lexical / embedding / query_type),
        # since each has its own independent cache and fallback.
        cache_hits = {"lexical": 0, "embedding": 0, "query_type": 0}
        cache_total = 0

        runs = load_run(run_paths)
        qrels = load_qrels(qrels_path)

        self.ranker_to_id = build_ranker_to_id(runs, ranker_to_id)
        self.num_rankers = len(self.ranker_to_id)

        metrics_lookup = load_precomputed_metrics(metrics_csv)
        _require_metric_column(metrics_lookup, "ndcg_cut_100", metrics_csv)

        # Raw embedding width isn't fixed (768 for BERT/Contriever, 384 for
        # MiniLM, etc.) - infer it once, and
        # use it to fix this dataset's (fixed lexical+embedding+query_type)
        # legacy layout, replacing the old hardcoded QUERY_FEATURE_DIM.
        if embedding_cache is not None and "embedding_dim" in embedding_cache.meta:
            embedding_dim = embedding_cache.meta["embedding_dim"]
        elif embedding_lookup:
            embedding_dim = int(np.asarray(next(iter(embedding_lookup.values()))).shape[0])
        else:
            raise ValueError(
                "QPPQueryOnlyDataset couldn't determine the raw embedding "
                "width - embedding_cache has no 'embedding_dim' in its meta "
                "and no embedding_lookup was provided to infer it from."
            )
        self.embedding_slice = slice(LEXICAL_FEATURE_DIM, LEXICAL_FEATURE_DIM + embedding_dim)
        self.query_feature_dim = LEXICAL_FEATURE_DIM + embedding_dim + QUERY_TYPE_FEATURE_DIM
        self.query_feature_names = (
            LEXICAL_FEATURE_NAMES + embedding_feature_names(embedding_dim) + QUERY_TYPE_FEATURE_NAMES
        )

        for run_name, run_data in runs.items():
            ranker_id = self.ranker_to_id[run_name]

            for qid in run_data:
                if qid not in qrels or qid not in queries:
                    continue

                raw_query = queries[qid]
                query_terms = raw_query.lower().split()
                if not query_terms:
                    continue

                label = metrics_lookup.get((run_name, qid), {}).get("ndcg_cut_100", 0.0)

                if qid not in groups:
                    cache_total += 1

                    if lexical_cache is not None and qid in lexical_cache.query_feats:
                        lexical_vec = lexical_cache.query_feats[qid]
                        cache_hits["lexical"] += 1
                    elif index_stats is not None:
                        lexical_vec = build_query_features(query_terms, index_stats)
                    else:
                        raise RuntimeError(
                            f"No lexical_cache entry for qid={qid!r} and no "
                            f"index_stats fallback was provided. Rebuild the "
                            f"lexical feature cache to cover this query, or "
                            f"pass --index."
                        )

                    if embedding_cache is not None and qid in embedding_cache.query_feats:
                        embedding_vec = embedding_cache.query_feats[qid]
                        cache_hits["embedding"] += 1
                    elif embedding_lookup is not None:
                        embedding_vec = build_embedding_feature(qid, embedding_lookup)
                    else:
                        raise RuntimeError(
                            f"No embedding_cache entry for qid={qid!r} and no "
                            f"embedding_lookup fallback was provided. Rebuild "
                            f"the embedding feature cache to cover this query, "
                            f"or pass --query_embeddings."
                        )

                    if query_type_cache is not None and qid in query_type_cache.query_feats:
                        query_type_vec = query_type_cache.query_feats[qid]
                        cache_hits["query_type"] += 1
                    elif query_type_classifier is not None:
                        query_type_vec = build_query_type_feature(raw_query, query_type_classifier)
                    else:
                        raise RuntimeError(
                            f"No query_type_cache entry for qid={qid!r} and no "
                            f"query_type_classifier fallback was provided. "
                            f"Rebuild the query_type feature cache to cover "
                            f"this query, or pass a query_type_classifier."
                        )

                    query_feats = np.concatenate([lexical_vec, embedding_vec, query_type_vec])
                    groups[qid] = {
                        "qid": qid,
                        "query_feats": query_feats,
                        "labels": np.zeros(self.num_rankers, dtype=np.float32),
                    }
                g = groups[qid]
                g["labels"][ranker_id] = label

        self.samples = list(groups.values())

        if cache_total > 0 and (lexical_cache or embedding_cache or query_type_cache):
            print(
                "  feature cache hits: "
                f"lexical={cache_hits['lexical']}/{cache_total} "
                f"embedding={cache_hits['embedding']}/{cache_total} "
                f"query_type={cache_hits['query_type']}/{cache_total}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["query_feats"]),   # (Q,)
            torch.from_numpy(s["labels"]),         # (R,)
        )


def _feature_col_stats(col: np.ndarray) -> dict:
    if len(col) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "nonzero": 0, "n": 0}
    return {
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": float(col.mean()),
        "nonzero": int(np.count_nonzero(col)),
        "n": int(len(col)),
    }


def _feature_block_report(
    query_rows: List[np.ndarray],
    query_feature_names: List[str],
) -> dict:
    """Per-feature min/max/mean/nonzero for the query feature rows.
    query_feature_names must be passed explicitly - the embedding block's raw
    width (and therefore its names) is variable, not fixed."""
    query_arr = (
        np.stack(query_rows) if query_rows else np.zeros((0, len(query_feature_names)), dtype=np.float32)
    )
    query_stats = {name: _feature_col_stats(query_arr[:, i]) for i, name in enumerate(query_feature_names)}
    return {
        "n_query_rows": len(query_rows),
        "query_features": query_stats,
    }


def build_query_only_feature_sanity_report(dataset: "QPPQueryOnlyDataset") -> dict:
    """
    Summarize the query feature tensors a built QPPQueryOnlyDataset holds --
    per-feature min/max/mean/nonzero-count across all samples, plus the
    tensor shape. Same convention as build_feature_sanity_report: shape/names
    are read off the dataset instance (dataset.query_feature_dim/
    dataset.query_feature_names), not fixed global constants, since the
    embedding block's raw width varies by source.
    """
    query_rows = [s["query_feats"] for s in dataset.samples]
    return {
        "num_samples": len(dataset.samples),
        "num_rankers": dataset.num_rankers,
        "query_feats_shape": [dataset.query_feature_dim],
        "query_feature_names": dataset.query_feature_names,
        "branches": {
            "original": _feature_block_report(
                query_rows, query_feature_names=dataset.query_feature_names,
            )
        },
    }
