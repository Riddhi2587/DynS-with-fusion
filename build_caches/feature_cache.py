"""
Precomputed caches of the pre-retrieval features from
features.py. QPPQueryOnlyDataset (dataset.py) merges up to three SEPARATE
per-qid caches - lexical (5-dim, features.build_query_features), embedding
(raw, variable-width - 768 for BERT/Contriever, 384 for MiniLM, etc.,
features.build_embedding_feature), and query_type (1-dim,
features.build_query_type_feature) - at dataset construction time, rather
than one combined cache. This lets each piece be built/rebuilt independently:
the lexical cache needs a Lucene index (index_stats), while the embedding and
query_type caches don't need Lucene at all, only a raw embedding source (e.g.
data/cache/bert-query-embeddings/cls/*.cls.pkl) / the classifier respectively.

FeatureCache is a generic container reused by all three builders below, each
a separate, independently-rebuildable single-purpose cache file
(--lexical_cache/--embedding_cache/--query_type_cache).

Usage:
    lexical_cache = build_feature_cache(run_paths, queries, index_stats)
    embedding_lookup = load_query_embeddings("data/cache/bert-query-embeddings/cls/dl19.cls.pkl")
    embedding_cache = build_embedding_cache(queries, embedding_lookup)
    query_type_cache = build_query_type_cache(run_paths, queries, QueryTypeClassifier())
    save_feature_cache(lexical_cache, "dl19_lexical.pkl")
    ...
    cache = load_feature_cache("dl19_lexical.pkl")
"""

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from features import (
    QueryTypeClassifier,
    build_embedding_feature,
    build_query_features,
    build_query_type_feature,
)


@dataclass
class FeatureCache:
    query_feats: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def load_query_embeddings(
    path: str, expected_dim: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Load a precomputed {qid: raw embedding vector} pkl (see e.g.
    data/cache/bert-query-embeddings/cls/*.cls.pkl). The raw width is not
    fixed - it depends on the embedding source (768 for BERT/Contriever, 384
    for MiniLM, etc.) - so it's inferred from the first entry and every other
    entry is validated to match it, catching a malformed/mixed cache file at
    load time rather than surfacing as a confusing shape mismatch deep inside
    model training. Pass `expected_dim` for an additional explicit sanity
    check against a known width; leave it None to just infer-and-validate
    consistency."""
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    dim = expected_dim
    for qid, vec in embeddings.items():
        vec = np.asarray(vec)
        if dim is None:
            dim = int(vec.shape[0])
        if vec.shape != (dim,):
            raise ValueError(
                f"Query embedding for qid={qid!r} in {path!r} has shape "
                f"{vec.shape}, expected ({dim},) "
                f"({'explicit expected_dim' if expected_dim is not None else 'inferred from another entry in this file'})."
            )
    return embeddings


def build_feature_cache(
    run_paths: List[str],
    queries: Dict[str, str],
    index_stats,
) -> FeatureCache:
    """
    Precompute every qid's 5-dim lexical/IDF query feature vector
    (features.build_query_features) over all (ranker, qid) pairs in
    run_paths, intersected with queries. Not qrels-filtered, so the cache
    stays valid even if qrels change later.
    """
    from dataset import load_run  # deferred: dataset.py imports this module

    runs = load_run(run_paths)

    query_feats: Dict[str, np.ndarray] = {}
    for run_data in runs.values():
        for qid in run_data:
            if qid in query_feats or qid not in queries:
                continue
            query_terms = queries[qid].lower().split()
            if not query_terms:
                continue
            query_feats[qid] = build_query_features(query_terms, index_stats)

    meta = {"num_qids": len(query_feats), "built_from": list(run_paths)}
    return FeatureCache(query_feats=query_feats, meta=meta)


def build_embedding_cache(
    queries: Dict[str, str],
    embedding_lookup: Dict[str, np.ndarray],
) -> FeatureCache:
    """
    Precompute every qid's raw, L2-normalized query representation feature
    (features.build_embedding_feature) for every qid in queries. Like
    query_type, this needs no run files - it's pure lookup + L2-normalize, no
    dependency on run-file content (no Lucene index either - only the
    precomputed embedding_lookup, see load_query_embeddings).

    Records the inferred raw width in meta["embedding_dim"] so downstream
    code (dataset.py) can read it directly instead of re-scanning
    query_feats to find it.
    """
    query_feats: Dict[str, np.ndarray] = {
        qid: build_embedding_feature(qid, embedding_lookup) for qid in queries
    }

    embedding_dim = int(next(iter(query_feats.values())).shape[0]) if query_feats else 0
    meta = {"num_qids": len(query_feats), "embedding_dim": embedding_dim}
    return FeatureCache(query_feats=query_feats, meta=meta)


def build_query_type_cache(
    queries: Dict[str, str],
    query_type_classifier: QueryTypeClassifier,
) -> FeatureCache:
    """
    Precompute every qid's 1-dim query_type feature
    (features.build_query_type_feature) for every qid in queries. Unlike the
    lexical/embedding caches, this needs no run files at all - query_type is
    a pure function of the raw query text, so there's nothing to scope down
    to (no Lucene index either - only the classifier).
    """
    query_feats: Dict[str, np.ndarray] = {
        qid: build_query_type_feature(raw_query, query_type_classifier)
        for qid, raw_query in queries.items()
    }

    meta = {"num_qids": len(query_feats)}
    return FeatureCache(query_feats=query_feats, meta=meta)


def save_feature_cache(cache: FeatureCache, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_feature_cache(path: str) -> FeatureCache:
    with open(path, "rb") as f:
        cache = pickle.load(f)
    parts = []
    if cache.query_feats:
        parts.append(f"{cache.meta.get('num_qids', len(cache.query_feats))} qids")
    print(f"Loaded feature cache from {path}: {', '.join(parts) if parts else 'empty'}")
    return cache
