"""
Precomputed caches of the pre-retrieval/doc-content features from
features.py. QPPQueryOnlyDataset (dataset.py) merges up to three SEPARATE
per-qid caches - lexical (5-dim, features.build_query_features), embedding
(raw, variable-width - 768 for BERT/Contriever, 384 for MiniLM, etc.,
features.build_embedding_feature), and query_type (1-dim,
features.build_query_type_feature) - at dataset construction time, rather
than one combined cache. This lets each piece be built/rebuilt independently:
the lexical cache needs a Lucene index (index_stats), while the embedding and
query_type caches don't need Lucene at all, only a raw embedding source (e.g.
data/cache/bert-query-embeddings/cls/*.cls.pkl) / the classifier respectively.

QPPDataset (the full doc+query model) additionally consumes a FOURTH,
independent cache - doc_content_feats, keyed by (qid, doc_id) pairs rather
than qid alone - covering the 8 Lucene-derived per-document feature columns
(features.DOC_TERM_FEATURE_NAMES; everything in DOC_FEATURE_NAMES except
`score`, which is copied straight from the run file, can be rescaled by
--score_scale, and so is never cached - see features.assemble_doc_feature).
It's keyed at (qid, doc_id), NOT (ranker, qid, doc_id): those 8 columns don't
depend on which ranker retrieved the doc, so this dedupes work across
rankers whose top-k lists overlap for the same query. This matches the
doc-feature cache other branches (e.g. classification-head-MLP) already use
- see guide_docs/FEATURE_CACHE_GUIDE.md - so a cache built there loads here
unchanged.

FeatureCache is a generic container reused by all four builders below - a
single instance can carry query_feats, doc_content_feats, or (if built
elsewhere) both; each builder here only ever populates the one it's
responsible for, matching this branch's convention of separate,
independently-rebuildable single-purpose cache files
(--lexical_cache/--embedding_cache/--query_type_cache/--doc_feature_cache).

Usage:
    lexical_cache = build_feature_cache(run_paths, queries, index_stats)
    embedding_lookup = load_query_embeddings("data/cache/bert-query-embeddings/cls/dl19.cls.pkl")
    embedding_cache = build_embedding_cache(queries, embedding_lookup)
    query_type_cache = build_query_type_cache(run_paths, queries, QueryTypeClassifier())
    doc_feature_cache = build_doc_feature_cache(run_paths, queries, index_stats)
    save_feature_cache(lexical_cache, "dl19_lexical.pkl")
    ...
    cache = load_feature_cache("dl19_lexical.pkl")
"""

import csv
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from features import (
    QueryTypeClassifier,
    build_doc_term_features,
    build_embedding_feature,
    build_entity_count_feature,
    build_query_features,
    build_query_type_feature,
    build_scs_pmi_feature,
)


@dataclass
class FeatureCache:
    query_feats: Dict[str, np.ndarray] = field(default_factory=dict)
    doc_content_feats: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
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


def load_entity_count_lookup(path: str) -> Dict[str, float]:
    """Load a precomputed {qid: named-entity count} lookup from a CSV with
    (at least) `qid` and `num_entities` columns - see
    data/cache/entity-counts/query_entity_counts.csv. Any other columns (e.g. `dataset`,
    `qtext`) are ignored; qid is expected to be globally unique across the
    file. Raises ValueError if the same qid appears twice with conflicting
    counts."""
    lookup: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
            count = float(row["num_entities"])
            if qid in lookup and lookup[qid] != count:
                raise ValueError(
                    f"Conflicting entity counts for qid={qid!r} in {path!r}: "
                    f"{lookup[qid]!r} vs {count!r}."
                )
            lookup[qid] = count
    return lookup


def load_scs_pmi_lookup(path: str) -> Dict[str, np.ndarray]:
    """Load a precomputed {qid: [scs, avg_pmi, max_pmi]} lookup from a CSV
    with (at least) `qid`, `scs`, `avg_pmi`, `max_pmi` columns - see
    scs_pmi_features.csv. Any other columns (e.g. `dataset`, `qtext`) are
    ignored; qid is expected to be globally unique across the file. Raises
    ValueError if the same qid appears twice with conflicting values.

    Any of `scs`/`avg_pmi`/`max_pmi` may be empty for some rows. Missing
    values are mean-imputed: each column is filled with the mean of that
    column's non-missing values across this file, computed once here. This
    is the only imputation in this codebase - every other feature lookup
    fails loudly on a missing qid instead (see build_scs_pmi_feature) -
    because here the qid IS present, just with a partially missing row, and
    dropping/erroring on those qids would shrink the usable query set rather
    than degrade gracefully.
    """
    raw_rows: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
            scs = float(row["scs"]) if row["scs"].strip() else None
            avg_pmi = float(row["avg_pmi"]) if row["avg_pmi"].strip() else None
            max_pmi = float(row["max_pmi"]) if row["max_pmi"].strip() else None
            if qid in raw_rows and raw_rows[qid] != (scs, avg_pmi, max_pmi):
                raise ValueError(
                    f"Conflicting scs/pmi values for qid={qid!r} in {path!r}: "
                    f"{raw_rows[qid]!r} vs {(scs, avg_pmi, max_pmi)!r}."
                )
            raw_rows[qid] = (scs, avg_pmi, max_pmi)

    scs_vals = [v[0] for v in raw_rows.values() if v[0] is not None]
    avg_pmi_vals = [v[1] for v in raw_rows.values() if v[1] is not None]
    max_pmi_vals = [v[2] for v in raw_rows.values() if v[2] is not None]
    scs_mean = float(np.mean(scs_vals)) if scs_vals else 0.0
    avg_pmi_mean = float(np.mean(avg_pmi_vals)) if avg_pmi_vals else 0.0
    max_pmi_mean = float(np.mean(max_pmi_vals)) if max_pmi_vals else 0.0

    lookup: Dict[str, np.ndarray] = {}
    for qid, (scs, avg_pmi, max_pmi) in raw_rows.items():
        lookup[qid] = np.array(
            [
                scs if scs is not None else scs_mean,
                avg_pmi if avg_pmi is not None else avg_pmi_mean,
                max_pmi if max_pmi is not None else max_pmi_mean,
            ],
            dtype=np.float32,
        )
    return lookup


def build_scs_pmi_cache(
    queries: Dict[str, str],
    scs_pmi_lookup: Dict[str, np.ndarray],
) -> FeatureCache:
    """
    Precompute every qid's 3-dim scs_pmi feature (features.build_scs_pmi_feature)
    for every qid in queries. Like entity_count, this needs no run files -
    pure lookup, no dependency on run-file content (no Lucene index either -
    only the precomputed scs_pmi_lookup, see load_scs_pmi_lookup).
    """
    query_feats: Dict[str, np.ndarray] = {
        qid: build_scs_pmi_feature(qid, scs_pmi_lookup) for qid in queries
    }

    meta = {"num_qids": len(query_feats)}
    return FeatureCache(query_feats=query_feats, meta=meta)


def build_entity_count_cache(
    queries: Dict[str, str],
    entity_count_lookup: Dict[str, float],
) -> FeatureCache:
    """
    Precompute every qid's 1-dim entity_count feature
    (features.build_entity_count_feature) for every qid in queries. Like
    embedding/query_type, this needs no run files - pure lookup, no
    dependency on run-file content (no Lucene index either - only the
    precomputed entity_count_lookup, see load_entity_count_lookup).
    """
    query_feats: Dict[str, np.ndarray] = {
        qid: build_entity_count_feature(qid, entity_count_lookup) for qid in queries
    }

    meta = {"num_qids": len(query_feats)}
    return FeatureCache(query_feats=query_feats, meta=meta)


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


def build_doc_feature_cache(
    run_paths: List[str],
    queries: Dict[str, str],
    index_stats,
) -> FeatureCache:
    """
    Precompute every (qid, doc_id) pair's 8-dim Lucene-derived doc-content
    feature vector (features.build_doc_term_features - DOC_TERM_FEATURE_NAMES,
    everything except `score`) over every (ranker, qid, doc) triple in
    run_paths, intersected with queries. Deduped by (qid, doc_id): a doc
    appearing under multiple rankers' lists for the same query is computed
    once, since these 8 columns don't depend on which ranker retrieved the
    doc (score does - it's never cached, see features.assemble_doc_feature).
    Not qrels-filtered, and NOT truncated by top_k - covers every doc in the
    run files, so it stays valid regardless of the --top_k used later. This
    is a doc-only cache (query_feats left empty) - see the module docstring
    for why this branch keeps it as its own single-purpose artifact rather
    than combining it with build_feature_cache's lexical output.
    """
    from dataset import load_run  # deferred: dataset.py imports this module

    runs = load_run(run_paths)

    doc_content_feats: Dict[Tuple[str, str], np.ndarray] = {}
    for run_data in runs.values():
        for qid, doc_list in run_data.items():
            if qid not in queries:
                continue
            query_terms = queries[qid].lower().split()
            if not query_terms:
                continue
            for doc_id, _score in doc_list:
                key = (qid, doc_id)
                if key in doc_content_feats:
                    continue
                doc_content_feats[key] = build_doc_term_features(doc_id, query_terms, index_stats)

    meta = {"num_doc_pairs": len(doc_content_feats), "built_from": list(run_paths)}
    return FeatureCache(doc_content_feats=doc_content_feats, meta=meta)


def save_feature_cache(cache: FeatureCache, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_feature_cache(path: str) -> FeatureCache:
    with open(path, "rb") as f:
        cache = pickle.load(f)
    # Pickles saved before `doc_content_feats` was added to FeatureCache
    # unpickle via __new__ + __dict__ update, NOT __init__ - so they never
    # get the dataclass field default and are simply missing the attribute
    # entirely (not just empty). Backfill it so every FeatureCache instance
    # is safe to access uniformly regardless of which version saved it.
    if not hasattr(cache, "doc_content_feats"):
        cache.doc_content_feats = {}
    parts = []
    if cache.query_feats:
        parts.append(f"{cache.meta.get('num_qids', len(cache.query_feats))} qids")
    if cache.doc_content_feats:
        parts.append(f"{cache.meta.get('num_doc_pairs', len(cache.doc_content_feats))} (qid, doc_id) pairs")
    print(f"Loaded feature cache from {path}: {', '.join(parts) if parts else 'empty'}")
    return cache
