"""
Equivalence tests for the three SEPARATE query-only feature caches
(feature_cache.py) - lexical, embedding, query_type: QPPQueryOnlyDataset must
merge them per-qid into an identical query_feats vector whether a piece comes
from its own cache or its own live-computation fallback, fall back correctly
per-piece on a partial cache, and raise a clear, piece-specific error on an
uncovered miss with no fallback available for that piece.

Also covers the FOURTH cache, doc_content_feats (keyed by (qid, doc_id) -
see guide_docs/FEATURE_CACHE_GUIDE.md), built by build_doc_feature_cache.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_metrics_csv import build_metrics_csv
from dataset import QPPQueryOnlyDataset
from feature_cache import (
    build_doc_feature_cache,
    build_embedding_cache,
    build_feature_cache,
    build_query_type_cache,
    load_feature_cache,
    load_query_embeddings,
    save_feature_cache,
)
from features import DOC_TERM_FEATURE_NAMES, build_doc_term_features

# Realistic raw embedding width (e.g. BERT/Contriever CLS) - deliberately NOT
# features.EMBEDDING_DIM (32), which is now only the model's learned
# projection's output width, unrelated to the raw cache width tested here.
RAW_EMBEDDING_DIM = 768


class DummyIndexStats:
    """Deterministic fake IndexStats - build_query_features only needs .idf();
    doc_term_counts/bm25_tf back build_doc_term_features for the doc-cache tests."""

    def idf(self, term):
        return float(len(term))

    def doc_term_counts(self, doc_id):
        return Counter({doc_id: 1})

    def bm25_tf(self, term, tf_map, doc_len, k1=1.2, b=0.75):
        return float(tf_map.get(term, 0))


class DummyQueryTypeClassifier:
    """Deterministic fake QueryTypeClassifier - no network/model download.
    Mimics question-detection with a trivial rule: ends with "?" -> 1.0."""

    def classify(self, query):
        return 1.0 if query.strip().endswith("?") else 0.0


QUERIES = {
    "q1": "one two three",
    "q2": "four five",
    "q3": "six",
}

# One distinct fixed embedding per qid, along different axes (not just
# different scales of the same direction) so their L2-normalized forms stay
# distinguishable per qid (build_embedding_feature L2-normalizes - see
# features.py).
EMBEDDING_LOOKUP = {
    "q1": np.eye(RAW_EMBEDDING_DIM, dtype=np.float32)[0] * 5.0,
    "q2": np.eye(RAW_EMBEDDING_DIM, dtype=np.float32)[1] * 2.0,
    "q3": np.eye(RAW_EMBEDDING_DIM, dtype=np.float32)[2] * 3.0,
}

RUN_ROWS = {
    "bm25.res": [
        ("q1", "d1", 1, 3.0),
        ("q1", "d2", 2, 2.0),
        ("q2", "d3", 1, 1.5),
    ],
    "rm3.res": [
        ("q1", "d1", 1, 4.0),
        ("q3", "d4", 1, 0.5),
    ],
}

QRELS_ROWS = [
    ("q1", "d1", 1),
    ("q1", "d2", 0),
    ("q2", "d3", 1),
    ("q3", "d4", 1),
]


@pytest.fixture
def run_paths(tmp_path):
    paths = []
    for fname, rows in RUN_ROWS.items():
        p = tmp_path / fname
        with open(p, "w") as f:
            for qid, docid, rank, score in rows:
                f.write(f"{qid} Q0 {docid} {rank} {score} run\n")
        paths.append(str(p))
    return paths


@pytest.fixture
def qrels_path(tmp_path):
    p = tmp_path / "qrels.txt"
    with open(p, "w") as f:
        for qid, docid, rel in QRELS_ROWS:
            f.write(f"{qid} 0 {docid} {rel}\n")
    return str(p)


@pytest.fixture
def metrics_csv_path(tmp_path, run_paths, qrels_path):
    """QPPQueryOnlyDataset requires metrics_csv (no pytrec_eval fallback) -
    build one from the same fixture data via build_metrics_csv.py so these
    feature-cache tests can still construct datasets."""
    p = tmp_path / "metrics.csv"
    build_metrics_csv(run_paths, qrels_path, str(p))
    return str(p)


def _query_feats_by_qid(dataset):
    return {s["qid"]: s["query_feats"] for s in dataset.samples}


def test_full_caches_match_no_cache(run_paths, qrels_path, metrics_csv_path):
    index_stats = DummyIndexStats()
    classifier = DummyQueryTypeClassifier()

    baseline = QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, index_stats,
        embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=classifier,
        metrics_csv=metrics_csv_path,
    )

    lexical_cache = build_feature_cache(run_paths, QUERIES, index_stats)
    embedding_cache = build_embedding_cache(QUERIES, EMBEDDING_LOOKUP)
    query_type_cache = build_query_type_cache(QUERIES, classifier)
    assert lexical_cache.meta["num_qids"] == 3
    assert embedding_cache.meta["num_qids"] == 3
    assert query_type_cache.meta["num_qids"] == 3

    cached = QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, index_stats,
        lexical_cache=lexical_cache,
        embedding_cache=embedding_cache,
        query_type_cache=query_type_cache,
        metrics_csv=metrics_csv_path,
    )

    baseline_feats = _query_feats_by_qid(baseline)
    cached_feats = _query_feats_by_qid(cached)
    assert set(baseline_feats) == set(cached_feats) == {"q1", "q2", "q3"}
    for qid in baseline_feats:
        np.testing.assert_array_equal(baseline_feats[qid], cached_feats[qid])

    baseline_labels = {s["qid"]: s["labels"] for s in baseline.samples}
    cached_labels = {s["qid"]: s["labels"] for s in cached.samples}
    for qid in baseline_labels:
        np.testing.assert_array_equal(baseline_labels[qid], cached_labels[qid])


def test_query_feats_concatenates_lexical_embedding_and_query_type(
    run_paths, qrels_path, metrics_csv_path
):
    """The (5 + RAW_EMBEDDING_DIM + 1)-dim query_feats vector must be exactly
    [5 lexical/IDF features, RAW_EMBEDDING_DIM raw embedding dims
    (L2-normalized), 1 query_type flag], in that order - built here entirely
    via live fallback (no caches), which is the same assembly path a
    partial/no-cache QPPQueryOnlyDataset uses per piece. Also confirms
    dataset.query_feature_dim/embedding_slice are derived from the inferred
    raw width, not any fixed constant."""
    index_stats = DummyIndexStats()
    classifier = DummyQueryTypeClassifier()

    ds = QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, index_stats,
        embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=classifier,
        metrics_csv=metrics_csv_path,
    )
    assert ds.query_feature_dim == 5 + RAW_EMBEDDING_DIM + 1
    assert ds.embedding_slice == slice(5, 5 + RAW_EMBEDDING_DIM)
    feats_by_qid = _query_feats_by_qid(ds)

    for qid, raw_query in QUERIES.items():
        vec = feats_by_qid[qid]
        assert vec.shape == (5 + RAW_EMBEDDING_DIM + 1,)
        expected_embedding = EMBEDDING_LOOKUP[qid] / np.linalg.norm(EMBEDDING_LOOKUP[qid])
        np.testing.assert_allclose(vec[5:5 + RAW_EMBEDDING_DIM], expected_embedding, rtol=1e-6)
        assert vec[-1] == classifier.classify(raw_query)


def test_partial_caches_fall_back_independently_per_piece(run_paths, qrels_path, metrics_csv_path):
    """Each of the three caches can be independently missing a different qid
    - each piece falls back to its own live computation without affecting
    the other two pieces or the other qids."""
    index_stats = DummyIndexStats()
    classifier = DummyQueryTypeClassifier()

    baseline = QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, index_stats,
        embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=classifier,
        metrics_csv=metrics_csv_path,
    )
    baseline_feats = _query_feats_by_qid(baseline)

    lexical_cache = build_feature_cache(run_paths, QUERIES, index_stats)
    embedding_cache = build_embedding_cache(QUERIES, EMBEDDING_LOOKUP)
    query_type_cache = build_query_type_cache(QUERIES, classifier)
    del lexical_cache.query_feats["q1"]      # force a lexical miss for q1
    del embedding_cache.query_feats["q2"]    # force an embedding miss for q2
    del query_type_cache.query_feats["q3"]   # force a query_type miss for q3

    cached = QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, index_stats,
        lexical_cache=lexical_cache,
        embedding_cache=embedding_cache,
        query_type_cache=query_type_cache,
        embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=classifier,
        metrics_csv=metrics_csv_path,
    )
    cached_feats = _query_feats_by_qid(cached)

    for qid in baseline_feats:
        np.testing.assert_array_equal(baseline_feats[qid], cached_feats[qid])


def test_miss_without_lexical_fallback_raises(run_paths, qrels_path, metrics_csv_path):
    index_stats = DummyIndexStats()
    classifier = DummyQueryTypeClassifier()
    lexical_cache = build_feature_cache(run_paths, QUERIES, index_stats)
    del lexical_cache.query_feats["q2"]

    with pytest.raises(RuntimeError, match="lexical_cache.*q2"):
        QPPQueryOnlyDataset(
            run_paths, qrels_path, QUERIES, index_stats=None,
            lexical_cache=lexical_cache,
            embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=classifier,
            metrics_csv=metrics_csv_path,
        )


def test_miss_without_embedding_fallback_raises(run_paths, qrels_path, metrics_csv_path):
    index_stats = DummyIndexStats()
    classifier = DummyQueryTypeClassifier()
    embedding_cache = build_embedding_cache(QUERIES, EMBEDDING_LOOKUP)
    del embedding_cache.query_feats["q2"]

    with pytest.raises(RuntimeError, match="embedding_cache.*q2"):
        QPPQueryOnlyDataset(
            run_paths, qrels_path, QUERIES, index_stats,
            embedding_cache=embedding_cache,
            embedding_lookup=None, query_type_classifier=classifier,
            metrics_csv=metrics_csv_path,
        )


def test_miss_without_query_type_fallback_raises(run_paths, qrels_path, metrics_csv_path):
    index_stats = DummyIndexStats()
    classifier = DummyQueryTypeClassifier()
    query_type_cache = build_query_type_cache(QUERIES, classifier)
    del query_type_cache.query_feats["q2"]

    with pytest.raises(RuntimeError, match="query_type_cache.*q2"):
        QPPQueryOnlyDataset(
            run_paths, qrels_path, QUERIES, index_stats,
            query_type_cache=query_type_cache,
            embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=None,
            metrics_csv=metrics_csv_path,
        )


# --- doc_content_feats (the fourth, (qid, doc_id)-keyed cache) -------------

def test_build_doc_feature_cache_keys_and_values(run_paths):
    index_stats = DummyIndexStats()
    cache = build_doc_feature_cache(run_paths, QUERIES, index_stats)

    # RUN_ROWS: bm25.res has (q1,d1),(q1,d2),(q2,d3); rm3.res has (q1,d1)
    # (dup, same qid+doc as bm25.res -> deduped),(q3,d4).
    assert set(cache.doc_content_feats.keys()) == {("q1", "d1"), ("q1", "d2"), ("q2", "d3"), ("q3", "d4")}
    assert cache.meta["num_doc_pairs"] == 4
    assert cache.query_feats == {}

    expected = build_doc_term_features("d1", QUERIES["q1"].lower().split(), index_stats)
    np.testing.assert_array_equal(cache.doc_content_feats[("q1", "d1")], expected)
    for vec in cache.doc_content_feats.values():
        assert vec.shape == (len(DOC_TERM_FEATURE_NAMES),)


def test_build_doc_feature_cache_dedups_across_rankers(run_paths):
    """(q1, d1) appears in both bm25.res and rm3.res - must be computed (and
    stored) once, not once per ranker."""
    index_stats = DummyIndexStats()
    cache = build_doc_feature_cache(run_paths, QUERIES, index_stats)
    assert len([k for k in cache.doc_content_feats if k == ("q1", "d1")]) == 1


def test_doc_feature_cache_save_load_round_trip(run_paths, tmp_path):
    index_stats = DummyIndexStats()
    cache = build_doc_feature_cache(run_paths, QUERIES, index_stats)

    path = str(tmp_path / "doc_cache.pkl")
    save_feature_cache(cache, path)
    loaded = load_feature_cache(path)

    assert loaded.doc_content_feats.keys() == cache.doc_content_feats.keys()
    for key in cache.doc_content_feats:
        np.testing.assert_array_equal(loaded.doc_content_feats[key], cache.doc_content_feats[key])


def test_load_feature_cache_backfills_missing_doc_content_feats(tmp_path):
    """Regression test for an AttributeError hit in practice: FeatureCache
    pickles saved before doc_content_feats existed (with the old 2-field
    query_feats/meta shape) unpickle via __new__ + __dict__ update, NOT
    __init__ - so they never get the dataclass field default and are simply
    missing the attribute entirely, not just empty. Reproduce that exact
    shape by deleting the key from an instance's __dict__ before pickling
    (same effect as an old-format pickle), then confirm load_feature_cache
    backfills it to {} instead of raising AttributeError."""
    cache = build_feature_cache(run_paths=[], queries={}, index_stats=DummyIndexStats())
    del cache.__dict__["doc_content_feats"]
    assert "doc_content_feats" not in cache.__dict__

    path = str(tmp_path / "old_format_cache.pkl")
    save_feature_cache(cache, path)

    loaded = load_feature_cache(path)  # must not raise AttributeError
    assert loaded.doc_content_feats == {}


# --- load_query_embeddings: infer-and-validate-consistency (raw, variable width) ---

def test_load_query_embeddings_infers_width_from_first_entry(tmp_path):
    """No expected_dim given: the raw width is inferred from an arbitrary
    entry and every other entry validated against it - never a fixed
    constant."""
    import pickle
    path = tmp_path / "embeddings.pkl"
    embeddings = {qid: vec for qid, vec in EMBEDDING_LOOKUP.items()}
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

    loaded = load_query_embeddings(str(path))
    assert set(loaded) == set(EMBEDDING_LOOKUP)
    for qid, vec in loaded.items():
        assert np.asarray(vec).shape == (RAW_EMBEDDING_DIM,)


def test_load_query_embeddings_mismatched_widths_raise(tmp_path):
    """A malformed/mixed cache file (different qids at different raw widths)
    must raise at load time, not surface as a confusing shape mismatch deep
    inside model training."""
    import pickle
    path = tmp_path / "mismatched.pkl"
    embeddings = {
        "q1": np.zeros(RAW_EMBEDDING_DIM, dtype=np.float32),
        "q2": np.zeros(384, dtype=np.float32),
    }
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

    with pytest.raises(ValueError, match="q2"):
        load_query_embeddings(str(path))


def test_load_query_embeddings_expected_dim_mismatch_raises(tmp_path):
    """expected_dim, when given, is an additional explicit sanity check
    against a known width."""
    import pickle
    path = tmp_path / "embeddings.pkl"
    with open(path, "wb") as f:
        pickle.dump({"q1": np.zeros(RAW_EMBEDDING_DIM, dtype=np.float32)}, f)

    with pytest.raises(ValueError, match="384"):
        load_query_embeddings(str(path), expected_dim=384)

    # Matching expected_dim does not raise.
    load_query_embeddings(str(path), expected_dim=RAW_EMBEDDING_DIM)
