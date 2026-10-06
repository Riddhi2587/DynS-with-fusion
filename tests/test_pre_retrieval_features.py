"""
Unit tests for the pre-retrieval feature pieces added to features.py: the
three atomic builders (build_query_features - existing/lexical,
build_embedding_feature, build_query_type_feature), build_full_query_features
as their composed wrapper (shape/order/values, missing-qid error), and
QueryTypeClassifier's label parsing. QueryTypeClassifier is exercised via an
injected fake pipeline_fn, never a real transformers pipeline/model download,
so this file needs no network access.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (
    QueryTypeClassifier,
    build_embedding_feature,
    build_full_query_features,
    build_query_type_feature,
)

# Realistic raw embedding width (e.g. BERT/Contriever CLS) - deliberately NOT
# features.EMBEDDING_DIM (32), which is now only the model's learned
# projection's output width, unrelated to the raw, variable-width embedding
# these builders handle.
RAW_EMBEDDING_DIM = 768


class DummyIndexStats:
    """Deterministic fake IndexStats - build_query_features only needs .idf()."""

    def idf(self, term):
        return float(len(term))


EMBEDDING_LOOKUP = {
    "q1": np.arange(RAW_EMBEDDING_DIM, dtype=np.float32),
    "q2": np.full(RAW_EMBEDDING_DIM, -1.0, dtype=np.float32),
}


def _fixed_label_pipeline(label):
    def pipeline_fn(queries):
        return [{"label": label} for _ in queries]
    return pipeline_fn


def test_build_full_query_features_shape_order_values():
    index_stats = DummyIndexStats()
    classifier = QueryTypeClassifier(pipeline_fn=_fixed_label_pipeline("LABEL_1"))

    vec = build_full_query_features(
        "one two three", ["one", "two", "three"], "q1",
        index_stats, EMBEDDING_LOOKUP, classifier,
    )

    assert vec.shape == (5 + RAW_EMBEDDING_DIM + 1,)
    assert vec.dtype == np.float32
    # First 5 = lexical/IDF features (num_query_terms, num_unique_query_terms,
    # min_idf, max_idf, sum_idf), matching build_query_features's own formula.
    np.testing.assert_array_equal(vec[:5], [3.0, 3.0, 3.0, 5.0, 11.0])
    # Next RAW_EMBEDDING_DIM = the precomputed embedding, L2-normalized (not raw).
    expected_embedding = EMBEDDING_LOOKUP["q1"] / np.linalg.norm(EMBEDDING_LOOKUP["q1"])
    np.testing.assert_allclose(vec[5:5 + RAW_EMBEDDING_DIM], expected_embedding, rtol=1e-6)
    # Last 1 = query_type, parsed from the injected pipeline's label.
    assert vec[-1] == 1.0


def test_build_full_query_features_embedding_is_unit_norm():
    """The embedding block must always come out L2-normalized, regardless of
    the input's original scale (a raw embedding source is not unit-norm -
    see e.g. data/cache/bert-query-embeddings/cls/*.cls.pkl)."""
    index_stats = DummyIndexStats()
    classifier = QueryTypeClassifier(pipeline_fn=_fixed_label_pipeline("LABEL_0"))

    for qid in EMBEDDING_LOOKUP:
        vec = build_full_query_features(
            "one two three", ["one", "two", "three"], qid,
            index_stats, EMBEDDING_LOOKUP, classifier,
        )
        norm = np.linalg.norm(vec[5:5 + RAW_EMBEDDING_DIM])
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_build_full_query_features_missing_qid_raises_keyerror():
    index_stats = DummyIndexStats()
    classifier = QueryTypeClassifier(pipeline_fn=_fixed_label_pipeline("LABEL_0"))

    with pytest.raises(KeyError, match="q999"):
        build_full_query_features(
            "one two three", ["one", "two", "three"], "q999",
            index_stats, EMBEDDING_LOOKUP, classifier,
        )


def test_build_full_query_features_accepts_any_embedding_width():
    """build_embedding_feature/build_full_query_features no longer assert a
    fixed embedding shape - the raw width is variable by source (768 for
    BERT/Contriever, 384 for MiniLM, etc.), so any width must pass through
    unchanged (width consistency is validated elsewhere, at
    feature_cache.load_query_embeddings/dataset construction time, not here)."""
    index_stats = DummyIndexStats()
    classifier = QueryTypeClassifier(pipeline_fn=_fixed_label_pipeline("LABEL_0"))
    small_lookup = {"q1": np.arange(5, dtype=np.float32)}

    vec = build_full_query_features(
        "one two three", ["one", "two", "three"], "q1",
        index_stats, small_lookup, classifier,
    )
    assert vec.shape == (5 + 5 + 1,)


def test_build_embedding_feature_normalizes_and_matches_full_wrapper():
    vec = build_embedding_feature("q1", EMBEDDING_LOOKUP)
    expected = EMBEDDING_LOOKUP["q1"] / np.linalg.norm(EMBEDDING_LOOKUP["q1"])
    np.testing.assert_allclose(vec, expected, rtol=1e-6)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)


def test_build_embedding_feature_missing_qid_raises_keyerror():
    with pytest.raises(KeyError, match="q999"):
        build_embedding_feature("q999", EMBEDDING_LOOKUP)


def test_build_embedding_feature_accepts_any_width():
    """No fixed-shape assertion - the raw width is variable by source."""
    small_lookup = {"q1": np.array([3.0, 4.0], dtype=np.float32)}
    vec = build_embedding_feature("q1", small_lookup)
    assert vec.shape == (2,)
    np.testing.assert_allclose(vec, [0.6, 0.8], rtol=1e-6)


def test_build_query_type_feature_matches_full_wrapper():
    classifier = QueryTypeClassifier(pipeline_fn=_fixed_label_pipeline("LABEL_1"))
    vec = build_query_type_feature("some query", classifier)
    assert vec.shape == (1,)
    assert vec.dtype == np.float32
    assert vec[0] == 1.0


@pytest.mark.parametrize("label,expected", [("LABEL_0", 0.0), ("LABEL_1", 1.0)])
def test_query_type_classifier_parses_label(label, expected):
    classifier = QueryTypeClassifier(pipeline_fn=_fixed_label_pipeline(label))
    assert classifier.classify("what is the capital of france") == expected


def test_query_type_classifier_batch_classify_preserves_order():
    def pipeline_fn(queries):
        # Alternate labels so order is verifiable, not just count.
        return [{"label": "LABEL_1" if i % 2 == 0 else "LABEL_0"} for i in range(len(queries))]

    classifier = QueryTypeClassifier(pipeline_fn=pipeline_fn)
    results = classifier.batch_classify(["q a", "q b", "q c"])
    assert results == [1.0, 0.0, 1.0]


def test_query_type_classifier_does_not_load_real_pipeline_when_injected():
    """Constructing with an injected pipeline_fn must never attempt the real
    transformers.pipeline(...) load (no network access in tests)."""
    calls = []

    def pipeline_fn(queries):
        calls.append(list(queries))
        return [{"label": "LABEL_0"} for _ in queries]

    classifier = QueryTypeClassifier(pipeline_fn=pipeline_fn)
    classifier.classify("some query")
    assert calls == [["some query"]]
