"""
Unit tests for features.py's doc-feature split: build_doc_term_features (the
8-dim, Lucene-derived, cacheable part) + assemble_doc_feature (splices the
live `score` value back in) must together reproduce build_doc_features'
(the thin wrapper) output exactly - see guide_docs/FEATURE_CACHE_GUIDE.md /
feature_cache.build_doc_feature_cache.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (
    DOC_FEATURE_DIM,
    DOC_FEATURE_NAMES,
    DOC_TERM_FEATURE_NAMES,
    SCORE_FEATURE_INDICES,
    assemble_doc_feature,
    build_doc_features,
    build_doc_term_features,
)


class DummyIndexStats:
    """Deterministic fake IndexStats exercising both the doc_term_counts hit
    and miss (None) branches."""

    def idf(self, term):
        return float(len(term))

    def doc_term_counts(self, doc_id):
        if doc_id == "missing":
            return None
        return Counter({"a": 2, "b": 1})

    def bm25_tf(self, term, tf_map, doc_len, k1=1.2, b=0.75):
        return float(tf_map.get(term, 0))


def test_doc_term_feature_names_is_doc_feature_names_minus_score():
    assert DOC_TERM_FEATURE_NAMES == [n for n in DOC_FEATURE_NAMES if n != "score"]
    assert len(DOC_TERM_FEATURE_NAMES) == 8
    assert DOC_FEATURE_DIM == 9


def test_build_doc_term_features_shape_and_values():
    term_feats = build_doc_term_features("d1", ["a", "c"], DummyIndexStats())
    assert term_feats.shape == (8,)
    # num_doc_terms=3, num_unique_doc_terms=2, min_idf=1, max_idf=1, sum_idf=2,
    # overlap=1 (only "a" shared), bm25_tf_sum/max computed from query_terms=["a","c"]
    assert term_feats[0] == 3.0  # num_doc_terms (doc_len = 2+1)
    assert term_feats[1] == 2.0  # num_unique_doc_terms
    assert term_feats[5] == 1.0  # overlap


def test_build_doc_term_features_handles_missing_doc():
    term_feats = build_doc_term_features("missing", ["a"], DummyIndexStats())
    assert term_feats.shape == (8,)
    assert np.all(term_feats == 0.0)


def test_assemble_doc_feature_inserts_score_at_fixed_position():
    term_feats = build_doc_term_features("d1", ["a"], DummyIndexStats())
    full = assemble_doc_feature(term_feats, score=7.5)
    assert full.shape == (9,)
    assert full[SCORE_FEATURE_INDICES[0]] == 7.5
    # Everything else must match the term vector, just shifted around the
    # inserted score column.
    np.testing.assert_array_equal(full[: SCORE_FEATURE_INDICES[0]], term_feats[: SCORE_FEATURE_INDICES[0]])
    np.testing.assert_array_equal(full[SCORE_FEATURE_INDICES[0] + 1 :], term_feats[SCORE_FEATURE_INDICES[0] :])


@pytest.mark.parametrize("doc_id,query_terms,score", [
    ("d1", ["a", "c"], 3.5),
    ("d2", ["a", "b", "z"], -1.2),
    ("missing", ["a"], 0.0),
])
def test_build_doc_features_equals_term_features_plus_assemble(doc_id, query_terms, score):
    """Regression guard for the build_doc_features refactor: the thin
    wrapper must be byte-identical to computing the two pieces separately."""
    full_direct = build_doc_features(doc_id, query_terms, score, DummyIndexStats())
    term_feats = build_doc_term_features(doc_id, query_terms, DummyIndexStats())
    full_via_assemble = assemble_doc_feature(term_feats, score)
    np.testing.assert_array_equal(full_direct, full_via_assemble)
    assert full_direct.dtype == np.float32
