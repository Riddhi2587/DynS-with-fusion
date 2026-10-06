"""Tests for build_fusion_row.py's per-query metric helpers (query_macro_rbo,
query_mrr). These only take the predictions/metrics_lookup dict shapes, so no
CSV files or model/dataset are needed."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_fusion_row import query_macro_rbo, query_mrr

METRIC_KEY = "ndcg_cut_10"


def test_query_macro_rbo_is_one_for_identical_predicted_and_true_lists():
    predictions = {
        "q1": {"A": 3.0, "B": 2.0, "C": 1.0},
        "q2": {"A": 5.0, "B": 4.0, "C": 0.5},
    }
    metrics_lookup = {
        # exact same values as the predicted scores in each query, not just
        # the same relative order.
        ("q1", "A"): {METRIC_KEY: 3.0},
        ("q1", "B"): {METRIC_KEY: 2.0},
        ("q1", "C"): {METRIC_KEY: 1.0},
        ("q2", "A"): {METRIC_KEY: 5.0},
        ("q2", "B"): {METRIC_KEY: 4.0},
        ("q2", "C"): {METRIC_KEY: 0.5},
    }
    result = query_macro_rbo(predictions, metrics_lookup, METRIC_KEY, p=0.8)
    assert result == pytest.approx(1.0)


def test_query_macro_rbo_averages_identical_and_reversed_queries():
    predictions = {
        "q1": {"A": 3.0, "B": 2.0, "C": 1.0},
        "q2": {"A": 3.0, "B": 2.0, "C": 1.0},
    }
    metrics_lookup = {
        # q1: true order matches predicted order (A, B, C) -> RBO=1.0
        ("q1", "A"): {METRIC_KEY: 30.0},
        ("q1", "B"): {METRIC_KEY: 20.0},
        ("q1", "C"): {METRIC_KEY: 10.0},
        # q2: true order is fully reversed (C, B, A) -> RBO=0.72 (p=0.8,
        # matches test_evaluate.py's test_rbo_fully_reversed_order_matches_hand_computation)
        ("q2", "A"): {METRIC_KEY: 10.0},
        ("q2", "B"): {METRIC_KEY: 20.0},
        ("q2", "C"): {METRIC_KEY: 30.0},
    }
    result = query_macro_rbo(predictions, metrics_lookup, METRIC_KEY, p=0.8)
    assert result == pytest.approx((1.0 + 0.72) / 2)


def test_query_macro_rbo_skips_queries_with_fewer_than_two_rankers():
    predictions = {"q1": {"A": 1.0}}
    metrics_lookup = {("q1", "A"): {METRIC_KEY: 10.0}}
    assert math.isnan(query_macro_rbo(predictions, metrics_lookup, METRIC_KEY))


def test_query_macro_rbo_nan_when_no_queries_qualify():
    assert math.isnan(query_macro_rbo({}, {}, METRIC_KEY))


def test_query_macro_rbo_tied_metric_values_matches_hand_computation():
    # Predicted scores strictly ordered A,B,C; true metric values tie A and B
    # for best. Same numbers as test_evaluate.py's
    # test_rbo_tied_labels_matches_hand_computation -> 14/15.
    predictions = {"q1": {"A": 3.0, "B": 2.0, "C": 1.0}}
    metrics_lookup = {
        ("q1", "A"): {METRIC_KEY: 5.0},
        ("q1", "B"): {METRIC_KEY: 5.0},
        ("q1", "C"): {METRIC_KEY: 1.0},
    }
    result = query_macro_rbo(predictions, metrics_lookup, METRIC_KEY, p=0.8)
    assert result == pytest.approx(14 / 15)


def test_query_macro_rbo_tied_predicted_scores_matches_hand_computation():
    # Same as above with predicted/true roles swapped - agreement is
    # symmetric, so the result is identical.
    predictions = {"q1": {"A": 5.0, "B": 5.0, "C": 1.0}}
    metrics_lookup = {
        ("q1", "A"): {METRIC_KEY: 3.0},
        ("q1", "B"): {METRIC_KEY: 2.0},
        ("q1", "C"): {METRIC_KEY: 1.0},
    }
    result = query_macro_rbo(predictions, metrics_lookup, METRIC_KEY, p=0.8)
    assert result == pytest.approx(14 / 15)


def test_query_macro_rbo_full_tie_at_top_of_both_rankings_is_one():
    # Both predicted scores and true metric values tie A and B for the top
    # rank, so the induced (competition) rankings are identical -> RBO=1.
    predictions = {"q1": {"A": 1.0, "B": 1.0, "C": 0.5}}
    metrics_lookup = {
        ("q1", "A"): {METRIC_KEY: 2.0},
        ("q1", "B"): {METRIC_KEY: 2.0},
        ("q1", "C"): {METRIC_KEY: 1.0},
    }
    result = query_macro_rbo(predictions, metrics_lookup, METRIC_KEY, p=0.8)
    assert result == pytest.approx(1.0)


def test_query_mrr_averages_best_first_and_best_last_queries():
    predictions = {
        "q1": {"A": 3.0, "B": 2.0, "C": 1.0},  # true best A predicted rank 1
        "q2": {"A": 1.0, "B": 2.0, "C": 3.0},  # true best A predicted rank 3
    }
    metrics_lookup = {
        ("q1", "A"): {METRIC_KEY: 10.0},
        ("q1", "B"): {METRIC_KEY: 5.0},
        ("q1", "C"): {METRIC_KEY: 1.0},
        ("q2", "A"): {METRIC_KEY: 10.0},
        ("q2", "B"): {METRIC_KEY: 5.0},
        ("q2", "C"): {METRIC_KEY: 1.0},
    }
    result = query_mrr(predictions, metrics_lookup, METRIC_KEY)
    assert result == pytest.approx((1.0 + 1 / 3) / 2)


def test_query_mrr_tie_for_best_uses_earliest_predicted_rank():
    predictions = {"q1": {"A": 5.0, "B": 1.0, "C": 1.0}}
    metrics_lookup = {
        ("q1", "A"): {METRIC_KEY: 10.0},
        ("q1", "B"): {METRIC_KEY: 10.0},  # ties with A for true best
        ("q1", "C"): {METRIC_KEY: 1.0},
    }
    # A is predicted rank 1 and ties for best label -> RR=1.0
    assert query_mrr(predictions, metrics_lookup, METRIC_KEY) == pytest.approx(1.0)
