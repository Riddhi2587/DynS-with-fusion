"""Tests for evaluate.py's classification-metrics helpers (classification_counts,
classwise_prf1, prediction_entropy). These only take probs/labels/ranker_masks/
id_to_ranker arrays, so no dataset/model/checkpoint is needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import (
    _rbo,
    _reciprocal_rank,
    classification_counts,
    classwise_prf1,
    prediction_entropy,
)

ID_TO_RANKER = {0: "bm25", 1: "dense", 2: "fusion"}


def test_classification_counts_known_confusion_matrix():
    # 4 queries, all 3 rankers valid each time.
    # q0: true=0, pred=0 (correct)
    # q1: true=1, pred=1 (correct)
    # q2: true=1, pred=0 (wrong: fp for class0, fn for class1)
    # q3: true=2, pred=0 (wrong: fp for class0, fn for class2)
    labels = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
    ])
    masks = np.ones((4, 3), dtype=bool)

    tp, fp, fn, support = classification_counts(probs, labels, masks)
    np.testing.assert_array_equal(tp, [1, 1, 0])
    np.testing.assert_array_equal(fp, [2, 0, 0])
    np.testing.assert_array_equal(fn, [0, 1, 1])
    np.testing.assert_array_equal(support, [1, 2, 1])


def test_classwise_prf1_matches_hand_computation():
    labels = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
    ])
    masks = np.ones((4, 3), dtype=bool)

    per_class, macro_f1 = classwise_prf1(probs, labels, masks, ID_TO_RANKER)

    # class 0: tp=1, fp=2, fn=0 -> precision=1/3, recall=1.0, f1=2*(1/3)/(4/3)=0.5
    assert per_class[0]["ranker"] == "bm25"
    assert per_class[0]["precision"] == pytest.approx(1 / 3)
    assert per_class[0]["recall"] == pytest.approx(1.0)
    assert per_class[0]["f1"] == pytest.approx(0.5)
    assert per_class[0]["support"] == 1

    # class 1: tp=1, fp=0, fn=1 -> precision=1.0, recall=0.5, f1=2*0.5/1.5=2/3
    assert per_class[1]["precision"] == pytest.approx(1.0)
    assert per_class[1]["recall"] == pytest.approx(0.5)
    assert per_class[1]["f1"] == pytest.approx(2 / 3)
    assert per_class[1]["support"] == 2

    # class 2: tp=0, fp=0, fn=1 -> precision=0.0 (no predictions), recall=0.0, f1=0.0
    assert per_class[2]["precision"] == pytest.approx(0.0)
    assert per_class[2]["recall"] == pytest.approx(0.0)
    assert per_class[2]["f1"] == pytest.approx(0.0)
    assert per_class[2]["support"] == 1

    # all 3 classes have support > 0, so macro_f1 averages all 3 f1s
    expected_macro = np.mean([0.5, 2 / 3, 0.0])
    assert macro_f1 == pytest.approx(expected_macro)


def test_classwise_prf1_excludes_zero_support_class_from_macro_avg():
    # class 2 is never the true label anywhere (support=0) and is never
    # predicted either, so it should not appear in the macro average, and
    # nothing should raise/produce nan.
    labels = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    probs = np.array([
        [0.9, 0.1, 0.0],
        [0.2, 0.8, 0.0],
    ])
    masks = np.ones((2, 3), dtype=bool)

    per_class, macro_f1 = classwise_prf1(probs, labels, masks, ID_TO_RANKER)

    assert per_class[2]["support"] == 0
    assert per_class[2]["f1"] == pytest.approx(0.0)
    assert not np.isnan(macro_f1)
    # only classes 0 and 1 have support > 0, both perfectly classified -> f1=1
    assert macro_f1 == pytest.approx(1.0)


def test_prediction_entropy_zero_for_near_certain_prediction():
    probs = np.array([[1.0, 0.0, 0.0]])
    masks = np.ones((1, 3), dtype=bool)
    entropies = prediction_entropy(probs, masks)
    assert len(entropies) == 1
    assert entropies[0] == pytest.approx(0.0)


def test_prediction_entropy_log_k_for_uniform_distribution():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    masks = np.ones((1, 3), dtype=bool)
    entropies = prediction_entropy(probs, masks)
    assert entropies[0] == pytest.approx(np.log(3))


def test_prediction_entropy_and_classwise_prf1_ignore_masked_rankers():
    # 3 global classes, but only rankers 0 and 1 are valid for this query;
    # ranker 2 has nonzero prob mass in the raw array (simulating stale
    # values) but must be ignored since its mask entry is False.
    probs = np.array([[0.5, 0.5, 0.3]])
    masks = np.array([[True, True, False]])
    entropies = prediction_entropy(probs, masks)
    assert entropies[0] == pytest.approx(np.log(2))

    labels = np.array([[1.0, 0.0, 1.0]])  # ranker 2's label must be ignored
    per_class, _ = classwise_prf1(probs, labels, masks, ID_TO_RANKER)
    assert per_class[2]["support"] == 0


def test_rbo_identical_lists_is_one():
    preds = [3.0, 2.0, 1.0, 0.5]
    labels = [3.0, 2.0, 1.0, 0.5]  # exact same list, not just same order
    assert _rbo(preds, labels, p=0.8) == pytest.approx(1.0)


def test_rbo_identical_order_is_one():
    preds = [3.0, 2.0, 1.0]
    labels = [30.0, 20.0, 10.0]  # same order as preds, different scale/values
    assert _rbo(preds, labels, p=0.8) == pytest.approx(1.0)


def test_rbo_fully_reversed_order_matches_hand_computation():
    preds = [3.0, 2.0, 1.0]   # order: 0, 1, 2
    labels = [1.0, 2.0, 3.0]  # order: 2, 1, 0 (fully reversed)
    # A_1=0/1, A_2=1/2, A_3=3/3=1
    # total = 1*0 + 0.8*0.5 + 0.64*1 = 1.04
    # RBO = 0.2*1.04 + 0.8**3 = 0.208 + 0.512 = 0.72
    assert _rbo(preds, labels, p=0.8) == pytest.approx(0.72)


def test_rbo_nan_for_fewer_than_two_items():
    assert np.isnan(_rbo([1.0], [1.0]))


def test_rbo_tied_labels_matches_hand_computation():
    # preds strictly ordered 0,1,2; labels tie items 0 and 1 for best, so
    # both enter the "seen" set together at d=1 (competition ranking).
    preds = [3.0, 2.0, 1.0]
    labels = [5.0, 5.0, 1.0]
    # pred_rank=[1,2,3], true_rank=[1,1,3]
    # d=1: seen_pred={0} seen_true={0,1} -> A=2*1/3=2/3
    # d=2: seen_pred={0,1} seen_true={0,1} -> A=2*2/4=1
    # d=3: seen_pred={0,1,2} seen_true={0,1,2} -> A=1
    # total = 2/3 + 0.8*1 + 0.64*1 = 158/75
    # RBO = 0.2*(158/75) + 0.8**3 = 14/15
    assert _rbo(preds, labels, p=0.8) == pytest.approx(14 / 15)


def test_rbo_tied_predictions_matches_hand_computation():
    # Same as above with preds/labels roles swapped - agreement is symmetric
    # in seen_pred/seen_true, so the result is identical.
    preds = [5.0, 5.0, 1.0]
    labels = [3.0, 2.0, 1.0]
    assert _rbo(preds, labels, p=0.8) == pytest.approx(14 / 15)


def test_rbo_full_tie_at_top_of_both_rankings_is_one():
    # Both preds and labels tie items 0 and 1 for the top rank, so the
    # induced (competition) rankings are identical -> RBO = 1, unlike an
    # arbitrary index-order tiebreak which could split the tie and understate
    # agreement.
    preds = [1.0, 1.0, 0.5]
    labels = [2.0, 2.0, 1.0]
    assert _rbo(preds, labels, p=0.8) == pytest.approx(1.0)


def test_reciprocal_rank_true_best_predicted_first():
    preds = [3.0, 2.0, 1.0]
    labels = [10.0, 5.0, 1.0]  # true best is index 0, also predicted top
    assert _reciprocal_rank(preds, labels) == pytest.approx(1.0)


def test_reciprocal_rank_true_best_predicted_last():
    preds = [1.0, 2.0, 3.0]
    labels = [10.0, 5.0, 1.0]  # true best is index 0, predicted rank 3
    assert _reciprocal_rank(preds, labels) == pytest.approx(1 / 3)


def test_reciprocal_rank_tie_for_best_uses_earliest_predicted_rank():
    preds = [5.0, 1.0, 1.0]
    labels = [10.0, 10.0, 1.0]  # indices 0 and 1 tie for best label
    # index 0 is predicted rank 1, so the tie resolves to RR=1.0
    assert _reciprocal_rank(preds, labels) == pytest.approx(1.0)


def test_reciprocal_rank_nan_for_fewer_than_two_items():
    assert np.isnan(_reciprocal_rank([1.0], [1.0]))
