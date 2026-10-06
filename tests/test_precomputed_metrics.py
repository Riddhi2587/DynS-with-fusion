"""
Tests for the precomputed-metrics pipeline (dataset.load_precomputed_metrics,
QPPDataset/QPPQueryOnlyDataset's metrics_csv, evaluate.compute_labels_matrix,
build_metrics_csv.py). See guide_docs/PRECOMPUTED_METRICS_GUIDE.md.

pytrec_eval is banned from train/eval code (dataset.py/evaluate.py no longer
import it at all) - it's used in this file only to build a golden CSV and
independently verify build_metrics_csv.py's output, which is a test concern,
not a training/eval one.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import pytrec_eval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_metrics_csv import build_metrics_csv
from dataset import QPPDataset, QPPQueryOnlyDataset, load_precomputed_metrics, load_qrels, load_run
from evaluate import compute_labels_matrix

# Realistic raw embedding width (e.g. BERT/Contriever CLS) - these tests
# only care about label correctness, not embedding values, but a realistic
# width keeps the fixture honest about what real datasets look like now
# that the raw embedding is variable-width, not a fixed 32-dim PCA vector.
RAW_EMBEDDING_DIM = 768


class DummyIndexStats:
    """Deterministic fake IndexStats - just enough for build_doc_features/
    build_query_features to run without crashing; these tests only care
    about label correctness, not feature values."""

    def idf(self, term):
        return float(len(term))

    def doc_term_counts(self, doc_id):
        return Counter({doc_id: 1})

    def bm25_tf(self, term, tf_map, doc_len, k1=1.2, b=0.75):
        return float(tf_map.get(term, 0))


class DummyQueryTypeClassifier:
    """Deterministic fake QueryTypeClassifier - no network/model download;
    these tests only care about label correctness, not feature values."""

    def classify(self, query):
        return 0.0


QUERIES = {
    "q1": "one two",
    "q2": "three four",
    "q3": "five",
}

EMBEDDING_LOOKUP = {qid: np.zeros(RAW_EMBEDDING_DIM, dtype=np.float32) for qid in QUERIES}

RUN_ROWS = {
    "bm25.res": [
        ("q1", "d1", 1, 3.0),
        ("q2", "d2", 1, 1.0),
    ],
    "rm3.res": [
        ("q1", "d1", 1, 4.0),
        ("q3", "d3", 1, 0.5),
    ],
}

QRELS_ROWS = [
    ("q1", "d1", 1),
    ("q2", "d2", 1),
    ("q3", "d3", 1),
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


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_precomputed_metrics_parses_csv(tmp_path):
    p = tmp_path / "metrics.csv"
    _write_csv(
        p,
        [
            {"qid": "q1", "ranker": "bm25", "ndcg_cut_100": "0.5"},
            {"qid": "q2", "ranker": "bm25", "ndcg_cut_100": "0.7"},
        ],
        ["qid", "ranker", "ndcg_cut_100"],
    )
    lookup = load_precomputed_metrics(str(p))
    assert lookup[("bm25", "q1")] == {"ndcg_cut_100": 0.5}
    assert lookup[("bm25", "q2")] == {"ndcg_cut_100": 0.7}


def test_load_precomputed_metrics_raises_on_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    _write_csv(p, [], ["qid", "ranker", "ndcg_cut_100"])
    with pytest.raises(ValueError, match="No rows loaded"):
        load_precomputed_metrics(str(p))


@pytest.fixture
def metrics_csv_path(tmp_path):
    """Deliberately omits (bm25, q1) so its label must resolve to 0.0."""
    p = tmp_path / "metrics.csv"
    _write_csv(
        p,
        [
            {"qid": "q2", "ranker": "bm25", "ndcg_cut_100": "0.7"},
            {"qid": "q1", "ranker": "rm3", "ndcg_cut_100": "0.9"},
            {"qid": "q3", "ranker": "rm3", "ndcg_cut_100": "0.3"},
        ],
        ["qid", "ranker", "ndcg_cut_100"],
    )
    return str(p)


@pytest.fixture
def metrics_csv_missing_ndcg(tmp_path):
    p = tmp_path / "metrics_no_ndcg.csv"
    _write_csv(
        p,
        [{"qid": "q1", "ranker": "bm25", "map_cut_50": "0.4"}],
        ["qid", "ranker", "map_cut_50"],
    )
    return str(p)


def test_qppdataset_labels_come_from_metrics_csv(run_paths, qrels_path, metrics_csv_path):
    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=DummyQueryTypeClassifier(),
        metrics_csv=metrics_csv_path,
        feature_blocks=("lexical", "embedding", "query_type", "doc_feats"),
    )
    labels_by_qid = {s["qid"]: s["labels"] for s in ds.samples}
    bm25_id, rm3_id = ds.ranker_to_id["bm25"], ds.ranker_to_id["rm3"]

    assert labels_by_qid["q1"][bm25_id] == pytest.approx(0.0)   # missing row
    assert labels_by_qid["q1"][rm3_id] == pytest.approx(0.9)
    assert labels_by_qid["q2"][bm25_id] == pytest.approx(0.7)
    assert labels_by_qid["q3"][rm3_id] == pytest.approx(0.3)


def test_qppqueryonlydataset_labels_come_from_metrics_csv(run_paths, qrels_path, metrics_csv_path):
    ds = QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        embedding_lookup=EMBEDDING_LOOKUP, query_type_classifier=DummyQueryTypeClassifier(),
        metrics_csv=metrics_csv_path,
    )
    labels_by_qid = {s["qid"]: s["labels"] for s in ds.samples}
    bm25_id, rm3_id = ds.ranker_to_id["bm25"], ds.ranker_to_id["rm3"]

    assert labels_by_qid["q1"][bm25_id] == pytest.approx(0.0)   # missing row
    assert labels_by_qid["q1"][rm3_id] == pytest.approx(0.9)
    assert labels_by_qid["q2"][bm25_id] == pytest.approx(0.7)
    assert labels_by_qid["q3"][rm3_id] == pytest.approx(0.3)


def test_qppdataset_requires_metrics_csv(run_paths, qrels_path):
    with pytest.raises(ValueError, match="metrics_csv"):
        QPPDataset(run_paths, qrels_path, QUERIES, DummyIndexStats())


def test_qppqueryonlydataset_requires_metrics_csv(run_paths, qrels_path):
    with pytest.raises(ValueError, match="metrics_csv"):
        QPPQueryOnlyDataset(run_paths, qrels_path, QUERIES, DummyIndexStats())


def test_missing_metric_column_raises(run_paths, qrels_path, metrics_csv_missing_ndcg):
    # embedding_lookup is required here purely so the raw embedding width
    # can be resolved at construction time (feature_blocks includes
    # "embedding") - this test is about the missing-metric-column error, not
    # embeddings, so it just needs SOME valid embedding source to reach that
    # check.
    with pytest.raises(ValueError, match="ndcg_cut_100"):
        QPPDataset(
            run_paths, qrels_path, QUERIES, DummyIndexStats(),
            metrics_csv=metrics_csv_missing_ndcg,
            feature_blocks=("lexical", "embedding", "query_type", "doc_feats"),
            embedding_lookup=EMBEDDING_LOOKUP,
        )
    with pytest.raises(ValueError, match="ndcg_cut_100"):
        QPPQueryOnlyDataset(
            run_paths, qrels_path, QUERIES, DummyIndexStats(),
            metrics_csv=metrics_csv_missing_ndcg,
            embedding_lookup=EMBEDDING_LOOKUP,
        )


def test_compute_labels_matrix_reads_from_csv(metrics_csv_path):
    id_to_ranker = {0: "bm25", 1: "rm3"}
    qids = ["q1", "q2", "q3"]
    labels = compute_labels_matrix(qids, id_to_ranker, "ndcg_cut.100", metrics_csv_path)

    expected = np.array([
        [0.0, 0.9],   # q1: bm25 missing -> 0.0, rm3 -> 0.9
        [0.7, 0.0],   # q2: bm25 -> 0.7, rm3 has no row -> 0.0
        [0.0, 0.3],   # q3: bm25 has no row -> 0.0, rm3 -> 0.3
    ], dtype=np.float32)
    np.testing.assert_array_almost_equal(labels, expected)


def test_build_metrics_csv_matches_direct_pytrec_eval(tmp_path, run_paths, qrels_path):
    """End-to-end sanity check (the guide's own recommended spot-check):
    build_metrics_csv.py's output, read back via load_precomputed_metrics,
    must match a direct pytrec_eval computation on the same run/qrels."""
    out_path = tmp_path / "built_metrics.csv"
    build_metrics_csv(run_paths, qrels_path, str(out_path))

    lookup = load_precomputed_metrics(str(out_path))

    runs = load_run(run_paths)
    qrels = load_qrels(qrels_path)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.100"})
    for ranker, run_data in runs.items():
        run_for_eval = {qid: dict(doc_list) for qid, doc_list in run_data.items()}
        results = evaluator.evaluate(run_for_eval)
        for qid, scores in results.items():
            expected = scores.get("ndcg_cut_100", 0.0)
            actual = lookup.get((ranker, qid), {}).get("ndcg_cut_100", 0.0)
            assert actual == pytest.approx(expected), (ranker, qid)
