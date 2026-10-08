"""
build_query_only_feature_sanity_report should summarize the query feature
tensors a built QPPQueryOnlyDataset holds - shape metadata plus per-feature
min/max/mean/nonzero stats - and produce a JSON-serializable dict, since it's
meant to be dumped straight to a log file.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics_helper import write_metrics_csv
from dataset import QPPQueryOnlyDataset, build_query_only_feature_sanity_report
from features import (
    LEXICAL_FEATURE_NAMES,
    QUERY_TYPE_FEATURE_NAMES,
    embedding_feature_names,
)

# Realistic raw embedding width (e.g. BERT/Contriever CLS) - deliberately NOT
# features.EMBEDDING_DIM (32), which is now only the model's learned
# projection's output width, unrelated to the raw cache width tested here.
RAW_EMBEDDING_DIM = 768
# The legacy fixed layout's expected dim/names, computed the same way
# QPPQueryOnlyDataset now computes them (LEXICAL_FEATURE_DIM + raw_dim +
# QUERY_TYPE_FEATURE_DIM) - there is no fixed QUERY_FEATURE_DIM/
# QUERY_FEATURE_NAMES constant to import anymore, since the embedding
# component of that layout is a variable raw width, not fixed.
EXPECTED_QUERY_FEATURE_NAMES = (
    LEXICAL_FEATURE_NAMES + embedding_feature_names(RAW_EMBEDDING_DIM) + QUERY_TYPE_FEATURE_NAMES
)
EXPECTED_QUERY_FEATURE_DIM = len(EXPECTED_QUERY_FEATURE_NAMES)


class DummyIndexStats:
    """Deterministic fake IndexStats - build_query_features only needs .idf()."""

    def idf(self, term):
        return float(len(term))


class DummyQueryTypeClassifier:
    """Deterministic fake QueryTypeClassifier - no network/model download.
    Mimics question-detection with a trivial rule: ends with "?" -> 1.0."""

    def classify(self, query):
        return 1.0 if query.strip().endswith("?") else 0.0


QUERIES = {
    "q1": "one two three",
    "q2": "four five?",
    "q3": "six",
}

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
    p = tmp_path / "metrics.csv"
    write_metrics_csv(run_paths, qrels_path, str(p))
    return str(p)


@pytest.fixture
def train_ds(run_paths, qrels_path, metrics_csv_path):
    return QPPQueryOnlyDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        embedding_lookup=EMBEDDING_LOOKUP,
        query_type_classifier=DummyQueryTypeClassifier(),
        metrics_csv=metrics_csv_path,
    )


def test_report_shape_and_metadata(train_ds):
    report = build_query_only_feature_sanity_report(train_ds)

    assert report["num_samples"] == len(train_ds)
    assert report["num_rankers"] == train_ds.num_rankers
    assert report["query_feats_shape"] == [EXPECTED_QUERY_FEATURE_DIM]
    assert report["query_feature_names"] == EXPECTED_QUERY_FEATURE_NAMES

    block = report["branches"]["original"]
    assert block["n_query_rows"] == len(train_ds)
    assert set(block["query_features"]) == set(EXPECTED_QUERY_FEATURE_NAMES)
    for name in EXPECTED_QUERY_FEATURE_NAMES:
        assert block["query_features"][name]["n"] == len(train_ds)


def test_report_is_json_serializable(train_ds):
    report = build_query_only_feature_sanity_report(train_ds)
    json.dumps(report)  # must not raise
