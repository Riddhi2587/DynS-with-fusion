"""
build_query_only_feature_sanity_report / build_feature_sanity_report should
summarize the query (and, for the latter, doc) feature tensors a built
QPPQueryOnlyDataset / QPPDataset holds - shape metadata plus per-feature
min/max/mean/nonzero stats - and produce a JSON-serializable dict, since it's
meant to be dumped straight to a log file.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_metrics_csv import build_metrics_csv
from dataset import (
    QPPDataset,
    QPPQueryOnlyDataset,
    build_feature_sanity_report,
    build_query_only_feature_sanity_report,
)
from feature_cache import FeatureCache, build_doc_feature_cache
from features import (
    ALL_FEATURE_BLOCKS,
    DOC_FEATURE_DIM,
    ENTITY_COUNT_FEATURE_DIM,
    SCS_PMI_FEATURE_DIM,
    LEXICAL_FEATURE_NAMES,
    LIST_FEATURE_DIM,
    LIST_FEATURE_NAMES,
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
    """Deterministic fake IndexStats - build_query_features only needs .idf();
    doc_term_counts/bm25_tf back build_doc_features for the QPPDataset tests."""

    def idf(self, term):
        return float(len(term))

    def doc_term_counts(self, doc_id):
        return Counter({doc_id: 1})

    def bm25_tf(self, term, tf_map, doc_len, k1=1.2, b=0.75):
        return float(tf_map.get(term, 0))


class RaisingIndexStats:
    """A live-Lucene-fallback object whose methods raise if ever called - a
    strong regression guard that QPPDataset never falls back to it when a
    doc_feature_cache (require_caches=True) fully covers the data."""

    def idf(self, term):
        raise AssertionError("idf() should never be called when caches are complete")

    def doc_term_counts(self, doc_id):
        raise AssertionError("doc_term_counts() should never be called when caches are complete")

    def bm25_tf(self, term, tf_map, doc_len, k1=1.2, b=0.75):
        raise AssertionError("bm25_tf() should never be called when caches are complete")


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

ENTITY_COUNT_LOOKUP = {"q1": 1.0, "q2": 2.0, "q3": 0.0}
SCS_PMI_LOOKUP = {q: [0.1, 0.2, 0.3] for q in ("q1", "q2", "q3")}

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
    build_metrics_csv(run_paths, qrels_path, str(p))
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
    assert block["n_doc_rows"] == 0
    assert block["n_query_rows"] == len(train_ds)
    assert block["dead_term_features"] == []
    assert set(block["query_features"]) == set(EXPECTED_QUERY_FEATURE_NAMES)
    for name in EXPECTED_QUERY_FEATURE_NAMES:
        assert block["query_features"][name]["n"] == len(train_ds)


def test_report_is_json_serializable(train_ds):
    report = build_query_only_feature_sanity_report(train_ds)
    json.dumps(report)  # must not raise


@pytest.fixture
def qpp_train_ds(run_paths, qrels_path, metrics_csv_path):
    # Explicitly selects a subset (unlike the true ALL_FEATURE_BLOCKS
    # default - see test_qppdataset_default_feature_blocks_is_all_six below)
    # since this fixture backs several tests that only care about the
    # query/list/doc report shape.
    return QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        embedding_lookup=EMBEDDING_LOOKUP,
        query_type_classifier=DummyQueryTypeClassifier(),
        metrics_csv=metrics_csv_path,
        feature_blocks=("lexical", "embedding", "query_type", "doc_feats"),
    )


def test_qppdataset_report_shape_and_metadata(qpp_train_ds):
    """QPPDataset's query_feats must be the FULL (lexical+embedding+
    query_type) vector, not just the 5-dim lexical vector - covering every
    expected query feature name, same as QPPQueryOnlyDataset, plus the
    doc-feature and list-feature blocks QPPQueryOnlyDataset doesn't have."""
    report = build_feature_sanity_report(qpp_train_ds)

    assert report["num_samples"] == len(qpp_train_ds)
    assert report["num_rankers"] == qpp_train_ds.num_rankers
    assert report["query_feats_shape"] == [EXPECTED_QUERY_FEATURE_DIM]
    assert report["doc_feats_shape"] == [qpp_train_ds.num_rankers, qpp_train_ds.top_k, DOC_FEATURE_DIM]
    assert DOC_FEATURE_DIM == 9
    assert report["list_feats_shape"] == [qpp_train_ds.num_rankers, LIST_FEATURE_DIM]
    assert report["query_feature_names"] == EXPECTED_QUERY_FEATURE_NAMES
    assert report["list_feature_names"] == LIST_FEATURE_NAMES

    block = report["branches"]["original"]
    assert block["n_query_rows"] == len(qpp_train_ds)
    assert block["n_doc_rows"] > 0
    assert block["n_list_rows"] > 0
    assert set(block["query_features"]) == set(EXPECTED_QUERY_FEATURE_NAMES)
    assert set(block["list_features"]) == set(LIST_FEATURE_NAMES)
    for name in EXPECTED_QUERY_FEATURE_NAMES:
        assert block["query_features"][name]["n"] == len(qpp_train_ds)


def test_qppdataset_report_is_json_serializable(qpp_train_ds):
    report = build_feature_sanity_report(qpp_train_ds)
    json.dumps(report)  # must not raise


def test_qppdataset_list_feats_are_raw_mean_var_once_per_ranker(qpp_train_ds):
    """mean_score/var_score must land in list_feats, computed once per
    (query, ranker) from that ranker's real scores - not duplicated into
    doc_feats (which is why doc_feats' last dim shrank to 9)."""
    bm25_id = qpp_train_ds.ranker_to_id["bm25"]
    rm3_id = qpp_train_ds.ranker_to_id["rm3"]
    sample_q1 = next(s for s in qpp_train_ds.samples if s["qid"] == "q1")

    assert sample_q1["doc_feats"].shape[-1] == 9

    # bm25.res q1: scores [3.0, 2.0] -> mean=2.5, var=0.25
    bm25_list_feats = sample_q1["list_feats"][bm25_id]
    assert bm25_list_feats == pytest.approx([2.5, 0.25])

    # rm3.res q1: scores [4.0] -> mean=4.0, var=0.0
    rm3_list_feats = sample_q1["list_feats"][rm3_id]
    assert rm3_list_feats == pytest.approx([4.0, 0.0])


def test_qppdataset_default_feature_blocks_is_all_six(run_paths, qrels_path, metrics_csv_path):
    """Omitting feature_blocks must reproduce today's behavior exactly -
    regression safety for every pre-existing QPPDataset caller/test. Unlike
    qpp_train_ds (which explicitly selects a subset), this constructs its own
    dataset with feature_blocks genuinely omitted."""
    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        embedding_lookup=EMBEDDING_LOOKUP,
        query_type_classifier=DummyQueryTypeClassifier(),
        entity_count_lookup=ENTITY_COUNT_LOOKUP,
        scs_pmi_lookup=SCS_PMI_LOOKUP,
        metrics_csv=metrics_csv_path,
    )
    assert ds.feature_blocks == ALL_FEATURE_BLOCKS
    # EXPECTED_QUERY_FEATURE_DIM is the legacy fixed layout (lexical+embedding+
    # query_type only, no entity_count) - the true default adds entity_count
    # on top of it (see features.resolve_query_feature_layout).
    assert ds.query_feature_dim == EXPECTED_QUERY_FEATURE_DIM + ENTITY_COUNT_FEATURE_DIM + SCS_PMI_FEATURE_DIM
    assert ds.doc_feature_dim == DOC_FEATURE_DIM
    assert ds.list_feature_dim == LIST_FEATURE_DIM


def test_qppdataset_feature_blocks_lexical_only(run_paths, qrels_path, metrics_csv_path):
    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        metrics_csv=metrics_csv_path,
        feature_blocks=("lexical",),
    )
    assert ds.feature_blocks == ("lexical",)
    assert ds.query_feature_dim == 5
    assert ds.doc_feature_dim == 0
    assert ds.list_feature_dim == 0
    assert ds.embedding_slice is None

    sample = ds.samples[0]
    assert sample["query_feats"].shape == (5,)
    assert sample["doc_feats"].shape == (ds.num_rankers, ds.top_k, 0)
    assert sample["list_feats"].shape == (ds.num_rankers, 0)


def test_qppdataset_feature_blocks_doc_feats_only_no_query_pieces_needed(run_paths, qrels_path, metrics_csv_path):
    """No lexical_cache/embedding_lookup/query_type_classifier at all - none
    of the 3 query pieces should be touched since only doc_feats is selected."""
    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        metrics_csv=metrics_csv_path,
        feature_blocks=("doc_feats",),
    )
    assert ds.query_feature_dim == 0
    assert ds.doc_feature_dim == DOC_FEATURE_DIM
    assert ds.list_feature_dim == LIST_FEATURE_DIM

    sample = ds.samples[0]
    assert sample["query_feats"].shape == (0,)
    assert sample["doc_feats"].shape[-1] == DOC_FEATURE_DIM


def test_qppdataset_feature_blocks_query_only_needs_no_index_stats(run_paths, qrels_path, metrics_csv_path):
    """Excluding doc_feats means zero Lucene work at all - same efficiency
    QPPQueryOnlyDataset already has - so this must build fine with
    index_stats=None as long as the selected query pieces have another
    source (here: a lexical_cache covering every qid, plus the embedding/
    query_type live fallbacks already used elsewhere in this file)."""
    lexical_cache = FeatureCache(
        query_feats={qid: np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32) for qid in QUERIES}
    )
    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, index_stats=None,
        metrics_csv=metrics_csv_path,
        lexical_cache=lexical_cache,
        embedding_lookup=EMBEDDING_LOOKUP,
        query_type_classifier=DummyQueryTypeClassifier(),
        feature_blocks=("lexical", "embedding", "query_type"),
    )
    assert ds.doc_feature_dim == 0
    assert ds.list_feature_dim == 0
    assert ds.query_feature_dim == EXPECTED_QUERY_FEATURE_DIM
    for s in ds.samples:
        assert s["doc_feats"].shape == (ds.num_rankers, ds.top_k, 0)


def test_qppdataset_feature_blocks_report_reflects_selected_subset(qpp_train_ds, run_paths, qrels_path, metrics_csv_path):
    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        metrics_csv=metrics_csv_path,
        feature_blocks=("lexical", "doc_feats"),
    )
    report = build_feature_sanity_report(ds)
    assert report["feature_blocks"] == ["lexical", "doc_feats"]
    assert report["query_feats_shape"] == [5]
    assert report["doc_feats_shape"][-1] == DOC_FEATURE_DIM
    assert report["list_feats_shape"] == [ds.num_rankers, LIST_FEATURE_DIM]

    block = report["branches"]["original"]
    assert set(block["query_features"]) == {
        "num_query_terms", "num_unique_query_terms", "min_idf", "max_idf", "sum_idf",
    }
    assert block["n_doc_rows"] > 0
    assert block["n_list_rows"] > 0


def test_qppdataset_doc_feature_cache_needs_zero_lucene_calls(run_paths, qrels_path, metrics_csv_path):
    """A doc_feature_cache that fully covers every (qid, doc_id) pair in the
    run files, combined with require_caches=True, must let QPPDataset build
    with index_stats=None and never touch Lucene at all - the concrete
    regression test for 'I don't want to use Lucene during training or
    eval'. RaisingIndexStats would AssertionError if the live-fallback path
    were ever reached."""
    doc_feature_cache = build_doc_feature_cache(run_paths, QUERIES, DummyIndexStats())
    assert doc_feature_cache.meta["num_doc_pairs"] == 4  # (q1,d1) (q1,d2) (q2,d3) (q3,d4)

    ds = QPPDataset(
        run_paths, qrels_path, QUERIES, index_stats=None,
        metrics_csv=metrics_csv_path,
        doc_feature_cache=doc_feature_cache,
        require_caches=True,
        feature_blocks=("doc_feats",),
    )
    assert ds.doc_feature_dim == DOC_FEATURE_DIM
    assert len(ds.samples) > 0

    # Baseline built via live Lucene (DummyIndexStats, not raising) must
    # produce numerically identical doc_feats.
    baseline = QPPDataset(
        run_paths, qrels_path, QUERIES, DummyIndexStats(),
        metrics_csv=metrics_csv_path,
        feature_blocks=("doc_feats",),
    )
    cached_by_qid = {s["qid"]: s["doc_feats"] for s in ds.samples}
    baseline_by_qid = {s["qid"]: s["doc_feats"] for s in baseline.samples}
    for qid in baseline_by_qid:
        np.testing.assert_array_equal(cached_by_qid[qid], baseline_by_qid[qid])

    # A second construction with RaisingIndexStats passed as index_stats
    # (still require_caches=True) must ALSO never call it - proves the
    # cache-hit path is taken even when a live fallback object is present.
    ds_with_raising_fallback = QPPDataset(
        run_paths, qrels_path, QUERIES, RaisingIndexStats(),
        metrics_csv=metrics_csv_path,
        doc_feature_cache=doc_feature_cache,
        require_caches=True,
        feature_blocks=("doc_feats",),
    )
    assert len(ds_with_raising_fallback.samples) == len(ds.samples)


def test_qppdataset_doc_feature_cache_miss_raises_under_require_caches(run_paths, qrels_path, metrics_csv_path):
    doc_feature_cache = build_doc_feature_cache(run_paths, QUERIES, DummyIndexStats())
    del doc_feature_cache.doc_content_feats[("q1", "d1")]

    with pytest.raises(RuntimeError, match=r"doc_feature_cache.*q1.*d1"):
        QPPDataset(
            run_paths, qrels_path, QUERIES, index_stats=None,
            metrics_csv=metrics_csv_path,
            doc_feature_cache=doc_feature_cache,
            require_caches=True,
            feature_blocks=("doc_feats",),
        )
