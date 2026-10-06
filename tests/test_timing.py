"""
Tests for the runtime-measurement code (timing_utils, time_feature_computation,
time_eval_live). Everything uses stubs - a fake IndexStats, a fake sentence
encoder, a fake query-type pipeline - so no GPU, Lucene index or network access
is needed.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import timing_utils
from embedding_live import MiniLMEncoder, make_query_type_classifier
from feature_cache import build_embedding_cache, build_feature_cache, build_query_type_cache
from features import build_query_features, resolve_doc_feature_layout, resolve_query_feature_layout
from model import QPPMLP
from time_eval_live import CSV_HEADER, resolve_live_blocks, time_queries
from time_feature_computation import ALL_STEPS, time_dataset, time_lexical
from timing_utils import SUMMARY_HEADER, Timer, append_rows_csv, summary_row

EMB_DIM = 8


class FakeIndexStats:
    def __init__(self):
        self._idf_cache = {}

    def idf(self, term):
        self._idf_cache[term] = len(term) / 10.0
        return self._idf_cache[term]


class FakeEncoder:
    device = torch.device("cpu")

    def encode(self, texts, batch_size=32):
        return np.stack([np.full(EMB_DIM, float(len(t)), dtype=np.float32) for t in texts])


def fake_pipeline(texts):
    return [{"label": f"LABEL_{len(t) % 2}"} for t in texts]


QUERIES = {"1": "what is a llama", "2": "llama farm", "3": "how do llamas sleep at night"}


def write_run(tmp_path):
    path = tmp_path / "toy.res"
    lines = [f"{qid} Q0 d{qid} 1 5.0 toy" for qid in QUERIES]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def make_model(blocks, num_rankers=3, top_k=10):
    dim, _, emb_slice = resolve_query_feature_layout(
        blocks, embedding_dim=EMB_DIM if "embedding" in blocks else None
    )
    doc_dim, list_dim, _, _ = resolve_doc_feature_layout(blocks)
    return QPPMLP(
        doc_feature_dim=doc_dim, list_feature_dim=list_dim, query_feature_dim=dim,
        embedding_slice=emb_slice, top_k=top_k, hidden_dims=[16], num_rankers=num_rankers,
        reduce_embedding=False,
    )


# ---- timing_utils -----------------------------------------------------------

def test_timer_is_nonnegative_and_syncs_only_for_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with Timer(torch.device("cuda")) as t:
        pass
    assert t.seconds >= 0 and calls == []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with Timer(torch.device("cuda")) as t:
        pass
    assert len(calls) == 2  # before and after the region
    with Timer(torch.device("cpu")):
        pass
    with Timer():
        pass
    assert len(calls) == 2  # CPU / no-device timers never sync


def test_append_rows_csv_writes_header_once(tmp_path):
    path = str(tmp_path / "sub" / "t.csv")
    append_rows_csv(path, SUMMARY_HEADER, [summary_row("d", "lexical", 2.0, 4, "cpu")])
    append_rows_csv(path, SUMMARY_HEADER, [summary_row("d", "embedding", 1.0, None, "cpu")])
    rows = list(csv.DictReader(open(path)))
    assert [r["step"] for r in rows] == ["lexical", "embedding"]
    assert rows[0]["mean_per_item_s"] == "0.500000" and rows[1]["mean_per_item_s"] == ""
    assert open(path).read().count("dataset,step") == 1


# ---- time_feature_computation ----------------------------------------------

def test_time_lexical_matches_build_feature_cache(tmp_path):
    run = write_run(tmp_path)
    from dataset import load_run

    idx = FakeIndexStats()
    feats, seconds, n = time_lexical(QUERIES, load_run([run]), idx)
    ref = build_feature_cache([run], QUERIES, FakeIndexStats())
    assert n == len(ref.query_feats) == 3 and seconds >= 0
    for qid, vec in ref.query_feats.items():
        np.testing.assert_array_equal(feats[qid], vec)


def test_time_dataset_rows_and_caches_match_builders(tmp_path):
    from dataset import load_run

    runs = load_run([write_run(tmp_path)])
    clf = make_query_type_classifier(pipeline_fn=fake_pipeline)
    idx = FakeIndexStats()
    rows, caches = time_dataset(
        "toy", QUERIES, runs, idx, clf, FakeEncoder(), torch.device("cpu"), batch_size=2,
    )
    assert [r["step"] for r in rows] == [
        "lexical", "query_type", "query_type_batched",
        "embedding_encode", "embedding_normalize", "embedding_total",
    ]
    assert all(r["dataset"] == "toy" and float(r["seconds"]) >= 0 for r in rows)
    ref_qt = build_query_type_cache(QUERIES, clf)
    for qid in QUERIES:
        np.testing.assert_array_equal(caches["query_type"].query_feats[qid], ref_qt.query_feats[qid])
    lookup = {qid: FakeEncoder().encode([raw])[0] for qid, raw in QUERIES.items()}
    ref_emb = build_embedding_cache(QUERIES, lookup)
    for qid in QUERIES:
        np.testing.assert_allclose(caches["embedding"].query_feats[qid], ref_emb.query_feats[qid])
    assert caches["embedding"].meta["embedding_dim"] == EMB_DIM


def test_time_dataset_with_no_steps_does_nothing():
    rows, caches = time_dataset(
        "toy", QUERIES, None, None, None, None, torch.device("cpu"), steps=(),
    )
    assert rows == [] and caches == {}


def test_all_steps_constant():
    assert ALL_STEPS == ("lexical", "query_type", "embedding")


# ---- time_eval_live ---------------------------------------------------------

def test_time_queries_one_row_per_qid_and_live_vectors_correct():
    blocks = ("lexical", "embedding", "query_type")
    idx = FakeIndexStats()
    clf = make_query_type_classifier(pipeline_fn=fake_pipeline)
    model = make_model(blocks)
    rows, live = time_queries(
        "toy", QUERIES, blocks, idx, FakeEncoder(), clf, model, torch.device("cpu"),
        num_rankers=3, top_k=10, warmup=1, keep_vectors_for=2,
    )
    assert [r["qid"] for r in rows] == list(QUERIES)
    assert list(rows[0].keys()) == CSV_HEADER
    for r in rows:
        parts = [float(r[k]) for k in ("lexical_s", "embedding_s", "query_type_s", "assembly_s", "mlp_s")]
        assert all(p >= 0 for p in parts)
        assert float(r["total_s"]) == pytest.approx(sum(parts), abs=1e-5)
        assert r["device"] == "cpu"
    assert len(live) == 2
    qid = "1"
    np.testing.assert_array_equal(
        live[qid]["lexical"], build_query_features(QUERIES[qid].lower().split(), FakeIndexStats())
    )
    assert live[qid]["embedding"].shape == (EMB_DIM,)
    assert np.linalg.norm(live[qid]["embedding"]) == pytest.approx(1.0, abs=1e-5)


def test_time_queries_clears_idf_cache_per_query_by_default():
    blocks = ("lexical",)
    idx = FakeIndexStats()
    model = make_model(blocks)
    rows, _ = time_queries(
        "toy", {"1": "aa bb", "2": "cc"}, blocks, idx, None, None, model,
        torch.device("cpu"), num_rankers=3, top_k=10, warmup=0,
    )
    assert set(idx._idf_cache) == {"cc"}  # only the last query's terms survive
    idx2 = FakeIndexStats()
    time_queries(
        "toy", {"1": "aa bb", "2": "cc"}, blocks, idx2, None, None, model,
        torch.device("cpu"), num_rankers=3, top_k=10, warmup=0, keep_idf_cache=True,
    )
    assert set(idx2._idf_cache) == {"aa", "bb", "cc"}


def test_time_queries_lexical_only_leaves_other_columns_blank():
    blocks = ("lexical",)
    model = make_model(blocks)
    assert model.query_feature_dim == 5
    rows, _ = time_queries(
        "toy", QUERIES, blocks, FakeIndexStats(), None, None, model,
        torch.device("cpu"), num_rankers=3, top_k=10, warmup=0,
    )
    for r in rows:
        assert r["embedding_s"] == "" and r["query_type_s"] == ""
        assert float(r["lexical_s"]) >= 0 and float(r["mlp_s"]) >= 0


def test_resolve_live_blocks():
    assert resolve_live_blocks(["query_type", "lexical"]) == ("lexical", "query_type")
    with pytest.raises(ValueError, match="no live"):
        resolve_live_blocks(["lexical", "doc_feats"])


def test_encoder_wrapper_uses_injected_model_and_returns_float32():
    class M:
        def encode(self, texts, **kw):
            assert kw["normalize_embeddings"] is False
            return np.ones((len(texts), 4), dtype=np.float64)

    enc = MiniLMEncoder(device=torch.device("cpu"), model=M()).load()
    out = enc.encode(["a", "b"])
    assert out.shape == (2, 4) and out.dtype == np.float32


# ---- --num_queries random sampling (time_feature_computation) -----------------

import json

import time_feature_computation as tfc

POOL_QUERIES = {str(i): f"query number {i} about llamas" for i in range(1, 13)}


def toy_runs(qids):
    return {"toy": {qid: [("d1", 1.0)] for qid in qids}}


def test_sample_queries_size_membership_and_determinism():
    runs = toy_runs(POOL_QUERIES)
    sample, pool = tfc.sample_queries("d", POOL_QUERIES, runs, 5, seed=42)
    assert len(sample) == 5 and pool == 12
    assert set(sample) <= set(POOL_QUERIES)
    assert all(sample[q] == POOL_QUERIES[q] for q in sample)
    again, _ = tfc.sample_queries("d", POOL_QUERIES, runs, 5, seed=42)
    assert list(again) == list(sample)  # same seed -> same qids, same order
    other, _ = tfc.sample_queries("d", POOL_QUERIES, runs, 5, seed=7)
    assert set(other) != set(sample)


def test_sample_queries_independent_of_input_order():
    runs = toy_runs(POOL_QUERIES)
    shuffled = dict(reversed(list(POOL_QUERIES.items())))
    a, _ = tfc.sample_queries("d", POOL_QUERIES, runs, 5, seed=1)
    b, _ = tfc.sample_queries("d", shuffled, runs, 5, seed=1)
    assert list(a) == list(b)


def test_sample_queries_excludes_empty_text_and_qids_without_runs():
    queries = {"1": "ok query", "2": "   ", "3": "has no run", "4": "another ok"}
    runs = toy_runs(["1", "2", "4"])  # "3" has no run; "2" is blank
    sample, pool = tfc.sample_queries("d", queries, runs, 2, seed=0)
    assert pool == 2 and set(sample) == {"1", "4"}


def test_sample_queries_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="only 12"):
        tfc.sample_queries("d", POOL_QUERIES, toy_runs(POOL_QUERIES), 13, seed=0)


def test_parse_num_queries():
    names = {"llm-judged", "msmarco-dev"}
    assert tfc.parse_num_queries(["llm-judged=100", "msmarco-dev=300"], names) == {
        "llm-judged": 100, "msmarco-dev": 300,
    }
    assert tfc.parse_num_queries(None, names) == {}
    for bad in (["nope=5"], ["llm-judged"], ["llm-judged=abc"], ["llm-judged=0"],
                ["llm-judged=5", "llm-judged=6"]):
        with pytest.raises(ValueError):
            tfc.parse_num_queries(bad, names)


def _write_toy_dataset(tmp_path):
    q = tmp_path / "toy.tsv"
    q.write_text("".join(f"{qid}\t{text}\n" for qid, text in POOL_QUERIES.items()))
    run = tmp_path / "toy.res"
    run.write_text("".join(f"{qid} Q0 d1 1 5.0 toy\n" for qid in POOL_QUERIES))
    return str(q), str(run)


def _patch_models(monkeypatch):
    import features

    class StubEncoder(FakeEncoder):
        def __init__(self, device=None):
            pass

        def load(self):
            return self

    monkeypatch.setattr(features, "IndexStats", lambda path: FakeIndexStats())
    monkeypatch.setattr(tfc, "MiniLMEncoder", StubEncoder)
    monkeypatch.setattr(
        tfc, "make_query_type_classifier",
        lambda device=None: make_query_type_classifier(pipeline_fn=fake_pipeline),
    )


def test_main_samples_and_every_step_reports_n_items(tmp_path, monkeypatch):
    q, run = _write_toy_dataset(tmp_path)
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    argv = ["prog", "--dataset", "toy", q, run, "--index", "fake", "--device", "cpu",
            "--num_queries", "toy=5", "--seed", "3", "--output_dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    tfc.main()

    rows = list(csv.DictReader(open(out / "train_timing_summary.csv")))
    timed = [r for r in rows if r["step"] in (
        "lexical", "query_type", "query_type_batched", "embedding_encode",
        "embedding_normalize", "embedding_total")]
    assert len(timed) == 6 and all(r["n_items"] == "5" for r in timed)

    qids = (out / "sampled_qids_toy.txt").read_text().split()
    assert len(qids) == 5 and set(qids) <= set(POOL_QUERIES)
    env = json.load(open(out / "train_timing_summary.csv.env.json"))
    assert env["seed"] == 3
    assert env["sampling"]["toy"]["num_queries"] == 5 and env["sampling"]["toy"]["pool_size"] == 12

    # Same seed -> identical sample on a fresh run
    out2 = tmp_path / "out2"
    argv[argv.index("--output_dir") + 1] = str(out2)
    monkeypatch.setattr(sys, "argv", argv)
    tfc.main()
    assert (out2 / "sampled_qids_toy.txt").read_text().split() == qids


def test_main_without_num_queries_uses_all_queries(tmp_path, monkeypatch):
    q, run = _write_toy_dataset(tmp_path)
    _patch_models(monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--dataset", "toy", q, run, "--index", "fake", "--device", "cpu",
        "--output_dir", str(out)])
    tfc.main()
    rows = list(csv.DictReader(open(out / "train_timing_summary.csv")))
    assert {r["n_items"] for r in rows if r["step"] == "query_type"} == {"12"}
    assert not (out / "sampled_qids_toy.txt").exists()


def test_main_rejects_save_caches_with_num_queries(tmp_path, monkeypatch):
    q, run = _write_toy_dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--dataset", "toy", q, run, "--index", "fake", "--device", "cpu",
        "--num_queries", "toy=5", "--save_caches_dir", str(tmp_path / "caches"),
        "--output_dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="partial"):
        tfc.main()


def test_main_errors_when_asking_for_more_queries_than_pool(tmp_path, monkeypatch):
    q, run = _write_toy_dataset(tmp_path)
    _patch_models(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--dataset", "toy", q, run, "--index", "fake", "--device", "cpu",
        "--num_queries", "toy=99", "--output_dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="only 12"):
        tfc.main()


# ---- feature samples (feature_samples.py + both scripts) ----------------------

import feature_samples as fs
import time_eval_live as tel


def _correct_lexical(raw):
    return build_query_features(raw.lower().split(), FakeIndexStats())


def test_pick_sample_qids():
    qids = list(POOL_QUERIES)
    a = fs.pick_sample_qids(qids, 10, seed=1)
    assert len(a) == 10 and set(a) <= set(qids)
    assert fs.pick_sample_qids(qids, 10, seed=1) == a
    assert fs.pick_sample_qids(reversed(qids), 10, seed=1) == a  # input order irrelevant
    assert set(fs.pick_sample_qids(qids, 10, seed=2)) != set(a)
    assert sorted(fs.pick_sample_qids(qids, 99, seed=1)) == sorted(qids)  # fewer than n -> all


def test_describe_query_lexical_self_check():
    raw = "What is a llama llama"
    entry = fs.describe_query("1", raw, lexical=_correct_lexical(raw), index_stats=FakeIndexStats())
    lex = entry["lexical"]
    assert list(lex["values"]) == fs.LEXICAL_FEATURE_NAMES
    assert lex["tokens"] == ["what", "is", "a", "llama", "llama"]
    assert lex["per_term_idf"]["llama"] == pytest.approx(0.5)
    assert lex["values"]["num_query_terms"] == 5 and lex["values"]["num_unique_query_terms"] == 4
    assert lex["recomputed_matches"] is True

    bad = _correct_lexical(raw).copy()
    bad[4] += 1.0  # corrupt sum_idf
    assert fs.describe_query("1", raw, lexical=bad, index_stats=FakeIndexStats())[
        "lexical"]["recomputed_matches"] is False


def test_describe_query_without_index_stats_omits_idf_detail():
    raw = "llama farm"
    lex = fs.describe_query("1", raw, lexical=_correct_lexical(raw))["lexical"]
    assert set(lex) == {"values"}


def test_describe_query_embedding_query_type_and_omitted_blocks():
    emb = np.arange(1, EMB_DIM + 1, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    entry = fs.describe_query(
        "1", "q", embedding=emb, query_type=np.array([1.0], dtype=np.float32),
    )
    assert entry["embedding"]["dim"] == EMB_DIM
    assert entry["embedding"]["l2_norm"] == pytest.approx(1.0, abs=1e-6)
    assert len(entry["embedding"]["values"]) == EMB_DIM
    assert entry["query_type"] == {"value": 1.0, "label": "natural_language"}
    assert "lexical" not in entry
    assert fs.describe_query("1", "q", query_type=np.array([0.0]))["query_type"]["label"] == "keyword"
    assert set(fs.describe_query("1", "q")) == {"qid", "query"}


def test_write_feature_samples_round_trips(tmp_path):
    entry = fs.describe_query(
        "1", "llama farm", _correct_lexical("llama farm"), np.ones(EMB_DIM) / np.sqrt(EMB_DIM),
        np.array([0.0]), FakeIndexStats(),
    )
    path = tmp_path / "sub" / "s.json"
    fs.write_feature_samples(str(path), "train", "toy", 7, [entry], {"blocks": ["lexical"]})
    data = json.load(open(path))
    assert data["side"] == "train" and data["dataset"] == "toy" and data["seed"] == 7
    assert data["n"] == 1 and data["samples"][0] == entry and data["blocks"] == ["lexical"]
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["lexical_feature_names"] == fs.LEXICAL_FEATURE_NAMES


def test_build_train_samples_uses_computed_caches(tmp_path):
    from dataset import load_run

    runs = load_run([write_run(tmp_path)])
    clf = make_query_type_classifier(pipeline_fn=fake_pipeline)
    _, caches = time_dataset(
        "toy", QUERIES, runs, FakeIndexStats(), clf, FakeEncoder(), torch.device("cpu"),
    )
    samples = tfc.build_train_samples(QUERIES, caches, FakeIndexStats(), 2, seed=0)
    assert len(samples) == 2
    for s in samples:
        assert s["query"] == QUERIES[s["qid"]]
        assert s["lexical"]["recomputed_matches"] is True
        assert s["embedding"]["l2_norm"] == pytest.approx(1.0, abs=1e-5)
        assert s["query_type"]["label"] in ("keyword", "natural_language")
    assert tfc.build_train_samples(QUERIES, {}, None, 2, seed=0) == []


def test_main_writes_train_feature_samples(tmp_path, monkeypatch):
    q, run = _write_toy_dataset(tmp_path)
    _patch_models(monkeypatch)
    base = ["prog", "--dataset", "toy", q, run, "--index", "fake", "--device", "cpu"]

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", base + ["--output_dir", str(out)])
    tfc.main()
    data = json.load(open(out / "feature_samples_train_toy.json"))
    assert data["n"] == 10 and len(data["samples"]) == 10  # default 10 of the 12 timed queries
    assert data["blocks"] == ["embedding", "lexical", "query_type"]

    out2 = tmp_path / "out2"
    monkeypatch.setattr(sys, "argv", base + [
        "--num_queries", "toy=6", "--num_feature_samples", "3", "--seed", "5",
        "--output_dir", str(out2)])
    tfc.main()
    sampled = set((out2 / "sampled_qids_toy.txt").read_text().split())
    got = json.load(open(out2 / "feature_samples_train_toy.json"))
    assert len(got["samples"]) == 3 and {s["qid"] for s in got["samples"]} <= sampled

    out3 = tmp_path / "out3"
    monkeypatch.setattr(sys, "argv", base + ["--num_feature_samples", "0", "--output_dir", str(out3)])
    tfc.main()
    assert not (out3 / "feature_samples_train_toy.json").exists()


def test_time_queries_keep_qids_and_build_eval_samples():
    blocks = ("lexical", "embedding", "query_type")
    clf = make_query_type_classifier(pipeline_fn=fake_pipeline)
    model = make_model(blocks)
    keep = ["2", "3"]
    rows, live = time_queries(
        "toy", QUERIES, blocks, FakeIndexStats(), FakeEncoder(), clf, model,
        torch.device("cpu"), num_rankers=3, top_k=10, warmup=0, keep_qids=keep,
    )
    assert len(rows) == 3 and set(live) == set(keep)  # only the requested qids are kept
    samples = tel.build_eval_samples(QUERIES, live, keep, FakeIndexStats())
    assert [s["qid"] for s in samples] == keep
    for s in samples:
        assert s["lexical"]["recomputed_matches"] is True
        assert s["embedding"]["dim"] == EMB_DIM
        assert s["embedding"]["l2_norm"] == pytest.approx(1.0, abs=1e-5)


def test_time_queries_first_n_and_keep_qids_combine():
    blocks = ("lexical",)
    _, live = time_queries(
        "toy", QUERIES, blocks, FakeIndexStats(), None, None, make_model(blocks),
        torch.device("cpu"), num_rankers=3, top_k=10, warmup=0,
        keep_vectors_for=1, keep_qids=["3"],
    )
    assert set(live) == {"1", "3"}
