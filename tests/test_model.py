import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (
    DOC_FEATURE_DIM,
    EMBEDDING_DIM,
    LIST_FEATURE_DIM,
)
from model import QPPMLP

# Arbitrary fixed query dim for shape tests that aren't about the embedding
# block at all - the old QUERY_FEATURE_DIM constant (5 + EMBEDDING_DIM + 1)
# no longer exists as a plain int, since the embedding component of that
# legacy layout is now a variable raw width, not fixed. These tests just
# need SOME query_feature_dim to exercise the general MLP-head shape
# formula, so a local constant stands in for it.
TEST_QUERY_DIM = 37


@pytest.mark.parametrize("top_k,num_rankers", [(100, 1), (100, 5), (100, 10), (50, 3)])
def test_mlp_input_dim_matches_formula(top_k, num_rankers):
    """
    For a query and R rankers, the MLP's input (all rankers' flattened doc
    features, each preceded by that ranker's once-only list features,
    concatenated with the single shared query feature vector) must have
    dimension TEST_QUERY_DIM + R*(LIST_FEATURE_DIM + top_k*DOC_FEATURE_DIM).
    No embedding_slice here - this test is about the general shape formula,
    not the embedding block specifically (see test_embedding_proj_* below
    for that).
    """
    assert DOC_FEATURE_DIM == 9
    assert LIST_FEATURE_DIM == 2

    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=TEST_QUERY_DIM,
        top_k=top_k,
        hidden_dims=[64, 32],
        num_rankers=num_rankers,
    )

    expected_dim = TEST_QUERY_DIM + num_rankers * (LIST_FEATURE_DIM + top_k * DOC_FEATURE_DIM)

    first_linear = model.mlp[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == expected_dim

    # Confirm a real forward pass actually consumes an input of this size.
    B = 2
    doc_feats = torch.randn(B, num_rankers, top_k, DOC_FEATURE_DIM)
    list_feats = torch.randn(B, num_rankers, LIST_FEATURE_DIM)
    query_feats = torch.randn(B, TEST_QUERY_DIM)
    pad_mask = torch.zeros(B, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(B, num_rankers, dtype=torch.bool)

    logits = model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)
    assert logits.shape == (B, num_rankers)


def test_list_feats_placed_once_per_ranker_not_duplicated():
    """Regression guard for the mean_score/var_score de-duplication: list
    features must appear ONCE per ranker (not once per doc), and forward()
    must actually be sensitive to them (not silently dropped)."""
    torch.manual_seed(0)
    top_k, num_rankers = 5, 3
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=TEST_QUERY_DIM,
        top_k=top_k,
        hidden_dims=[16],
        num_rankers=num_rankers,
    )
    model.eval()

    # The input dim must scale as (LIST_FEATURE_DIM + top_k*DOC_FEATURE_DIM)
    # per ranker, i.e. list_feats contributes LIST_FEATURE_DIM per ranker
    # regardless of top_k - not LIST_FEATURE_DIM*top_k (which is what "one
    # copy per doc" would look like).
    expected_dim = TEST_QUERY_DIM + num_rankers * (LIST_FEATURE_DIM + top_k * DOC_FEATURE_DIM)
    assert model.mlp[0].in_features == expected_dim

    B = 1
    doc_feats = torch.zeros(B, num_rankers, top_k, DOC_FEATURE_DIM)
    query_feats = torch.zeros(B, TEST_QUERY_DIM)
    pad_mask = torch.zeros(B, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(B, num_rankers, dtype=torch.bool)

    list_feats_a = torch.zeros(B, num_rankers, LIST_FEATURE_DIM)
    list_feats_b = torch.ones(B, num_rankers, LIST_FEATURE_DIM) * 50.0

    with torch.no_grad():
        out_a = model(doc_feats, list_feats_a, query_feats, pad_mask, ranker_mask)
        out_b = model(doc_feats, list_feats_b, query_feats, pad_mask, ranker_mask)

    assert not torch.allclose(out_a, out_b)


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_fit_standardization_excludes_embedding_columns_by_default(raw_dim):
    """The raw embedding block (whatever its width - 768 BERT/Contriever, 384
    MiniLM, etc.) is L2-normalized per query at feature-construction time.
    By default (standardize_embedding=False) fit_standardization must leave
    its mean/std buffers at identity (0, 1) regardless of raw_dim, while
    still correctly standardizing the other columns."""
    torch.manual_seed(0)
    top_k, num_rankers = 4, 2
    lexical_dim = 5
    query_feature_dim = lexical_dim + raw_dim + 1
    emb_slice = slice(lexical_dim, lexical_dim + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        top_k=top_k,
        hidden_dims=[16],
        num_rankers=num_rankers,
    )

    n = 64
    doc_feats = torch.randn(n, num_rankers, top_k, DOC_FEATURE_DIM)
    list_feats = torch.randn(n, num_rankers, LIST_FEATURE_DIM)
    pad_mask = torch.zeros(n, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(n, num_rankers, dtype=torch.bool)
    labels = torch.zeros(n, num_rankers)

    query_feats = torch.zeros(n, query_feature_dim)
    # Non-embedding columns (lexical [:5] + query_type [-1]): a clearly
    # non-trivial distribution, so fit_standardization must be computing
    # real statistics for them, not also forcing them to identity.
    query_feats[:, :lexical_dim] = torch.randn(n, lexical_dim) * 20.0 + 100.0
    query_feats[:, -1] = torch.randint(0, 2, (n,)).float()
    # Embedding columns: a different non-trivial distribution - if
    # fit_standardization didn't exclude them, their buffers would end up
    # matching THIS distribution instead of identity.
    query_feats[:, emb_slice] = torch.randn(n, raw_dim) * 3.0 + 7.0

    loader = DataLoader(
        TensorDataset(doc_feats, list_feats, query_feats, pad_mask, ranker_mask, labels),
        batch_size=16,
    )
    model.fit_standardization(loader)

    assert model.embedding_slice == emb_slice
    assert torch.allclose(model.query_mean[emb_slice], torch.zeros(raw_dim))
    assert torch.allclose(model.query_std[emb_slice], torch.ones(raw_dim))

    # Non-embedding columns should reflect the real (non-trivial) statistics,
    # not also be forced to identity.
    assert not torch.allclose(model.query_mean[:lexical_dim], torch.zeros(lexical_dim))
    assert model.query_mean[:lexical_dim].mean().item() > 50.0


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_fit_standardization_computes_real_stats_when_standardize_embedding_true(raw_dim):
    """standardize_embedding=True is the opt-in: the raw embedding slice
    should get real computed statistics like every other query feature,
    instead of being forced to identity."""
    torch.manual_seed(0)
    lexical_dim = 5
    query_feature_dim = lexical_dim + raw_dim + 1
    emb_slice = slice(lexical_dim, lexical_dim + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        top_k=3,
        hidden_dims=[16],
        num_rankers=2,
        standardize_embedding=True,
    )

    n = 64
    doc_feats = torch.randn(n, 2, 3, DOC_FEATURE_DIM)
    list_feats = torch.randn(n, 2, LIST_FEATURE_DIM)
    pad_mask = torch.zeros(n, 2, 3, dtype=torch.bool)
    ranker_mask = torch.ones(n, 2, dtype=torch.bool)
    labels = torch.zeros(n, 2)

    query_feats = torch.zeros(n, query_feature_dim)
    query_feats[:, emb_slice] = torch.randn(n, raw_dim) * 3.0 + 7.0

    loader = DataLoader(
        TensorDataset(doc_feats, list_feats, query_feats, pad_mask, ranker_mask, labels),
        batch_size=16,
    )
    model.fit_standardization(loader)

    assert not torch.allclose(model.query_mean[emb_slice], torch.zeros(raw_dim))
    assert model.query_mean[emb_slice].mean().item() > 5.0


def test_doc_feats_disabled_input_dim_is_query_only():
    """A model built for a doc_feats-excluded --features config (see
    dataset.py's QPPDataset.feature_blocks / features.resolve_doc_feature_layout,
    which resolve doc_feature_dim=list_feature_dim=0 in that case) must have
    an input dim of exactly query_feature_dim - no ranker-count-dependent
    term at all - and forward() must still run with 0-width doc/list tensors."""
    top_k, num_rankers = 7, 4
    model = QPPMLP(
        doc_feature_dim=0,
        list_feature_dim=0,
        query_feature_dim=TEST_QUERY_DIM,
        top_k=top_k,
        hidden_dims=[16],
        num_rankers=num_rankers,
    )
    assert model.mlp[0].in_features == TEST_QUERY_DIM

    B = 3
    doc_feats = torch.zeros(B, num_rankers, top_k, 0)
    list_feats = torch.zeros(B, num_rankers, 0)
    query_feats = torch.randn(B, TEST_QUERY_DIM)
    pad_mask = torch.ones(B, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(B, num_rankers, dtype=torch.bool)

    logits = model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)
    assert logits.shape == (B, num_rankers)

    # fit_standardization must also run cleanly at width 0, including under
    # per_ranker (which guards the doc-score index_select on doc_feature_dim
    # > 0 - this is the regression test for that guard).
    labels = torch.zeros(B, num_rankers)
    loader = DataLoader(
        TensorDataset(doc_feats, list_feats, query_feats, pad_mask, ranker_mask, labels),
        batch_size=2,
    )
    per_ranker_model = QPPMLP(
        doc_feature_dim=0, list_feature_dim=0, query_feature_dim=TEST_QUERY_DIM,
        top_k=top_k, hidden_dims=[16], num_rankers=num_rankers, score_norm="per_ranker",
    )
    per_ranker_model.fit_standardization(loader)  # must not raise (index_select guard)
    per_ranker_model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_embedding_slice_at_nonstandard_offset(raw_dim):
    """When 'lexical' is excluded from --features but 'embedding' is
    included, the embedding block sits at offset 0 within query_feats, not
    at some fixed legacy offset - fit_standardization must exclude exactly
    the slice it's given, wherever it is and whatever its (variable) raw
    width (see features.resolve_query_feature_layout)."""
    torch.manual_seed(1)
    top_k, num_rankers = 3, 2
    query_feature_dim = raw_dim + 1  # embedding + query_type, no lexical
    emb_slice = slice(0, raw_dim)  # NOT some fixed legacy offset
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        top_k=top_k,
        hidden_dims=[16],
        num_rankers=num_rankers,
    )

    n = 40
    doc_feats = torch.randn(n, num_rankers, top_k, DOC_FEATURE_DIM)
    list_feats = torch.randn(n, num_rankers, LIST_FEATURE_DIM)
    pad_mask = torch.zeros(n, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(n, num_rankers, dtype=torch.bool)
    labels = torch.zeros(n, num_rankers)

    query_feats = torch.zeros(n, query_feature_dim)
    # Embedding columns (now at the START, offset 0): non-trivial distribution.
    query_feats[:, emb_slice] = torch.randn(n, raw_dim) * 4.0 + 9.0
    # query_type column (now at the END, offset raw_dim): non-trivial too.
    query_feats[:, -1] = torch.randn(n) * 10.0 + 200.0

    loader = DataLoader(
        TensorDataset(doc_feats, list_feats, query_feats, pad_mask, ranker_mask, labels),
        batch_size=8,
    )
    model.fit_standardization(loader)

    assert model.embedding_slice == emb_slice
    assert torch.allclose(model.query_mean[emb_slice], torch.zeros(raw_dim))
    assert torch.allclose(model.query_std[emb_slice], torch.ones(raw_dim))
    # The query_type column (outside emb_slice) must reflect real statistics.
    assert model.query_mean[-1].item() > 100.0


@pytest.mark.parametrize("raw_dim", [768, 384, 32])
def test_embedding_proj_shape_matches_raw_dim_not_hardcoded(raw_dim):
    """embedding_proj must reduce whatever raw_dim it's given down to
    EMBEDDING_DIM, proving the projection's input width is never hardcoded -
    tested across several raw_dims, including one (32) that happens to equal
    EMBEDDING_DIM, to make sure that's not silently relied upon."""
    emb_slice = slice(5, 5 + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=5 + raw_dim + 1,
        embedding_slice=emb_slice,
        top_k=3,
        num_rankers=2,
    )
    assert model.embedding_proj is not None
    assert model.embedding_proj.weight.shape == (EMBEDDING_DIM, raw_dim)
    assert model.embedding_proj.bias.shape == (EMBEDDING_DIM,)


def test_embedding_proj_is_none_without_embedding_slice():
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=TEST_QUERY_DIM,
        top_k=3,
        num_rankers=2,
    )
    assert model.embedding_proj is None


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_reduce_embedding_false_skips_projection(raw_dim):
    """reduce_embedding=False must skip building embedding_proj entirely and
    feed the raw embedding straight into the MLP at its full width, instead
    of reducing it to EMBEDDING_DIM."""
    emb_slice = slice(5, 5 + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=5 + raw_dim + 1,
        embedding_slice=emb_slice,
        top_k=3,
        num_rankers=2,
        reduce_embedding=False,
    )
    assert model.embedding_proj is None
    assert model.reduce_embedding is False
    first_linear = model.mlp[0]
    expected_in_dim = 2 * (3 * DOC_FEATURE_DIM + LIST_FEATURE_DIM) + (5 + raw_dim + 1)
    assert first_linear.in_features == expected_in_dim


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_forward_output_shape_with_reduce_embedding_false(raw_dim):
    """Forward pass must still run and produce (B, num_rankers) logits when
    reduce_embedding=False, and the raw embedding values (post-standardization,
    which defaults to identity for the embedding slice) must reach the MLP
    input unchanged, since there is no learned projection in between."""
    top_k, num_rankers = 5, 3
    query_feature_dim = 5 + raw_dim + 1
    emb_slice = slice(5, 5 + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        top_k=top_k,
        hidden_dims=[16],
        num_rankers=num_rankers,
        reduce_embedding=False,
    )
    B = 2
    doc_feats = torch.randn(B, num_rankers, top_k, DOC_FEATURE_DIM)
    list_feats = torch.randn(B, num_rankers, LIST_FEATURE_DIM)
    query_feats = torch.randn(B, query_feature_dim)
    pad_mask = torch.zeros(B, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(B, num_rankers, dtype=torch.bool)

    logits = model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)
    assert logits.shape == (B, num_rankers)

    projected = model._project_embedding(model._standardize(doc_feats, list_feats, query_feats)[2])
    assert torch.allclose(projected[:, emb_slice], query_feats[:, emb_slice])


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_forward_output_shape_unaffected_by_raw_embedding_width(raw_dim):
    """The MLP head's output shape must depend only on num_rankers, never on
    the raw embedding width - embedding_proj absorbs that variability before
    the shared MLP sees it."""
    top_k, num_rankers = 5, 3
    query_feature_dim = 5 + raw_dim + 1
    emb_slice = slice(5, 5 + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        top_k=top_k,
        hidden_dims=[16],
        num_rankers=num_rankers,
    )
    B = 2
    doc_feats = torch.randn(B, num_rankers, top_k, DOC_FEATURE_DIM)
    list_feats = torch.randn(B, num_rankers, LIST_FEATURE_DIM)
    query_feats = torch.randn(B, query_feature_dim)
    pad_mask = torch.zeros(B, num_rankers, top_k, dtype=torch.bool)
    ranker_mask = torch.ones(B, num_rankers, dtype=torch.bool)

    logits = model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)
    assert logits.shape == (B, num_rankers)


def test_embedding_proj_receives_gradient():
    """A backward pass must produce a nonzero gradient on embedding_proj's
    weight, proving it's actually trained end-to-end (not frozen/detached)."""
    raw_dim = 768
    emb_slice = slice(5, 5 + raw_dim)
    model = QPPMLP(
        doc_feature_dim=DOC_FEATURE_DIM,
        list_feature_dim=LIST_FEATURE_DIM,
        query_feature_dim=5 + raw_dim + 1,
        embedding_slice=emb_slice,
        top_k=3,
        num_rankers=2,
    )
    B = 4
    doc_feats = torch.randn(B, 2, 3, DOC_FEATURE_DIM)
    list_feats = torch.randn(B, 2, LIST_FEATURE_DIM)
    query_feats = torch.randn(B, 5 + raw_dim + 1)
    pad_mask = torch.zeros(B, 2, 3, dtype=torch.bool)
    ranker_mask = torch.ones(B, 2, dtype=torch.bool)

    logits = model(doc_feats, list_feats, query_feats, pad_mask, ranker_mask)
    logits.sum().backward()

    assert model.embedding_proj.weight.grad is not None
    assert model.embedding_proj.weight.grad.abs().sum().item() > 0
