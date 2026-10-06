import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import EMBEDDING_DIM
from model import QueryOnlyMLP

# Arbitrary fixed query dim for shape tests that aren't about the embedding
# block at all - there is no longer a fixed QUERY_FEATURE_DIM constant to
# import, since the embedding component of the legacy layout is now a
# variable raw width, not fixed.
TEST_QUERY_DIM = 37


@pytest.mark.parametrize("num_rankers", [1, 5, 8, 10])
def test_query_only_mlp_input_dim_matches_query_feature_dim(num_rankers):
    """
    QueryOnlyMLP's input is just the shared query feature vector - no doc
    features, no top_k, no per-ranker input - so its first Linear layer's
    input dim must be exactly query_feature_dim regardless of num_rankers,
    when no embedding_slice is given (embedding_proj tests below cover the
    case where one is).
    """
    model = QueryOnlyMLP(
        query_feature_dim=TEST_QUERY_DIM,
        hidden_dims=[64, 32],
        num_rankers=num_rankers,
    )

    first_linear = model.mlp[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == TEST_QUERY_DIM

    last_linear = model.mlp[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    assert last_linear.out_features == num_rankers

    # Confirm a real forward pass actually consumes an input of this size
    # and produces one logit per ranker, with no mask/pad_mask arguments.
    B = 3
    query_feats = torch.randn(B, TEST_QUERY_DIM)
    logits = model(query_feats)
    assert logits.shape == (B, num_rankers)


def test_query_only_mlp_has_no_doc_buffers_or_modules():
    """QueryOnlyMLP must carry no doc-feature state at all - a regression
    guard so the doc-side of QPPMLP never leaks back in."""
    model = QueryOnlyMLP(query_feature_dim=TEST_QUERY_DIM, num_rankers=4)
    buffer_names = dict(model.named_buffers()).keys()
    assert "doc_mean" not in buffer_names
    assert "doc_std" not in buffer_names
    assert "ranker_score_mean" not in buffer_names
    assert "ranker_score_std" not in buffer_names


def test_query_only_mlp_output_depends_on_query_feats():
    """A positive test that the model actually uses its input: two very
    different query feature vectors through the same model instance must
    give different logits (guards against a silently-constant/no-op head)."""
    torch.manual_seed(0)
    model = QueryOnlyMLP(query_feature_dim=TEST_QUERY_DIM, num_rankers=3)
    model.eval()

    q1 = torch.zeros(1, TEST_QUERY_DIM)
    q2 = torch.ones(1, TEST_QUERY_DIM) * 50.0

    with torch.no_grad():
        out1 = model(q1)
        out2 = model(q2)

    assert not torch.allclose(out1, out2)


def test_embedding_slice_omitted_means_no_embedding_proj():
    """QueryOnlyMLP no longer auto-detects the embedding block from
    query_feature_dim alone (that heuristic couldn't disambiguate a
    variable-width raw embedding) - embedding_slice must be passed
    explicitly, and omitting it means no embedding_proj at all, matching
    QPPMLP's convention."""
    model = QueryOnlyMLP(query_feature_dim=TEST_QUERY_DIM, num_rankers=2)
    assert model.embedding_slice is None
    assert model.embedding_proj is None
    # And a query_feature_dim that WOULD have tripped the old
    # `>= EMBEDDING_START + EMBEDDING_DIM` auto-detect heuristic still gets
    # no embedding_proj without an explicit embedding_slice.
    model_wide = QueryOnlyMLP(query_feature_dim=5 + EMBEDDING_DIM + 1, num_rankers=2)
    assert model_wide.embedding_slice is None
    assert model_wide.embedding_proj is None


@pytest.mark.parametrize("raw_dim", [768, 384, 32])
def test_embedding_proj_shape_matches_raw_dim_not_hardcoded(raw_dim):
    """embedding_proj must reduce whatever raw_dim it's given down to
    EMBEDDING_DIM - tested across several raw_dims, including one (32) that
    happens to equal EMBEDDING_DIM, to make sure that's not silently relied
    upon."""
    emb_slice = slice(5, 5 + raw_dim)
    model = QueryOnlyMLP(
        query_feature_dim=5 + raw_dim + 1,
        embedding_slice=emb_slice,
        num_rankers=2,
    )
    assert model.embedding_proj is not None
    assert model.embedding_proj.weight.shape == (EMBEDDING_DIM, raw_dim)
    assert model.embedding_proj.bias.shape == (EMBEDDING_DIM,)


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_reduce_embedding_false_skips_projection(raw_dim):
    """reduce_embedding=False must skip building embedding_proj entirely and
    feed the raw embedding straight into the MLP at its full width, instead
    of reducing it to EMBEDDING_DIM."""
    emb_slice = slice(5, 5 + raw_dim)
    model = QueryOnlyMLP(
        query_feature_dim=5 + raw_dim + 1,
        embedding_slice=emb_slice,
        num_rankers=2,
        reduce_embedding=False,
    )
    assert model.embedding_proj is None
    assert model.reduce_embedding is False
    first_linear = model.mlp[0]
    assert first_linear.in_features == 5 + raw_dim + 1


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_forward_output_shape_with_reduce_embedding_false(raw_dim):
    """Forward pass must still run and produce (B, num_rankers) logits when
    reduce_embedding=False, and the raw embedding values (post-standardization,
    which defaults to identity for the embedding slice) must reach the MLP
    input unchanged, since there is no learned projection in between."""
    query_feature_dim = 5 + raw_dim + 1
    emb_slice = slice(5, 5 + raw_dim)
    model = QueryOnlyMLP(
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        num_rankers=4,
        reduce_embedding=False,
    )
    B = 3
    query_feats = torch.randn(B, query_feature_dim)
    logits = model(query_feats)
    assert logits.shape == (B, 4)

    standardized = (query_feats - model.query_mean) / model.query_std
    projected = model._project_embedding(standardized)
    assert torch.allclose(projected[:, emb_slice], query_feats[:, emb_slice])


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_forward_output_shape_unaffected_by_raw_embedding_width(raw_dim):
    """Output shape must depend only on num_rankers, never on the raw
    embedding width - embedding_proj absorbs that variability."""
    query_feature_dim = 5 + raw_dim + 1
    emb_slice = slice(5, 5 + raw_dim)
    model = QueryOnlyMLP(
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        num_rankers=4,
    )
    B = 3
    query_feats = torch.randn(B, query_feature_dim)
    logits = model(query_feats)
    assert logits.shape == (B, 4)


def test_embedding_proj_receives_gradient():
    """A backward pass must produce a nonzero gradient on embedding_proj's
    weight, proving it's actually trained end-to-end."""
    raw_dim = 768
    query_feature_dim = 5 + raw_dim + 1
    emb_slice = slice(5, 5 + raw_dim)
    model = QueryOnlyMLP(
        query_feature_dim=query_feature_dim,
        embedding_slice=emb_slice,
        num_rankers=3,
    )
    B = 4
    query_feats = torch.randn(B, query_feature_dim)
    logits = model(query_feats)
    logits.sum().backward()

    assert model.embedding_proj.weight.grad is not None
    assert model.embedding_proj.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize("raw_dim", [768, 384])
def test_fit_standardization_excludes_embedding_columns_by_default(raw_dim):
    """The raw embedding block is L2-normalized per query at
    feature-construction time. By default (standardize_embedding=False)
    fit_standardization should leave its mean/std buffers at identity
    (0, 1) regardless of raw_dim, while still correctly standardizing the
    other columns."""
    torch.manual_seed(0)
    lexical_dim = 5
    query_feature_dim = lexical_dim + raw_dim + 1
    emb_slice = slice(lexical_dim, lexical_dim + raw_dim)
    model = QueryOnlyMLP(
        query_feature_dim=query_feature_dim, embedding_slice=emb_slice,
        hidden_dims=[16], num_rankers=2,
    )

    n = 200
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

    labels = torch.zeros(n, 2)
    loader = DataLoader(TensorDataset(query_feats, labels), batch_size=32)
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
    model = QueryOnlyMLP(
        query_feature_dim=query_feature_dim, embedding_slice=emb_slice,
        hidden_dims=[16], num_rankers=2, standardize_embedding=True,
    )

    n = 200
    query_feats = torch.zeros(n, query_feature_dim)
    query_feats[:, emb_slice] = torch.randn(n, raw_dim) * 3.0 + 7.0
    labels = torch.zeros(n, 2)
    loader = DataLoader(TensorDataset(query_feats, labels), batch_size=32)
    model.fit_standardization(loader)

    assert not torch.allclose(model.query_mean[emb_slice], torch.zeros(raw_dim))
    assert model.query_mean[emb_slice].mean().item() > 5.0
