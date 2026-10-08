"""
Unit tests for features.py's configurable-feature-block layout resolvers:
resolve_query_feature_layout and validate_feature_blocks, which let a caller
select any subset of {lexical, embedding, query_type}.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (
    ALL_FEATURE_BLOCKS,
    LEXICAL_FEATURE_NAMES,
    QUERY_TYPE_FEATURE_NAMES,
    embedding_feature_names,
    resolve_query_feature_layout,
    validate_feature_blocks,
)


def test_all_feature_blocks_canonical_order():
    assert ALL_FEATURE_BLOCKS == ("lexical", "embedding", "query_type")


@pytest.mark.parametrize("dim", [32, 768, 384])
def test_embedding_feature_names_is_width_agnostic(dim):
    """embedding_feature_names(dim) replaces the old fixed-width
    EMBEDDING_FEATURE_NAMES constant - it must produce exactly `dim` names,
    for any raw embedding width (768 BERT/Contriever, 384 MiniLM, etc.), not
    just the old fixed 32."""
    names = embedding_feature_names(dim)
    assert names == [f"query_emb_{i}" for i in range(dim)]
    assert len(names) == dim


class TestValidateFeatureBlocks:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            validate_feature_blocks(())

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown feature block"):
            validate_feature_blocks(("lexical", "bogus"))

    def test_all_valid_does_not_raise(self):
        validate_feature_blocks(ALL_FEATURE_BLOCKS)
        validate_feature_blocks(("lexical",))


class TestResolveQueryFeatureLayout:
    @pytest.mark.parametrize("raw_dim", [32, 768, 384])
    def test_all_three_matches_legacy_fixed_layout(self, raw_dim):
        dim, names, emb_slice = resolve_query_feature_layout(
            ("lexical", "embedding", "query_type"), embedding_dim=raw_dim
        )
        assert dim == 5 + raw_dim + 1
        assert names == LEXICAL_FEATURE_NAMES + embedding_feature_names(raw_dim) + QUERY_TYPE_FEATURE_NAMES
        assert emb_slice == slice(5, 5 + raw_dim)

    def test_lexical_only(self):
        dim, names, emb_slice = resolve_query_feature_layout(("lexical",))
        assert dim == 5
        assert names == LEXICAL_FEATURE_NAMES
        assert emb_slice is None

    @pytest.mark.parametrize("raw_dim", [32, 768, 384])
    def test_embedding_only_offset_is_zero_not_legacy_five(self, raw_dim):
        """The key regression case: without lexical preceding it, embedding's
        offset must be 0, NOT the legacy fixed EMBEDDING_START=5 - and its
        width must be exactly the supplied raw_dim, not a fixed constant."""
        dim, names, emb_slice = resolve_query_feature_layout(("embedding",), embedding_dim=raw_dim)
        assert dim == raw_dim
        assert names == embedding_feature_names(raw_dim)
        assert emb_slice == slice(0, raw_dim)

    def test_embedding_selected_without_embedding_dim_raises(self):
        """embedding_dim must be inferred from the loaded embedding source
        and passed in - it can never silently default to some fixed width."""
        with pytest.raises(ValueError, match="embedding_dim"):
            resolve_query_feature_layout(("embedding",))

    def test_lexical_and_query_type_no_embedding(self):
        dim, names, emb_slice = resolve_query_feature_layout(("lexical", "query_type"))
        assert dim == 5 + 1
        assert names == LEXICAL_FEATURE_NAMES + QUERY_TYPE_FEATURE_NAMES
        assert emb_slice is None

    @pytest.mark.parametrize("raw_dim", [32, 768, 384])
    def test_query_type_and_embedding_no_lexical_offset_is_zero(self, raw_dim):
        dim, names, emb_slice = resolve_query_feature_layout(
            ("query_type", "embedding"), embedding_dim=raw_dim
        )
        # Canonical concat order is embedding-before-query_type regardless of
        # the input order given (embedding precedes query_type in ALL_FEATURE_BLOCKS).
        assert dim == raw_dim + 1
        assert names == embedding_feature_names(raw_dim) + QUERY_TYPE_FEATURE_NAMES
        assert emb_slice == slice(0, raw_dim)
