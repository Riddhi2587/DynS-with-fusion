import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Dimensionality of the 5-dim lexical/IDF query feature block
# (build_query_features), and of the 1-dim query_type flag
# (build_query_type_feature). Named so resolve_query_feature_layout below
# doesn't rely on magic numbers.
LEXICAL_FEATURE_DIM = 5
QUERY_TYPE_FEATURE_DIM = 1

# Query-embedding (pre-retrieval) feature: dimensionality of the LEARNED
# projection's output (model.py's embedding_proj, an nn.Linear trained
# end-to-end), NOT the raw embedding's width. The raw, un-reduced query
# embedding (see feature_cache.load_query_embeddings / a source like
# data/cache/bert-query-embeddings/cls/*.cls.pkl) is variable-width -
# 768 for BERT/Contriever, 384 for all-MiniLM-L6-v2, etc. - and is inferred
# at load/dataset-construction time, never hardcoded (see
# resolve_query_feature_layout's embedding_dim param).
EMBEDDING_DIM = 32

# The three query-feature sub-blocks, in the fixed concatenation order used
# by build_full_query_features / resolve_query_feature_layout (which lets a
# caller select any subset of them while preserving this same relative
# order). The embedding block's raw width varies by source (768/384/...),
# so its names can't be a fixed module-level list - see
# embedding_feature_names below.
LEXICAL_FEATURE_NAMES = ["num_query_terms", "num_unique_query_terms", "min_idf", "max_idf", "sum_idf"]
QUERY_TYPE_FEATURE_NAMES = ["query_type"]


def embedding_feature_names(dim: int) -> List[str]:
    """Column names for a `dim`-wide raw query-embedding block. `dim` is
    whatever width the loaded embedding source turned out to be (inferred at
    load/dataset-construction time, e.g. via feature_cache.load_query_embeddings
    or QPPQueryOnlyDataset) - there is no fixed embedding width to
    default to."""
    return [f"query_emb_{i}" for i in range(dim)]


# Canonical order for the 3 selectable input feature blocks (see
# resolve_query_feature_layout / validate_feature_blocks).
ALL_FEATURE_BLOCKS = ("lexical", "embedding", "query_type")


def validate_feature_blocks(feature_blocks) -> None:
    """Raises ValueError if feature_blocks is empty, contains anything
    outside ALL_FEATURE_BLOCKS."""
    unknown = set(feature_blocks) - set(ALL_FEATURE_BLOCKS)
    if unknown:
        raise ValueError(
            f"Unknown feature block(s): {sorted(unknown)}. Must be a subset "
            f"of {ALL_FEATURE_BLOCKS}."
        )
    if not feature_blocks:
        raise ValueError(
            f"At least one feature block must be selected (from {ALL_FEATURE_BLOCKS})."
        )


def resolve_query_feature_layout(
    feature_blocks, embedding_dim: Optional[int] = None
) -> Tuple[int, List[str], Optional[slice]]:
    """
    Given a set/sequence of selected feature block names (subset of
    ALL_FEATURE_BLOCKS), returns (query_feature_dim, query_feature_names,
    embedding_slice) for the query-side vector formed by concatenating only
    the selected blocks from {lexical, embedding, query_type},
    in that fixed order.

    embedding_dim: required (and must be positive) when "embedding" is
    selected - the raw embedding block's width, inferred by the caller from
    whatever embedding source is loaded (e.g. feature_cache.load_query_embeddings
    / QPPQueryOnlyDataset's embedding_cache.meta["embedding_dim"]), since it varies by
    source (768 for BERT/Contriever, 384 for MiniLM, etc.) and can't be a
    fixed constant. Ignored if "embedding" isn't selected.

    embedding_slice locates the embedding block within the resulting vector
    - None if "embedding" isn't selected. Its offset depends on whether
    "lexical" is also selected, so it's always computed dynamically here
    (never assumed to be some fixed offset).
    """
    dim = 0
    names: List[str] = []
    embedding_slice = None
    if "lexical" in feature_blocks:
        dim += LEXICAL_FEATURE_DIM
        names += LEXICAL_FEATURE_NAMES
    if "embedding" in feature_blocks:
        if not embedding_dim:
            raise ValueError(
                "'embedding' is selected but embedding_dim was not supplied - "
                "the raw embedding width must be inferred from the loaded "
                "embedding source and passed in explicitly."
            )
        embedding_slice = slice(dim, dim + embedding_dim)
        dim += embedding_dim
        names += embedding_feature_names(embedding_dim)
    if "query_type" in feature_blocks:
        dim += QUERY_TYPE_FEATURE_DIM
        names += QUERY_TYPE_FEATURE_NAMES
    return dim, names, embedding_slice


class IndexStats:
    """IDF statistics from a Pyserini Lucene index."""

    def __init__(self, index_path: str):
        from pyserini.index.lucene import LuceneIndexReader
        self.reader = LuceneIndexReader(index_path)
        stats = self.reader.stats()
        self._num_docs: int = stats["documents"]
        self._idf_cache: Dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term not in self._idf_cache:
            try:
                df, _ = self.reader.get_term_counts(term, analyzer=None)
            except Exception:
                df = 0
            self._idf_cache[term] = math.log(
                (self._num_docs - df + 0.5) / (df + 0.5) + 1
            )
        return self._idf_cache[term]


def build_query_features(query_terms: List[str], index_stats: IndexStats) -> np.ndarray:
    """5-dim query-level feature vector u_q."""
    idfs = [index_stats.idf(t) for t in query_terms]
    return np.array(
        [
            float(len(query_terms)),
            float(len(set(query_terms))),
            min(idfs) if idfs else 0.0,
            max(idfs) if idfs else 0.0,
            sum(idfs),
        ],
        dtype=np.float32,
    )


class QueryTypeClassifier:
    """
    Pre-retrieval query_type feature: classifies a raw query as keyword-based
    (0.0) or natural-language (1.0), using the same model as the LTRR paper
    (shahrukhx01/bert-mini-finetune-question-detection), mirrored from that
    repo's featurization/pre_retrieval.py.

    Loads the underlying transformers pipeline lazily, once, on first use
    (not at construction time), so constructing one is cheap when it ends up
    unused (e.g. a fully-covered --feature_cache run). `pipeline_fn` is an
    injection point for tests: pass a fake callable(list[str]) ->
    list[{"label": ...}] to avoid any network access / model download.
    """

    MODEL_NAME = "shahrukhx01/bert-mini-finetune-question-detection"

    def __init__(self, pipeline_fn: Optional[Callable[[List[str]], List[dict]]] = None):
        self._pipeline_fn = pipeline_fn

    def _get_pipeline(self):
        if self._pipeline_fn is None:
            from transformers import pipeline
            self._pipeline_fn = pipeline("text-classification", model=self.MODEL_NAME)
        return self._pipeline_fn

    def classify(self, query: str) -> float:
        return self.batch_classify([query])[0]

    def batch_classify(self, queries: List[str]) -> List[float]:
        """Returns 0.0 (keyword-based) / 1.0 (natural language) per query,
        parsed from the pipeline's "LABEL_0"/"LABEL_1" output exactly like
        the LTRR repo: float(label.split("_")[-1])."""
        pipeline_fn = self._get_pipeline()
        results = pipeline_fn(queries)
        return [float(r["label"].split("_")[-1]) for r in results]


def build_embedding_feature(
    qid: str, embedding_lookup: Dict[str, np.ndarray]
) -> np.ndarray:
    """Raw, variable-width query representation feature: the precomputed
    embedding for qid (see e.g. data/cache/bert-query-embeddings/cls/*.cls.pkl,
    feature_cache.load_query_embeddings), L2-normalized here at
    feature-construction time. Its width depends on the embedding source
    (768 for BERT/Contriever, 384 for MiniLM, etc.) and is not validated
    against a fixed dimension here - see feature_cache.load_query_embeddings
    for the infer-and-validate-consistency check across qids.

    This is the only normalization ever applied to it here: whether it's
    z-scored on top of that is controlled by model.py's
    standardize_embedding flag, after being reduced to EMBEDDING_DIM by a
    trained nn.Linear (model.py's embedding_proj).

    Raises KeyError if qid has no precomputed embedding in embedding_lookup -
    this is deliberate (fail loudly rather than silently zero-filling)."""
    if qid not in embedding_lookup:
        raise KeyError(
            f"No precomputed query embedding for qid={qid!r} in the provided "
            f"--query_embeddings file."
        )
    embedding = np.asarray(embedding_lookup[qid], dtype=np.float32)
    return embedding / (np.linalg.norm(embedding) + 1e-12)


def build_query_type_feature(
    raw_query: str, query_type_classifier: QueryTypeClassifier
) -> np.ndarray:
    """1-dim query_type feature (0.0 keyword-based / 1.0 natural language).
    `raw_query` (unlowercased, unsplit) is used since question-detection is
    sensitive to case/punctuation, unlike the lexical/IDF features (which use
    tokenized `query_terms`)."""
    return np.array([query_type_classifier.classify(raw_query)], dtype=np.float32)


def build_full_query_features(
    raw_query: str,
    query_terms: List[str],
    qid: str,
    index_stats: IndexStats,
    embedding_lookup: Dict[str, np.ndarray],
    query_type_classifier: QueryTypeClassifier,
) -> np.ndarray:
    """(5 + raw_embedding_dim + 1)-dim query-level feature vector:
    build_query_features (5-dim lexical) + build_embedding_feature
    (raw_embedding_dim-dim, variable by source) + build_query_type_feature
    (1-dim), concatenated in that order.

    A convenience wrapper for computing every pre-retrieval feature fresh in
    one call (e.g. tests) - the actual train/eval path (dataset.py) calls the
    three atomic builders directly instead, since each one may come from its
    own independent cache (see feature_cache.build_feature_cache /
    build_embedding_cache / build_query_type_cache)."""
    lexical = build_query_features(query_terms, index_stats)
    embedding = build_embedding_feature(qid, embedding_lookup)
    query_type = build_query_type_feature(raw_query, query_type_classifier)
    return np.concatenate([lexical, embedding, query_type])
