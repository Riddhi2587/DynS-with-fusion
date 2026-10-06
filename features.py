import json
import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

DOC_FEATURE_DIM = 9

# Dimensionality of the 5-dim lexical/IDF query feature block
# (build_query_features), and of the 1-dim query_type flag
# (build_query_type_feature). Named so resolve_query_feature_layout below
# doesn't rely on magic numbers.
LEXICAL_FEATURE_DIM = 5
QUERY_TYPE_FEATURE_DIM = 1

# Dimensionality of the 1-dim entity_count feature (build_entity_count_feature)
# - a precomputed named-entity count for the query, looked up by qid from an
# externally-built {qid: count} lookup (see feature_cache.load_entity_count_lookup).
# Query-level and cache-only in the same sense as "embedding": there is no
# live NER computation here, only lookup - see build_entity_count_feature.
ENTITY_COUNT_FEATURE_DIM = 1

# Dimensionality of the 3-dim scs_pmi feature (build_scs_pmi_feature) - a
# precomputed (scs, avg_pmi, max_pmi) triple for the query, looked up by qid
# from an externally-built {qid: array} lookup (see
# feature_cache.load_scs_pmi_lookup). Query-level and cache-only, same
# convention as "entity_count": no live computation here, only lookup.
SCS_PMI_FEATURE_DIM = 3

# Dimensionality of the once-per-(query, ranker) "list features"
# (mean_score, var_score - see LIST_FEATURE_NAMES). These used to be baked
# into every document's feature vector (constant across all top_k docs in a
# list), wasting input capacity; they're now factored out into their own
# slot, built once per ranker's ranked list (see dataset.py's QPPDataset)
# instead of being duplicated top_k times.
LIST_FEATURE_DIM = 2

# Query-embedding (pre-retrieval) feature: dimensionality of the LEARNED
# projection's output (model.py's embedding_proj, an nn.Linear trained
# end-to-end), NOT the raw embedding's width. The raw, un-reduced query
# embedding (see feature_cache.load_query_embeddings / a source like
# data/cache/bert-query-embeddings/cls/*.cls.pkl) is variable-width -
# 768 for BERT/Contriever, 384 for all-MiniLM-L6-v2, etc. - and is inferred
# at load/dataset-construction time, never hardcoded (see
# resolve_query_feature_layout's embedding_dim param).
EMBEDDING_DIM = 32

# Position of the score-derived doc feature within the 9-dim vector: score.
# The ONLY per-doc feature whose distribution depends on which ranker
# produced the run, so it's the one handled specially by per_query /
# per_ranker score normalization. (mean_score/var_score used to be here too,
# but they're constant across a whole ranked list, not per-doc - see
# LIST_FEATURE_NAMES.)
SCORE_FEATURE_INDICES = (6,)

# Column names for the vectors returned by build_doc_features / build_query_features,
# in order. Shared by dataset.py's feature sanity-check report and the notebook so
# both label columns identically without duplicating the list.
DOC_FEATURE_NAMES = [
    "num_doc_terms", "num_unique_doc_terms", "min_idf", "max_idf", "sum_idf",
    "overlap", "score", "bm25_tf_sum", "bm25_tf_max",
]
# DOC_FEATURE_NAMES minus "score" (the only column NOT derived from the
# Lucene index - it's copied straight from the run file, and can be rescaled
# by --score_scale, so it's never cached). This is the cacheable subset - see
# build_doc_term_features / feature_cache.build_doc_feature_cache /
# guide_docs/FEATURE_CACHE_GUIDE.md.
DOC_TERM_FEATURE_NAMES = [
    "num_doc_terms", "num_unique_doc_terms", "min_idf", "max_idf", "sum_idf",
    "overlap", "bm25_tf_sum", "bm25_tf_max",
]
# Once-per-(query, ranker) "list-level" features (see LIST_FEATURE_DIM):
# constant across all docs in a ranked list, so stored once instead of being
# duplicated into every doc's feature vector.
LIST_FEATURE_NAMES = ["mean_score", "var_score"]

# The three query-feature sub-blocks, in the fixed concatenation order used
# by build_full_query_features / resolve_query_feature_layout (which lets a
# caller select any subset of them while preserving this same relative
# order). The embedding block's raw width varies by source (768/384/...),
# so its names can't be a fixed module-level list - see
# embedding_feature_names below.
LEXICAL_FEATURE_NAMES = ["num_query_terms", "num_unique_query_terms", "min_idf", "max_idf", "sum_idf"]
QUERY_TYPE_FEATURE_NAMES = ["query_type"]
# 1-dim precomputed named-entity count for the query (see
# ENTITY_COUNT_FEATURE_DIM / build_entity_count_feature).
ENTITY_COUNT_FEATURE_NAMES = ["num_entities"]
# 3-dim precomputed (scs, avg_pmi, max_pmi) triple for the query (see
# SCS_PMI_FEATURE_DIM / build_scs_pmi_feature).
SCS_PMI_FEATURE_NAMES = ["scs", "avg_pmi", "max_pmi"]


def embedding_feature_names(dim: int) -> List[str]:
    """Column names for a `dim`-wide raw query-embedding block. `dim` is
    whatever width the loaded embedding source turned out to be (inferred at
    load/dataset-construction time, e.g. via feature_cache.load_query_embeddings
    or QPPDataset/QPPQueryOnlyDataset) - there is no fixed embedding width to
    default to."""
    return [f"query_emb_{i}" for i in range(dim)]

# Canonical order for the 6 selectable input feature blocks (see
# resolve_query_feature_layout / resolve_doc_feature_layout /
# validate_feature_blocks). "doc_feats" bundles list_feats (LIST_FEATURE_DIM)
# and the per-doc block (DOC_FEATURE_DIM * top_k) together as one toggle.
# "entity_count" is query-side, grouped with the other query blocks
# (lexical/embedding/query_type) ahead of the doc-side ones. "scs_pmi" is
# query-side, grouped with the other query blocks alongside "entity_count".
ALL_FEATURE_BLOCKS = (
    "lexical", "embedding", "query_type", "entity_count", "scs_pmi", "doc_feats",
)


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
    the selected blocks from {lexical, embedding, query_type, entity_count},
    in that fixed order.

    embedding_dim: required (and must be positive) when "embedding" is
    selected - the raw embedding block's width, inferred by the caller from
    whatever embedding source is loaded (e.g. feature_cache.load_query_embeddings
    / QPPDataset's embedding_cache.meta["embedding_dim"]), since it varies by
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
    if "entity_count" in feature_blocks:
        dim += ENTITY_COUNT_FEATURE_DIM
        names += ENTITY_COUNT_FEATURE_NAMES
    if "scs_pmi" in feature_blocks:
        dim += SCS_PMI_FEATURE_DIM
        names += SCS_PMI_FEATURE_NAMES
    return dim, names, embedding_slice


def resolve_doc_feature_layout(feature_blocks) -> Tuple[int, int, List[str], List[str]]:
    """
    Given a set/sequence of selected feature block names, returns
    (doc_feature_dim, list_feature_dim, doc_feature_names, list_feature_names).

    doc_feature_dim/doc_feature_names and list_feature_dim/list_feature_names
    come ONLY from "doc_feats" (DOC_FEATURE_DIM per doc, plus
    LIST_FEATURE_DIM once per ranker); if "doc_feats" is not selected
    they're all (0, []).
    """
    doc_feature_dim = 0
    doc_feature_names: List[str] = []
    list_feature_dim = 0
    list_feature_names: List[str] = []

    if "doc_feats" in feature_blocks:
        doc_feature_dim = DOC_FEATURE_DIM
        doc_feature_names = list(DOC_FEATURE_NAMES)
        list_feature_dim += LIST_FEATURE_DIM
        list_feature_names += LIST_FEATURE_NAMES

    return doc_feature_dim, list_feature_dim, doc_feature_names, list_feature_names


class IndexStats:
    """IDF and term-level statistics from a Pyserini Lucene index."""

    def __init__(self, index_path: str):
        from pyserini.index.lucene import LuceneIndexReader
        self.reader = LuceneIndexReader(index_path)
        stats = self.reader.stats()
        self._num_docs: int = stats["documents"]
        total_terms = stats.get("total_terms", 0)
        self._avg_dl: float = (total_terms / self._num_docs) if self._num_docs > 0 else 1.0
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

    def doc_term_counts(self, doc_id: str) -> Optional[Counter]:
        try:
            tf_map = self.reader.get_document_vector(doc_id)
        except Exception:
            tf_map = None
        if tf_map:
            return Counter(tf_map)

        # Some prebuilt indexes (e.g. msmarco-v1-passage via
        # from_prebuilt_index) aren't built with -storeDocvectors, so
        # get_document_vector always comes back empty. Fall back to
        # tokenizing the stored raw text with the same `.lower().split()`
        # scheme used for query terms (see dataset.py), so doc/query terms
        # stay comparable for `overlap` and `bm25_tf`.
        try:
            doc = self.reader.doc(doc_id)
            raw = doc.raw() if doc else None
            contents = json.loads(raw).get("contents", "") if raw else ""
        except Exception:
            return None
        tokens = contents.lower().split()
        return Counter(tokens) if tokens else None

    def bm25_tf(self, term: str, tf_map: Counter, doc_len: int,
                k1: float = 1.2, b: float = 0.75) -> float:
        tf = tf_map.get(term, 0)
        if tf == 0:
            return 0.0
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self._avg_dl))


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


def build_entity_count_feature(
    qid: str, entity_count_lookup: Dict[str, float]
) -> np.ndarray:
    """1-dim entity_count feature: the precomputed named-entity count for qid
    (see data/cache/entity-counts/query_entity_counts.csv,
    feature_cache.load_entity_count_lookup), looked up directly - no live
    computation here, same convention as build_embedding_feature.

    Raises KeyError if qid has no precomputed count in entity_count_lookup -
    this is deliberate (fail loudly rather than silently zero-filling)."""
    if qid not in entity_count_lookup:
        raise KeyError(
            f"No precomputed entity count for qid={qid!r} in the provided "
            f"--entity_counts_csv / entity_count lookup."
        )
    return np.array([float(entity_count_lookup[qid])], dtype=np.float32)


def build_scs_pmi_feature(
    qid: str, scs_pmi_lookup: Dict[str, np.ndarray]
) -> np.ndarray:
    """3-dim (scs, avg_pmi, max_pmi) feature: the precomputed triple for qid
    (see feature_cache.load_scs_pmi_lookup, which mean-imputes any missing
    avg_pmi/max_pmi in the source CSV), looked up directly - no live
    computation here, same convention as build_entity_count_feature.

    Raises KeyError if qid has no precomputed triple in scs_pmi_lookup - this
    is deliberate (fail loudly rather than silently zero-filling)."""
    if qid not in scs_pmi_lookup:
        raise KeyError(
            f"No precomputed scs/pmi triple for qid={qid!r} in the provided "
            f"--scs_pmi_csv / scs_pmi lookup."
        )
    return np.asarray(scs_pmi_lookup[qid], dtype=np.float32)


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


def build_doc_term_features(
    doc_id: str,
    query_terms: List[str],
    index_stats: IndexStats,
) -> np.ndarray:
    """8-dim Lucene-derived per-document feature vector (DOC_TERM_FEATURE_NAMES)
    - everything in DOC_FEATURE_NAMES except `score`, the only column NOT
    derived from the index (see DOC_TERM_FEATURE_NAMES). This is the
    cacheable part of build_doc_features - see
    feature_cache.build_doc_feature_cache / guide_docs/FEATURE_CACHE_GUIDE.md."""
    tc = index_stats.doc_term_counts(doc_id)

    if tc is not None:
        doc_len = sum(tc.values())
        doc_idfs = [index_stats.idf(t) for t in tc]
        num_doc_terms = float(doc_len)
        num_unique_doc_terms = float(len(tc))
        min_idf = min(doc_idfs) if doc_idfs else 0.0
        max_idf = max(doc_idfs) if doc_idfs else 0.0
        sum_idf = sum(doc_idfs)
        overlap = float(len(set(query_terms) & set(tc)))
        bm25_tf_vals = [index_stats.bm25_tf(t, tc, doc_len) for t in query_terms]
        bm25_tf_sum = sum(bm25_tf_vals)
        bm25_tf_max = max(bm25_tf_vals) if bm25_tf_vals else 0.0
    else:
        num_doc_terms = 0.0
        num_unique_doc_terms = 0.0
        min_idf = 0.0
        max_idf = 0.0
        sum_idf = 0.0
        overlap = 0.0
        bm25_tf_sum = 0.0
        bm25_tf_max = 0.0

    return np.array(
        [
            num_doc_terms,
            num_unique_doc_terms,
            min_idf,
            max_idf,
            sum_idf,
            overlap,
            bm25_tf_sum,
            bm25_tf_max,
        ],
        dtype=np.float32,
    )


def assemble_doc_feature(term_feats: np.ndarray, score: float) -> np.ndarray:
    """Reassemble the full DOC_FEATURE_DIM-dim (9) per-document feature
    vector from a cached-or-computed DOC_TERM_FEATURE_NAMES-dim (8) term
    vector (build_doc_term_features) plus the live `score` value - inserted
    at SCORE_FEATURE_INDICES[0] (its fixed position). `score` is never part
    of the cacheable term vector: it's copied straight from the run file and
    can be rescaled by --score_scale, so it must always stay dynamic (see
    guide_docs/FEATURE_CACHE_GUIDE.md)."""
    return np.insert(term_feats, SCORE_FEATURE_INDICES[0], score).astype(np.float32)


def build_doc_features(
    doc_id: str,
    query_terms: List[str],
    score: float,
    index_stats: IndexStats,
) -> np.ndarray:
    """9-dim per-document feature vector x_j: build_doc_term_features (8-dim,
    Lucene-derived) with `score` inserted at SCORE_FEATURE_INDICES[0] (see
    assemble_doc_feature). mean_score/var_score (constant across a whole
    ranked list) are NOT included here - they're built once per (query,
    ranker) instead, see LIST_FEATURE_NAMES / dataset.py."""
    term_feats = build_doc_term_features(doc_id, query_terms, index_stats)
    return assemble_doc_feature(term_feats, score)