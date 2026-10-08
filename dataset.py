import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from build_caches.feature_cache import FeatureCache
from features import (
    ALL_FEATURE_BLOCKS,
    DOC_FEATURE_DIM,
    DOC_FEATURE_NAMES,
    LEXICAL_FEATURE_DIM,
    LEXICAL_FEATURE_NAMES,
    LIST_FEATURE_DIM,
    LIST_FEATURE_NAMES,
    QUERY_TYPE_FEATURE_DIM,
    QUERY_TYPE_FEATURE_NAMES,
    SCORE_FEATURE_INDICES,
    IndexStats,
    QueryTypeClassifier,
    assemble_doc_feature,
    build_doc_term_features,
    build_embedding_feature,
    build_entity_count_feature,
    build_query_features,
    build_query_type_feature,
    build_scs_pmi_feature,
    embedding_feature_names,
    resolve_doc_feature_layout,
    resolve_query_feature_layout,
    validate_feature_blocks,
)

SCORE_NORM_CHOICES = ("global", "per_query", "per_ranker")


def _canonical_run_name(path: str) -> str:
    """Strip year tokens so train/eval stems match (e.g. BM25.2019.100 → BM25.100)."""
    stem = Path(path).stem
    stem = re.sub(r'\.(20\d{2})', '', stem)       # .2019 / .2020
    stem = re.sub(r'_(1\d|2\d)(?=\.|$)', '', stem)  # _19 / _20
    return stem


def load_run(run_paths: List[str]) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Parse one or more TREC-format run files.

    Returns:
        {run_name: {qid: [(docid, score), ...]}} sorted by score descending.
    """
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run_path in run_paths:
        run_name = _canonical_run_name(run_path)
        with open(run_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                qid, _, docid, _rank, score, _ = parts[:6]
                runs[run_name][qid].append((docid, float(score)))

    for run_name in runs:
        for qid in runs[run_name]:
            runs[run_name][qid].sort(key=lambda x: -x[1])

    return dict(runs)


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Parse a TREC qrels file. Returns {qid: {docid: relevance}}."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            qrels[qid][docid] = int(rel)
    return dict(qrels)


@lru_cache(maxsize=None)
def load_precomputed_metrics(csv_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load a precomputed per-(ranker, qid) metrics CSV (columns: qid, ranker,
    plus one column per pytrec_eval metric key, e.g. ndcg_cut_100) into a
    {(ranker, qid): {metric_key: value}} lookup. Cached per path so repeated
    calls (once per metric, once per checkpoint) only hit disk once.

    See guide_docs/PRECOMPUTED_METRICS_GUIDE.md."""
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        metric_cols = [c for c in reader.fieldnames if c not in ("qid", "ranker")]
        for row in reader:
            lookup[(row["ranker"], row["qid"])] = {c: float(row[c]) for c in metric_cols}
    if not lookup:
        raise ValueError(f"No rows loaded from metrics CSV: {csv_path}")
    return lookup


def _require_metric_column(
    lookup: Dict[Tuple[str, str], Dict[str, float]], metric_key: str, csv_path: str
) -> None:
    """Raise a clear error if metric_key is missing from a loaded metrics
    CSV's columns entirely - almost always a wrong CSV path/wrong dataset,
    unlike a single missing (ranker, qid) row (silently resolves to 0.0,
    per guide_docs/PRECOMPUTED_METRICS_GUIDE.md's documented behavior)."""
    available = next(iter(lookup.values()))
    if metric_key not in available:
        raise ValueError(
            f"Metrics CSV {csv_path!r} has no {metric_key!r} column. "
            f"Available columns: {sorted(available.keys())}."
        )


def build_ranker_to_id(
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    ranker_to_id: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Ranker (run) name -> integer id, consistent across train/eval.

    Built from sorted run names unless an explicit map is injected (used at
    eval time to reuse the training map so that a ranker's column index
    means the same ranker in both). Raises if `runs` contains a ranker not
    present in an explicitly provided map, since a silently-added column
    would desync labels/predictions from the ranker they belong to.
    """
    if ranker_to_id is None:
        return {rn: i for i, rn in enumerate(sorted(runs.keys()))}
    ranker_to_id = dict(ranker_to_id)
    for rn in runs:
        if rn not in ranker_to_id:
            raise ValueError(
                f"Ranker '{rn}' is not in the provided ranker_to_id map. "
                f"Train and eval must share the same rankers (same run-file "
                f"names) for ranker columns to line up."
            )
    return ranker_to_id


class QPPDataset(Dataset):
    """
    One sample = one query, with all `num_rankers` rankers' data grouped
    together (this is what makes cross-entropy classification across rankers
    possible: each __getitem__ call returns a full set of per-ranker doc
    features and true labels for one query, sharing a single query feature
    vector). Rankers that have no run data for a given query are present as
    zero-filled rows with `ranker_mask[r] = False`, rather than being dropped.

    Returns per __getitem__:
        doc_feats   : (num_rankers, top_k, DOC_FEATURE_DIM)   float32
        list_feats  : (num_rankers, list_feature_dim)         float32  —
                       built ONCE per (query, ranker), never duplicated into
                       every doc's feature vector: mean_score/var_score (if
                       "doc_feats" selected - these used to be baked into
                       every doc's feature vector, constant across the whole
                       list, but are factored out so they aren't repeated
                       top_k times) - see model.py's QPPMLP, which places this at the start
                       of each ranker's flattened block.
        query_feats : (query_feature_dim,)                    float32  — shared
                       across all rankers, NOT duplicated per ranker
        pad_mask    : (num_rankers, top_k)                    bool  — True = padding doc position
        ranker_mask : (num_rankers,)                          bool  — True = ranker has data for this query
        labels      : (num_rankers,)                          float32  nDCG@100 (0.0 where ranker_mask is False)

    score_norm controls how the score-derived features (SCORE_FEATURE_INDICES,
    plus the LIST_FEATURE_NAMES mean_score/var_score slot) are treated. It
    only ever touches the doc_feats-derived portion of list_feats (the first
    LIST_FEATURE_DIM columns, when "doc_feats" is selected):
        "global"     : left raw here; the model globally standardizes them.
        "per_query"  : transformed HERE into scale-free, within-list features:
                         - score (per-doc)       -> within-list z-score
                         - mean_score list slot  -> coefficient of variation (NQC-like)
                         - var_score list slot   -> normalized top-1 gap
                       (the model then globally standardizes the result, which
                       is near-identity for already-standardized columns).
        "per_ranker" : left raw here; the MODEL standardizes the per-doc
                       `score` column AND the whole list_feats tensor
                       (mean_score/var_score) per ranker, using the ranker's column position (see
                       `ranker_to_id`).

    ranker_to_id:
        If provided, run names are mapped to ids using this dict (and any run
        name not present raises — this is what keeps eval aligned to training,
        since a ranker's column index must mean the same ranker in both). If
        None, the map is built from sorted run names.

    metrics_csv:
        Path to a precomputed per-(ranker, qid) metrics CSV (see
        `load_precomputed_metrics` and guide_docs/PRECOMPUTED_METRICS_GUIDE.md)
        - the nDCG@100 training label is looked up from it. Required: there
        is no pytrec_eval fallback (supply a precomputed one).

    lexical_cache, embedding_cache, query_type_cache, entity_count_cache,
    scs_pmi_cache:
        Five SEPARATE, independently optional `feature_cache.FeatureCache`
        instances (see `feature_cache.py`) - the 5-dim lexical/IDF vector,
        the raw (variable-width, e.g. 768/384) query representation, the
        1-dim query_type flag, the 1-dim precomputed entity_count, and the
        3-dim precomputed (scs, avg_pmi, max_pmi) triple - are looked up from
        whichever of these caches has them, and concatenated (in that order)
        into the query_feats vector per query, shared across rankers - same
        convention as `QPPQueryOnlyDataset` (minus entity_count/scs_pmi,
        which that class doesn't support). The embedding block's raw width is
        inferred once (from `embedding_cache.meta`, or from
        `embedding_lookup`) and passed to `resolve_query_feature_layout` as
        `embedding_dim` - see `self.embedding_slice`/`self.query_feature_dim`
        below.

    doc_feature_cache:
        A SIXTH, independent `feature_cache.FeatureCache` instance, keyed by
        `.doc_content_feats[(qid, doc_id)]` rather than by qid - the 8-dim
        Lucene-derived per-document feature vector (everything in
        DOC_FEATURE_NAMES except `score`; see features.DOC_TERM_FEATURE_NAMES
        / features.build_doc_term_features). `score` itself is never cached
        (it's read live from the run file and can be rescaled by
        `score_scale`); it's spliced back in via `features.assemble_doc_feature`
        after the term-feature lookup. You can pass the same `FeatureCache`
        object to both `lexical_cache` and `doc_feature_cache` if a single
        artifact already has both fields populated (e.g. one built on another
        branch - see guide_docs/FEATURE_CACHE_GUIDE.md) - each param only
        ever reads its own field.

    index_stats, embedding_lookup, query_type_classifier, entity_count_lookup,
    scs_pmi_lookup:
        The live-computation fallback for a cache miss, one per piece:
        `index_stats` (features.build_query_features for lexical,
        features.build_doc_term_features for doc_feature_cache) backs a
        `lexical_cache`/`doc_feature_cache` miss, `embedding_lookup` (a
        {qid: raw embedding vector} dict, variable width) backs an
        `embedding_cache` miss, `query_type_classifier` backs a
        `query_type_cache` miss,
        `entity_count_lookup` (a {qid: count} dict, see
        feature_cache.load_entity_count_lookup) backs an `entity_count_cache`
        miss, and `scs_pmi_lookup` (a {qid: [scs, avg_pmi, max_pmi]} dict,
        see feature_cache.load_scs_pmi_lookup) backs a `scs_pmi_cache` miss.
        Each is optional when its corresponding cache fully covers this
        dataset's queries/docs; if a miss occurs on a piece with no fallback
        available, a RuntimeError is raised naming the missing qid/
        (qid, doc_id) pair rather than silently failing.

    require_caches:
        By default, a cache miss on any of the six pieces (lexical,
        embedding, query_type, entity_count, scs_pmi, doc_feats) silently
        falls back to computing it live via
        `index_stats`/`embedding_lookup`/`query_type_classifier`/
        `entity_count_lookup`/`scs_pmi_lookup` when available - even if the
        caller intended the cache to be a hard requirement. Set
        require_caches=True to disable ALL SIX live fallbacks regardless of
        whether those objects were passed, so any cache missing an entry
        raises immediately instead of recomputing live. `train.py`/
        `evaluate.py` always pass this as True, so they never touch Lucene at
        all once every selected block has a cache.

    feature_blocks:
        Which of the 6 input blocks (see features.ALL_FEATURE_BLOCKS:
        "lexical", "embedding", "query_type", "entity_count", "scs_pmi",
        "doc_feats") to actually build, in any combination. Defaults to all
        six. A block left out is never
        computed - e.g. omitting
        "doc_feats" skips ALL per-doc Lucene work entirely (no index_stats
        calls at all), same efficiency as QPPQueryOnlyDataset; omitting
        "embedding" means embedding_lookup/embedding_cache are never
        consulted, etc. The resulting query_feats/doc_feats/list_feats
        tensors are narrower accordingly (a fully-excluded block has 0
        width, not simply zeroed) - see the actual resolved dims/names on
        this instance: `query_feature_dim`, `doc_feature_dim`,
        `list_feature_dim`, `query_feature_names`, `doc_feature_names`,
        `list_feature_names`, `embedding_slice` (all derived via
        features.resolve_query_feature_layout /
        features.resolve_doc_feature_layout - the single source of truth
        train.py/evaluate.py read model dimensions from, instead of
        importing fixed constants - the embedding block's raw width in
        particular is never fixed, so this instance is the only source of
        truth for it).
    """

    def __init__(
        self,
        run_paths: List[str],
        qrels_path: str,
        queries: Dict[str, str],
        index_stats: Optional[IndexStats] = None,
        top_k: int = 10,
        score_scale: float = None,
        score_norm: str = "global",
        ranker_to_id: Optional[Dict[str, int]] = None,
        metrics_csv: Optional[str] = None,
        lexical_cache: Optional[FeatureCache] = None,
        embedding_cache: Optional[FeatureCache] = None,
        query_type_cache: Optional[FeatureCache] = None,
        entity_count_cache: Optional[FeatureCache] = None,
        scs_pmi_cache: Optional[FeatureCache] = None,
        doc_feature_cache: Optional[FeatureCache] = None,
        embedding_lookup: Optional[Dict[str, np.ndarray]] = None,
        query_type_classifier: Optional[QueryTypeClassifier] = None,
        entity_count_lookup: Optional[Dict[str, float]] = None,
        scs_pmi_lookup: Optional[Dict[str, np.ndarray]] = None,
        require_caches: bool = False,
        feature_blocks=ALL_FEATURE_BLOCKS,
    ):
        if score_norm not in SCORE_NORM_CHOICES:
            raise ValueError(f"score_norm must be one of {SCORE_NORM_CHOICES}")
        if metrics_csv is None:
            raise ValueError(
                "QPPDataset requires metrics_csv (a precomputed per-(ranker, "
                "qid) metrics CSV) - there is no pytrec_eval fallback. Supply "
                "a precomputed one."
            )
        validate_feature_blocks(feature_blocks)
        # Canonicalize to ALL_FEATURE_BLOCKS's fixed order regardless of the
        # order the caller passed them in, and de-duplicate.
        self.feature_blocks = tuple(b for b in ALL_FEATURE_BLOCKS if b in feature_blocks)
        use_lexical = "lexical" in self.feature_blocks
        use_embedding = "embedding" in self.feature_blocks
        use_query_type = "query_type" in self.feature_blocks
        use_entity_count = "entity_count" in self.feature_blocks
        use_scs_pmi = "scs_pmi" in self.feature_blocks
        use_doc_feats = "doc_feats" in self.feature_blocks

        embedding_dim = None
        if use_embedding:
            if embedding_cache is not None and "embedding_dim" in embedding_cache.meta:
                embedding_dim = embedding_cache.meta["embedding_dim"]
            elif embedding_lookup:
                embedding_dim = int(np.asarray(next(iter(embedding_lookup.values()))).shape[0])
            else:
                raise ValueError(
                    "'embedding' is in feature_blocks but the raw embedding "
                    "width couldn't be determined - embedding_cache has no "
                    "'embedding_dim' in its meta and no embedding_lookup was "
                    "provided to infer it from."
                )

        self.query_feature_dim, self.query_feature_names, self.embedding_slice = (
            resolve_query_feature_layout(self.feature_blocks, embedding_dim=embedding_dim)
        )
        (
            self.doc_feature_dim,
            self.list_feature_dim,
            self.doc_feature_names,
            self.list_feature_names,
        ) = resolve_doc_feature_layout(self.feature_blocks)

        self.top_k = top_k
        self.score_norm = score_norm
        self.samples: List[dict] = []
        # qid -> group dict accumulating per-ranker rows, keyed by ranker column.
        groups: Dict[str, dict] = {}
        # Per-piece cache hit tracking (lexical / embedding / query_type /
        # entity_count / doc), since each has its own independent cache and
        # fallback. Only populated/reported for blocks actually selected.
        cache_hits = {
            "lexical": 0, "embedding": 0, "query_type": 0, "entity_count": 0, "scs_pmi": 0,
        }
        cache_total = 0
        doc_cache_hits = 0
        doc_cache_total = 0

        runs = load_run(run_paths)
        qrels = load_qrels(qrels_path)

        # Ranker -> integer id. Built from sorted run names unless an explicit
        # map is injected (used at eval time to reuse the training map so that
        # per-ranker statistics line up with the same rankers).
        self.ranker_to_id = build_ranker_to_id(runs, ranker_to_id)
        self.num_rankers = len(self.ranker_to_id)

        eps = 1e-8
        (i_score,) = SCORE_FEATURE_INDICES

        metrics_lookup = load_precomputed_metrics(metrics_csv)
        _require_metric_column(metrics_lookup, "ndcg_cut_100", metrics_csv)

        for run_name, run_data in runs.items():
            ranker_id = self.ranker_to_id[run_name]

            for qid, doc_list in run_data.items():
                if qid not in qrels or qid not in queries:
                    continue

                raw_query = queries[qid]
                query_terms = raw_query.lower().split()
                if not query_terms:
                    continue

                label = metrics_lookup.get((run_name, qid), {}).get("ndcg_cut_100", 0.0)

                if qid not in groups:
                    cache_total += 1
                    pieces = []

                    if use_lexical:
                        if lexical_cache is not None and qid in lexical_cache.query_feats:
                            lexical_vec = lexical_cache.query_feats[qid]
                            cache_hits["lexical"] += 1
                        elif not require_caches and index_stats is not None:
                            lexical_vec = build_query_features(query_terms, index_stats)
                        else:
                            raise RuntimeError(
                                f"No lexical_cache entry for qid={qid!r} and no "
                                f"live fallback is available (require_caches="
                                f"{require_caches}). Rebuild the lexical "
                                f"feature cache to cover this query."
                            )
                        pieces.append(lexical_vec)

                    if use_embedding:
                        if embedding_cache is not None and qid in embedding_cache.query_feats:
                            embedding_vec = embedding_cache.query_feats[qid]
                            cache_hits["embedding"] += 1
                        elif not require_caches and embedding_lookup is not None:
                            embedding_vec = build_embedding_feature(qid, embedding_lookup)
                        else:
                            raise RuntimeError(
                                f"No embedding_cache entry for qid={qid!r} and no "
                                f"live fallback is available (require_caches="
                                f"{require_caches}). Rebuild the embedding "
                                f"feature cache to cover this query."
                            )
                        pieces.append(embedding_vec)

                    if use_query_type:
                        if query_type_cache is not None and qid in query_type_cache.query_feats:
                            query_type_vec = query_type_cache.query_feats[qid]
                            cache_hits["query_type"] += 1
                        elif not require_caches and query_type_classifier is not None:
                            query_type_vec = build_query_type_feature(raw_query, query_type_classifier)
                        else:
                            raise RuntimeError(
                                f"No query_type_cache entry for qid={qid!r} and no "
                                f"live fallback is available (require_caches="
                                f"{require_caches}). Rebuild the query_type "
                                f"feature cache to cover this query."
                            )
                        pieces.append(query_type_vec)

                    if use_entity_count:
                        if entity_count_cache is not None and qid in entity_count_cache.query_feats:
                            entity_count_vec = entity_count_cache.query_feats[qid]
                            cache_hits["entity_count"] += 1
                        elif not require_caches and entity_count_lookup is not None:
                            entity_count_vec = build_entity_count_feature(qid, entity_count_lookup)
                        else:
                            raise RuntimeError(
                                f"No entity_count_cache entry for qid={qid!r} and no "
                                f"live fallback is available (require_caches="
                                f"{require_caches}). Rebuild the entity_count "
                                f"feature cache to cover this query."
                            )
                        pieces.append(entity_count_vec)

                    if use_scs_pmi:
                        if scs_pmi_cache is not None and qid in scs_pmi_cache.query_feats:
                            scs_pmi_vec = scs_pmi_cache.query_feats[qid]
                            cache_hits["scs_pmi"] += 1
                        elif not require_caches and scs_pmi_lookup is not None:
                            scs_pmi_vec = build_scs_pmi_feature(qid, scs_pmi_lookup)
                        else:
                            raise RuntimeError(
                                f"No scs_pmi_cache entry for qid={qid!r} and no "
                                f"live fallback is available (require_caches="
                                f"{require_caches}). Rebuild the scs_pmi "
                                f"feature cache to cover this query."
                            )
                        pieces.append(scs_pmi_vec)

                    query_feats = (
                        np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
                    )
                else:
                    query_feats = groups[qid]["query_feats"]

                if use_doc_feats:
                    doc_list = doc_list[:top_k]
                    scores = [s for _, s in doc_list]

                    if score_scale is not None and scores:
                        s_min, s_max = min(scores), max(scores)
                        rng = s_max - s_min
                        if rng > 0:
                            scores = [(s - s_min) / rng * score_scale for s in scores]
                            doc_list = [(d, s) for (d, _), s in zip(doc_list, scores)]
                        else:
                            scores = [score_scale / 2] * len(scores)
                            doc_list = [(d, score_scale / 2) for d, _ in doc_list]

                    mean_score = float(np.mean(scores)) if scores else 0.0
                    var_score = float(np.var(scores)) if scores else 0.0
                    base_list_feats = np.array([mean_score, var_score], dtype=np.float32)
                    doc_feats = []
                    for doc_id, score in doc_list:
                        doc_cache_total += 1
                        doc_key = (qid, doc_id)
                        if doc_feature_cache is not None and doc_key in doc_feature_cache.doc_content_feats:
                            term_feats = doc_feature_cache.doc_content_feats[doc_key]
                            doc_cache_hits += 1
                        elif not require_caches and index_stats is not None:
                            term_feats = build_doc_term_features(doc_id, query_terms, index_stats)
                        else:
                            raise RuntimeError(
                                f"No doc_feature_cache entry for (qid={qid!r}, "
                                f"doc_id={doc_id!r}) and no live fallback is "
                                f"available (require_caches={require_caches}). "
                                f"Rebuild the doc feature cache to cover this "
                                f"(qid, doc_id) pair."
                            )
                        feat = assemble_doc_feature(term_feats, score)

                        doc_feats.append(feat)

                    n_actual = len(doc_feats)
                    doc_feats_arr = (
                        np.stack(doc_feats)
                        if n_actual > 0
                        else np.zeros((0, self.doc_feature_dim), dtype=np.float32)
                    )

                    # per_query: replace the score-derived features with scale-free,
                    # within-list features. Done here because "within-list" == "within
                    # one sample". Uses only this list's own scores, so it needs no
                    # ranker identity and generalizes to unseen rankers.
                    if score_norm == "per_query" and n_actual > 0:
                        std_s = float(np.sqrt(var_score))
                        denom = std_s + eps
                        max_s = float(max(scores))
                        # score -> within-list z-score (per document)
                        doc_feats_arr[:, i_score] = (
                            doc_feats_arr[:, i_score] - mean_score
                        ) / denom
                        # mean_score list slot -> coefficient of variation (NQC-like);
                        # var_score list slot  -> normalized top-1 gap (how much the top
                        # doc stands out). Both constant across docs in this list, like
                        # the original features they replace.
                        cv = std_s / (abs(mean_score) + eps)
                        top1_gap = (max_s - mean_score) / denom
                        base_list_feats = np.array([cv, top1_gap], dtype=np.float32)

                    # pad to top_k
                    if n_actual < top_k:
                        pad = np.zeros((top_k - n_actual, self.doc_feature_dim), dtype=np.float32)
                        doc_feats_arr = np.vstack([doc_feats_arr, pad])
                    doc_feats_arr = doc_feats_arr.astype(np.float32)

                    pad_mask = np.zeros(top_k, dtype=bool)
                    pad_mask[n_actual:] = True
                else:
                    # doc_feats not selected: skip ALL per-doc Lucene work
                    # entirely (no build_doc_features calls) - same
                    # efficiency as QPPQueryOnlyDataset.
                    doc_feats_arr = np.zeros((top_k, self.doc_feature_dim), dtype=np.float32)
                    pad_mask = np.ones(top_k, dtype=bool)

                # list_feats comes only from "doc_feats": the mean_score/
                # var_score (or per_query's cv/top1_gap) slot.
                list_feats_row = (
                    base_list_feats if use_doc_feats else np.zeros(0, dtype=np.float32)
                )

                if qid not in groups:
                    groups[qid] = {
                        "qid": qid,
                        "doc_feats": np.zeros(
                            (self.num_rankers, top_k, self.doc_feature_dim), dtype=np.float32
                        ),
                        "list_feats": np.zeros(
                            (self.num_rankers, self.list_feature_dim), dtype=np.float32
                        ),
                        "pad_mask": np.ones((self.num_rankers, top_k), dtype=bool),
                        "ranker_mask": np.zeros(self.num_rankers, dtype=bool),
                        "labels": np.zeros(self.num_rankers, dtype=np.float32),
                        "query_feats": query_feats,          # (query_feature_dim,) same for every ranker
                    }
                g = groups[qid]
                g["doc_feats"][ranker_id] = doc_feats_arr
                g["list_feats"][ranker_id] = list_feats_row
                g["pad_mask"][ranker_id] = pad_mask
                g["ranker_mask"][ranker_id] = True
                g["labels"][ranker_id] = label

        self.samples = list(groups.values())

        active_pieces = {
            "lexical": use_lexical, "embedding": use_embedding, "query_type": use_query_type,
            "entity_count": use_entity_count, "scs_pmi": use_scs_pmi,
        }
        active_cache_hits = {k: v for k, v in cache_hits.items() if active_pieces[k]}
        if cache_total > 0 and active_cache_hits and (
            lexical_cache or embedding_cache or query_type_cache or entity_count_cache
            or scs_pmi_cache
        ):
            summary = " ".join(f"{k}={v}/{cache_total}" for k, v in active_cache_hits.items())
            print(f"  feature cache hits: {summary}")
        if doc_cache_total > 0 and doc_feature_cache is not None:
            print(f"  doc feature cache hits: {doc_cache_hits}/{doc_cache_total}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["doc_feats"]),      # (R, top_k, F)
            torch.from_numpy(s["list_feats"]),     # (R, LIST_FEATURE_DIM)
            torch.from_numpy(s["query_feats"]),    # (Q,)
            torch.from_numpy(s["pad_mask"]),       # (R, top_k)
            torch.from_numpy(s["ranker_mask"]),    # (R,)
            torch.from_numpy(s["labels"]),         # (R,)
        )


class QPPQueryOnlyDataset(Dataset):
    """
    Query-features-only ablation of QPPDataset: same per-query, per-ranker
    nDCG@100 labels, but with the ranked-list / doc-feature side dropped
    entirely (no Lucene doc-vector or BM25 lookups), leaving just the query
    feature vector u_q per query plus per-ranker labels, matching how
    QueryOnlyMLP (model.py) consumes them.

    Every query in this data has all `num_rankers` rankers present, so
    unlike QPPDataset there is no per-ranker "absent for this query" case
    to track - no ranker_mask.

    Returns per __getitem__:
        query_feats : (query_feature_dim,)   float32
        labels      : (num_rankers,)         float32  nDCG@100

    ranker_to_id: see QPPDataset — must match across train/eval so ranker
    columns mean the same ranker in both.

    lexical_cache, embedding_cache, query_type_cache:
        Three SEPARATE, independently optional `feature_cache.FeatureCache`
        instances (see `feature_cache.py`) - the 5-dim lexical/IDF vector,
        the raw (variable-width, e.g. 768/384) query representation, and the
        1-dim query_type flag are looked up from whichever of these caches
        has them, and concatenated (in that order) into `query_feats` per
        query. Each is independent: e.g. you can have a real precomputed
        `lexical_cache` (no Lucene needed at dataset-build time) while
        computing embedding/query_type live, or any other mix. The
        embedding block's raw width is inferred once (from
        `embedding_cache.meta["embedding_dim"]`, or from `embedding_lookup`)
        and exposed as `self.embedding_slice`/`self.query_feature_dim`/
        `self.query_feature_names` - never hardcoded.

    index_stats, embedding_lookup, query_type_classifier:
        The live-computation fallback for a cache miss, one per piece:
        `index_stats` (features.build_query_features) backs a
        `lexical_cache` miss, `embedding_lookup` (a {qid: raw embedding
        vector} dict, variable width, see `feature_cache.load_query_embeddings`,
        via features.build_embedding_feature) backs an `embedding_cache`
        miss, and `query_type_classifier` (a `features.QueryTypeClassifier`,
        via features.build_query_type_feature) backs a `query_type_cache`
        miss. Each is optional when its corresponding cache fully covers
        this dataset's queries; if a miss occurs on a piece with no
        fallback available, a RuntimeError is raised naming the missing qid
        rather than silently failing.

    metrics_csv:
        Path to a precomputed per-(ranker, qid) metrics CSV (see
        `load_precomputed_metrics` and guide_docs/PRECOMPUTED_METRICS_GUIDE.md)
        - the nDCG@100 label is looked up from it. Required: there is no
        pytrec_eval fallback (supply a precomputed one).
    """

    def __init__(
        self,
        run_paths: List[str],
        qrels_path: str,
        queries: Dict[str, str],
        index_stats: Optional[IndexStats] = None,
        ranker_to_id: Optional[Dict[str, int]] = None,
        lexical_cache: Optional[FeatureCache] = None,
        embedding_cache: Optional[FeatureCache] = None,
        query_type_cache: Optional[FeatureCache] = None,
        embedding_lookup: Optional[Dict[str, np.ndarray]] = None,
        query_type_classifier: Optional[QueryTypeClassifier] = None,
        metrics_csv: Optional[str] = None,
    ):
        if metrics_csv is None:
            raise ValueError(
                "QPPQueryOnlyDataset requires metrics_csv (a precomputed "
                "per-(ranker, qid) metrics CSV) - there is no pytrec_eval "
                "fallback. Supply a precomputed one."
            )
        self.samples: List[dict] = []
        groups: Dict[str, dict] = {}
        # Per-piece cache hit tracking (lexical / embedding / query_type),
        # since each has its own independent cache and fallback.
        cache_hits = {"lexical": 0, "embedding": 0, "query_type": 0}
        cache_total = 0

        runs = load_run(run_paths)
        qrels = load_qrels(qrels_path)

        self.ranker_to_id = build_ranker_to_id(runs, ranker_to_id)
        self.num_rankers = len(self.ranker_to_id)

        metrics_lookup = load_precomputed_metrics(metrics_csv)
        _require_metric_column(metrics_lookup, "ndcg_cut_100", metrics_csv)

        # Raw embedding width isn't fixed (768 for BERT/Contriever, 384 for
        # MiniLM, etc.) - infer it once, same convention as QPPDataset, and
        # use it to fix this dataset's (fixed lexical+embedding+query_type)
        # legacy layout, replacing the old hardcoded QUERY_FEATURE_DIM.
        if embedding_cache is not None and "embedding_dim" in embedding_cache.meta:
            embedding_dim = embedding_cache.meta["embedding_dim"]
        elif embedding_lookup:
            embedding_dim = int(np.asarray(next(iter(embedding_lookup.values()))).shape[0])
        else:
            raise ValueError(
                "QPPQueryOnlyDataset couldn't determine the raw embedding "
                "width - embedding_cache has no 'embedding_dim' in its meta "
                "and no embedding_lookup was provided to infer it from."
            )
        self.embedding_slice = slice(LEXICAL_FEATURE_DIM, LEXICAL_FEATURE_DIM + embedding_dim)
        self.query_feature_dim = LEXICAL_FEATURE_DIM + embedding_dim + QUERY_TYPE_FEATURE_DIM
        self.query_feature_names = (
            LEXICAL_FEATURE_NAMES + embedding_feature_names(embedding_dim) + QUERY_TYPE_FEATURE_NAMES
        )

        for run_name, run_data in runs.items():
            ranker_id = self.ranker_to_id[run_name]

            for qid in run_data:
                if qid not in qrels or qid not in queries:
                    continue

                raw_query = queries[qid]
                query_terms = raw_query.lower().split()
                if not query_terms:
                    continue

                label = metrics_lookup.get((run_name, qid), {}).get("ndcg_cut_100", 0.0)

                if qid not in groups:
                    cache_total += 1

                    if lexical_cache is not None and qid in lexical_cache.query_feats:
                        lexical_vec = lexical_cache.query_feats[qid]
                        cache_hits["lexical"] += 1
                    elif index_stats is not None:
                        lexical_vec = build_query_features(query_terms, index_stats)
                    else:
                        raise RuntimeError(
                            f"No lexical_cache entry for qid={qid!r} and no "
                            f"index_stats fallback was provided. Rebuild the "
                            f"lexical feature cache to cover this query, or "
                            f"pass --index."
                        )

                    if embedding_cache is not None and qid in embedding_cache.query_feats:
                        embedding_vec = embedding_cache.query_feats[qid]
                        cache_hits["embedding"] += 1
                    elif embedding_lookup is not None:
                        embedding_vec = build_embedding_feature(qid, embedding_lookup)
                    else:
                        raise RuntimeError(
                            f"No embedding_cache entry for qid={qid!r} and no "
                            f"embedding_lookup fallback was provided. Rebuild "
                            f"the embedding feature cache to cover this query, "
                            f"or pass --query_embeddings."
                        )

                    if query_type_cache is not None and qid in query_type_cache.query_feats:
                        query_type_vec = query_type_cache.query_feats[qid]
                        cache_hits["query_type"] += 1
                    elif query_type_classifier is not None:
                        query_type_vec = build_query_type_feature(raw_query, query_type_classifier)
                    else:
                        raise RuntimeError(
                            f"No query_type_cache entry for qid={qid!r} and no "
                            f"query_type_classifier fallback was provided. "
                            f"Rebuild the query_type feature cache to cover "
                            f"this query, or pass a query_type_classifier."
                        )

                    query_feats = np.concatenate([lexical_vec, embedding_vec, query_type_vec])
                    groups[qid] = {
                        "qid": qid,
                        "query_feats": query_feats,
                        "labels": np.zeros(self.num_rankers, dtype=np.float32),
                    }
                g = groups[qid]
                g["labels"][ranker_id] = label

        self.samples = list(groups.values())

        if cache_total > 0 and (lexical_cache or embedding_cache or query_type_cache):
            print(
                "  feature cache hits: "
                f"lexical={cache_hits['lexical']}/{cache_total} "
                f"embedding={cache_hits['embedding']}/{cache_total} "
                f"query_type={cache_hits['query_type']}/{cache_total}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["query_feats"]),   # (Q,)
            torch.from_numpy(s["labels"]),         # (R,)
        )


def _feature_col_stats(col: np.ndarray) -> dict:
    if len(col) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "nonzero": 0, "n": 0}
    return {
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": float(col.mean()),
        "nonzero": int(np.count_nonzero(col)),
        "n": int(len(col)),
    }


def _feature_block_report(
    doc_rows: List[np.ndarray],
    query_rows: List[np.ndarray],
    list_rows: Optional[List[np.ndarray]] = None,
    doc_feature_names: Optional[List[str]] = None,
    list_feature_names: Optional[List[str]] = None,
    query_feature_names: Optional[List[str]] = None,
) -> dict:
    """Per-feature min/max/mean/nonzero, plus which term-based doc features
    (everything except SCORE_FEATURE_INDICES) came back constant zero -- the
    tell that IndexStats.doc_term_counts isn't returning real term data.
    list_rows (mean_score/var_score, once per (query, ranker) - see
    LIST_FEATURE_NAMES) is optional since QPPQueryOnlyDataset has none.

    doc_feature_names/list_feature_names default to the fixed global name
    lists (DOC_FEATURE_NAMES/LIST_FEATURE_NAMES) when omitted.
    query_feature_names has no such default - the embedding block's raw
    width (and therefore its names) is variable, not fixed, so both
    build_feature_sanity_report and build_query_only_feature_sanity_report
    always pass the dataset instance's actual query_feature_names
    explicitly."""
    doc_feature_names = doc_feature_names if doc_feature_names is not None else DOC_FEATURE_NAMES
    list_feature_names = list_feature_names if list_feature_names is not None else LIST_FEATURE_NAMES
    if query_feature_names is None:
        raise ValueError("_feature_block_report requires query_feature_names to be passed explicitly.")
    list_rows = list_rows or []
    doc_arr = (
        np.stack(doc_rows) if doc_rows else np.zeros((0, len(doc_feature_names)), dtype=np.float32)
    )
    list_arr = (
        np.stack(list_rows) if list_rows else np.zeros((0, len(list_feature_names)), dtype=np.float32)
    )
    query_arr = (
        np.stack(query_rows) if query_rows else np.zeros((0, len(query_feature_names)), dtype=np.float32)
    )
    doc_stats = {name: _feature_col_stats(doc_arr[:, i]) for i, name in enumerate(doc_feature_names)}
    list_stats = {name: _feature_col_stats(list_arr[:, i]) for i, name in enumerate(list_feature_names)}
    query_stats = {name: _feature_col_stats(query_arr[:, i]) for i, name in enumerate(query_feature_names)}
    dead = [
        name for i, name in enumerate(doc_feature_names)
        if i not in SCORE_FEATURE_INDICES and doc_stats[name]["n"] > 0 and doc_stats[name]["nonzero"] == 0
    ]
    return {
        "n_doc_rows": len(doc_rows),
        "n_list_rows": len(list_rows),
        "n_query_rows": len(query_rows),
        "doc_features": doc_stats,
        "list_features": list_stats,
        "query_features": query_stats,
        "dead_term_features": dead,
    }


def build_feature_sanity_report(dataset: "QPPDataset") -> dict:
    """
    Summarize the actual doc/query feature tensors a built QPPDataset holds --
    per-feature min/max/mean/nonzero-count over the real (ranker-present,
    non-padded) doc rows across every ranker column, plus the fixed tensor
    shapes. Meant to be dumped to a JSON log file right after dataset
    construction so the feature pipeline can be sanity-checked after the fact
    without recomputing anything by hand.

    Shapes/names are read off the dataset instance (dataset.doc_feature_dim
    etc.), not the fixed global constants, since dataset.feature_blocks may
    have excluded some blocks (see QPPDataset's feature_blocks param) - an
    excluded block reports as a 0-width block with no names, not simply
    zeroed values.
    """
    top_k = dataset.top_k

    doc_rows: List[np.ndarray] = []
    list_rows: List[np.ndarray] = []
    query_rows: List[np.ndarray] = []
    for s in dataset.samples:
        doc_feats, pad_mask, ranker_mask = s["doc_feats"], s["pad_mask"], s["ranker_mask"]
        valid = ~pad_mask & ranker_mask[:, None]
        doc_rows.extend(doc_feats[valid])
        list_rows.extend(s["list_feats"][ranker_mask])
        query_rows.append(s["query_feats"])

    return {
        "num_samples": len(dataset.samples),
        "num_rankers": dataset.num_rankers,
        "top_k": top_k,
        "feature_blocks": list(dataset.feature_blocks),
        "doc_feats_shape": [dataset.num_rankers, top_k, dataset.doc_feature_dim],
        "list_feats_shape": [dataset.num_rankers, dataset.list_feature_dim],
        "query_feats_shape": [dataset.query_feature_dim],
        "doc_feature_names": dataset.doc_feature_names,
        "list_feature_names": dataset.list_feature_names,
        "query_feature_names": dataset.query_feature_names,
        "branches": {
            "original": _feature_block_report(
                doc_rows, query_rows, list_rows=list_rows,
                doc_feature_names=dataset.doc_feature_names,
                list_feature_names=dataset.list_feature_names,
                query_feature_names=dataset.query_feature_names,
            )
        },
    }


def build_query_only_feature_sanity_report(dataset: "QPPQueryOnlyDataset") -> dict:
    """
    Summarize the query feature tensors a built QPPQueryOnlyDataset holds --
    per-feature min/max/mean/nonzero-count across all samples, plus the
    tensor shape. Same convention as build_feature_sanity_report: shape/names
    are read off the dataset instance (dataset.query_feature_dim/
    dataset.query_feature_names), not fixed global constants, since the
    embedding block's raw width varies by source.
    """
    query_rows = [s["query_feats"] for s in dataset.samples]
    return {
        "num_samples": len(dataset.samples),
        "num_rankers": dataset.num_rankers,
        "query_feats_shape": [dataset.query_feature_dim],
        "query_feature_names": dataset.query_feature_names,
        "branches": {
            "original": _feature_block_report(
                [], query_rows, query_feature_names=dataset.query_feature_names,
            )
        },
    }
