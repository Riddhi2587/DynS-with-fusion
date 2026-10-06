"""
Dump a few computed feature vectors to JSON so the lexical / embedding /
query_type computation can be checked by eye. Shared by
time_feature_computation.py (train side) and time_eval_live.py (eval side).

Per sampled query the JSON holds:
  lexical     the 5 named values, the tokens, the per-term IDF the index returned,
              and `recomputed_matches`: the 5 values re-derived from those tokens
              and IDFs, compared with the stored vector (a self-check that the
              vector is consistent with its own inputs)
  embedding   dim, l2_norm (~1.0: this is the L2-normalized vector the model
              consumes) and the full vector
  query_type  the 0/1 value and its label (0 = keyword, 1 = natural_language)
A block that wasn't computed in the run is omitted.

Nothing here is on a timed path: callers build samples from vectors that were
already computed, after all timing is finished.
"""

import json
import os
import random
from typing import Dict, Iterable, List, Optional

import numpy as np

from embedding_live import MODEL_NAME as MINILM_MODEL_NAME
from features import LEXICAL_FEATURE_NAMES, QueryTypeClassifier

QUERY_TYPE_LABELS = {0.0: "keyword", 1.0: "natural_language"}


def pick_sample_qids(qids: Iterable[str], n: int, seed: int) -> List[str]:
    """Up to n random qids, in sample order. Sorted before sampling so the result
    doesn't depend on input order; fewer than n available just returns them all."""
    pool = sorted(qids)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def _lexical_entry(raw: str, vec, index_stats) -> Dict:
    values = {name: float(v) for name, v in zip(LEXICAL_FEATURE_NAMES, np.asarray(vec).tolist())}
    entry: Dict = {"values": values}
    if index_stats is None:
        return entry
    tokens = raw.lower().split()
    per_term_idf = {t: float(index_stats.idf(t)) for t in tokens}
    idfs = [per_term_idf[t] for t in tokens]
    recomputed = np.array(
        [
            float(len(tokens)),
            float(len(set(tokens))),
            min(idfs) if idfs else 0.0,
            max(idfs) if idfs else 0.0,
            sum(idfs),
        ],
        dtype=np.float32,
    )
    entry["tokens"] = tokens
    entry["per_term_idf"] = per_term_idf
    entry["recomputed_matches"] = bool(
        np.allclose(recomputed, np.asarray(vec, dtype=np.float32), rtol=1e-4, atol=1e-5)
    )
    return entry


def describe_query(
    qid: str,
    raw: str,
    lexical=None,
    embedding=None,
    query_type=None,
    index_stats=None,
) -> Dict:
    """JSON-serializable description of one query's computed features. Pass
    `index_stats` to also record per-term IDF and the lexical self-check (it
    performs IDF lookups, so call it outside any timed region)."""
    out: Dict = {"qid": qid, "query": raw}
    if lexical is not None:
        out["lexical"] = _lexical_entry(raw, lexical, index_stats)
    if embedding is not None:
        emb = np.asarray(embedding, dtype=np.float64)
        out["embedding"] = {
            "dim": int(emb.shape[0]),
            "l2_norm": float(np.linalg.norm(emb)),
            "values": np.asarray(embedding).tolist(),
        }
    if query_type is not None:
        value = float(np.asarray(query_type).reshape(-1)[0])
        out["query_type"] = {"value": value, "label": QUERY_TYPE_LABELS.get(value, "unknown")}
    return out


def write_feature_samples(
    path: str, side: str, dataset: str, seed: int, samples: List[Dict],
    meta: Optional[Dict] = None,
) -> None:
    payload = {
        "side": side,
        "dataset": dataset,
        "seed": seed,
        "n": len(samples),
        "lexical_feature_names": list(LEXICAL_FEATURE_NAMES),
        "embedding_model": MINILM_MODEL_NAME,
        "query_type_model": QueryTypeClassifier.MODEL_NAME,
        "notes": [
            "embedding.values is L2-normalized (the vector the MLP consumes); l2_norm should be ~1.0.",
            "query_type: 0 = keyword-based, 1 = natural-language.",
            "lexical.recomputed_matches re-derives the 5 values from tokens + per_term_idf.",
        ],
    }
    if meta:
        payload.update(meta)
    payload["samples"] = samples
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
