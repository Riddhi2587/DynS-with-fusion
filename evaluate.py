"""
Evaluate a trained QPPMLP: for each query, the model outputs a distribution
over the rankers (softmax over per-ranker logits). This script reports
classification accuracy (does the predicted top ranker match the true best
ranker?) plus correlation between predicted probability and the true metric
score, both overall and broken down per-ranker / per-query.

Example:
    python evaluate.py \
        --run data/dl20_runs.txt \
        --qrels data/dl20-passage.qrels \
        --queries data/dl20-queries.tsv \
        --lexical_cache dl20_lexical.pkl \
        --embedding_cache dl20_embedding.pkl \
        --query_type_cache dl20_query_type.pkl \
        --entity_count_cache dl20_entity_count.pkl \
        --scs_pmi_cache dl20_scs_pmi.pkl \
        --doc_feature_cache dl20_doc_features.pkl \
        --model_path model_best.pt \
        --score_norm per_ranker

Lucene is never touched here - every selected --features block must come
from a cache (see guide_docs/FEATURE_CACHE_GUIDE.md).

For per_ranker, pass --ranker_map pointing at the "<model_path>.rankers.json"
file written during training (evaluate.py looks for it automatically next to
--model_path if --ranker_map is not given).
"""

import argparse
import collections
import csv
import json
import os
import re

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, rankdata
from torch.utils.data import DataLoader

from dataset import QPPDataset, _require_metric_column, load_precomputed_metrics
from feature_cache import load_feature_cache
from features import ALL_FEATURE_BLOCKS, validate_feature_blocks
from model import DEFAULT_HIDDEN_DIMS, QPPMLP
from train import load_queries


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=64):
    """Returns (probs, ranker_masks), each (N_queries, num_rankers)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs, all_masks = [], []
    for doc_feats, list_feats, query_feats, pad_mask, ranker_mask, _ in loader:
        logits = model(
            doc_feats.to(device),
            list_feats.to(device),
            query_feats.to(device),
            pad_mask.to(device),
            ranker_mask.to(device),
        )
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_masks.append(ranker_mask.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_masks, axis=0)


def _corr(preds, labels):
    if len(preds) < 2:
        return float("nan"), float("nan")
    pr, _ = pearsonr(preds, labels)
    kt, _ = kendalltau(preds, labels)
    return float(pr), float(kt)


def _rbo(preds, labels, p=0.8):
    """Rank-Biased Overlap between the ranking induced by predicted scores
    and the ranking induced by true labels, over the same set of items (a
    query's valid rankers). Both rankings are complete permutations of the
    same finite item set (no indefinite/truncated tail), so this uses the
    exact closed form RBO(p) = (1-p) * sum_{d=1}^{k} p^(d-1) * A_d + p^k,
    rather than the Webber et al. extrapolated estimate meant for indefinite
    rankings. p is the persistence parameter controlling how much weight
    decays per rank.

    Ties (equal predicted scores or equal labels) are handled via competition
    ranking (1, 2, 2, 4, ...), so a tied group of items enters the "seen"
    prefix together at the depth of its best rank, rather than being split
    across depths by an arbitrary tiebreak. A_d is then the symmetric overlap
    2*|seen_pred ∩ seen_true| / (|seen_pred| + |seen_true|) - this is Webber
    et al.'s "tied" extension (also used by e.g. the dlukes/rbo package) -
    which reduces to the plain |intersection| / d formula when there are no
    ties, since both seen sets have exactly d items at every depth."""
    k = len(preds)
    if k < 2:
        return float("nan")
    pred_rank = rankdata(-np.asarray(preds), method="min")
    true_rank = rankdata(-np.asarray(labels), method="min")
    total = 0.0
    for d in range(1, k + 1):
        seen_pred = set(np.where(pred_rank <= d)[0])
        seen_true = set(np.where(true_rank <= d)[0])
        agreement = 2 * len(seen_pred & seen_true) / (len(seen_pred) + len(seen_true))
        total += (p ** (d - 1)) * agreement
    return float((1 - p) * total + p ** k)


def _reciprocal_rank(preds, labels):
    """Reciprocal rank of the true best-labeled item within the ranking
    induced by predicted scores. Ties for the best label value are resolved
    to the best (smallest) predicted rank among them, matching
    top1_accuracy's tie convention (ties for best label are equally
    correct)."""
    if len(preds) < 2:
        return float("nan")
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    pred_order = np.argsort(-preds, kind="stable")
    rank_of = {item: r + 1 for r, item in enumerate(pred_order)}
    best_val = np.max(labels)
    best_items = np.where(labels == best_val)[0]
    best_rank = min(rank_of[i] for i in best_items)
    return float(1.0 / best_rank)


def compute_labels_matrix(qids, id_to_ranker, metric, metrics_csv):
    """
    (N_queries, num_rankers) true metric-score matrix for any metric present
    as a column in metrics_csv (see dataset.load_precomputed_metrics and
    guide_docs/PRECOMPUTED_METRICS_GUIDE.md). There is no pytrec_eval
    fallback - build a metrics CSV with build_metrics_csv.py first.
    """
    key = metric.replace(".", "_")   # e.g. "ndcg_cut.10" -> "ndcg_cut_10"
    lookup = load_precomputed_metrics(metrics_csv)
    _require_metric_column(lookup, key, metrics_csv)

    num_rankers = len(id_to_ranker)
    labels = np.zeros((len(qids), num_rankers), dtype=np.float32)
    for i, qid in enumerate(qids):
        for r in range(num_rankers):
            labels[i, r] = lookup.get((id_to_ranker[r], qid), {}).get(key, 0.0)
    return labels


def topk_accuracy(probs, labels, ranker_masks, k=1):
    """Fraction of queries (with >=2 valid rankers) where the true best ranker
    is among the model's top-k predicted rankers."""
    correct = 0
    total = 0
    for i in range(probs.shape[0]):
        valid = np.where(ranker_masks[i])[0]
        if len(valid) < 2:
            continue
        true_best = valid[np.argmax(labels[i, valid])]
        top_pred = valid[np.argsort(-probs[i, valid])[:k]]
        correct += int(true_best in top_pred)
        total += 1
    return (correct / total if total else float("nan")), total


def top1_accuracy(probs, labels, ranker_masks):
    """Fraction of queries (with >=2 valid rankers) where the model's top-1
    predicted ranker is among the true best rankers, treating ties for the
    best label value as all equally correct."""
    correct = 0
    total = 0
    for i in range(probs.shape[0]):
        valid = np.where(ranker_masks[i])[0]
        if len(valid) < 2:
            continue
        best_val = np.max(labels[i, valid])
        true_best = valid[labels[i, valid] == best_val]
        top_pred = valid[np.argmax(probs[i, valid])]
        correct += int(top_pred in true_best)
        total += 1
    return (correct / total if total else float("nan")), total


def classification_counts(probs, labels, ranker_masks):
    """Per-class (tp, fp, fn, support) arrays of length num_rankers, treating
    "which ranker is best" as multiclass classification: for each query with
    >=2 valid rankers, predicted class = argmax of probs among valid rankers,
    true class = argmax of labels among valid rankers (first-max on ties)."""
    num_rankers = probs.shape[1]
    tp = np.zeros(num_rankers, dtype=np.int64)
    fp = np.zeros(num_rankers, dtype=np.int64)
    fn = np.zeros(num_rankers, dtype=np.int64)
    support = np.zeros(num_rankers, dtype=np.int64)
    for i in range(probs.shape[0]):
        valid = np.where(ranker_masks[i])[0]
        if len(valid) < 2:
            continue
        true_class = valid[np.argmax(labels[i, valid])]
        pred_class = valid[np.argmax(probs[i, valid])]
        support[true_class] += 1
        if pred_class == true_class:
            tp[pred_class] += 1
        else:
            fp[pred_class] += 1
            fn[true_class] += 1
    return tp, fp, fn, support


def classwise_prf1(probs, labels, ranker_masks, id_to_ranker):
    """Per-class precision/recall/F1 (0.0 under zero-division, matching
    sklearn's zero_division=0) plus macro_f1 averaged over classes with
    support > 0. Returns (per_class, macro_f1) where per_class is a list of
    dicts with keys ranker/precision/recall/f1/support."""
    tp, fp, fn, support = classification_counts(probs, labels, ranker_masks)
    per_class = []
    f1s = []
    for c in range(len(support)):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({
            "ranker": id_to_ranker[c],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support[c]),
        })
        if support[c] > 0:
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return per_class, macro_f1


def prediction_entropy(probs, ranker_masks, base=None):
    """Per-query entropy (natural log/nats by default; pass base=2 for bits)
    of the predicted distribution restricted to valid rankers, for queries
    with >=2 valid rankers. Returns a 1-D array, one entry per included query."""
    entropies = []
    log = np.log if base is None else (lambda x: np.log(x) / np.log(base))
    for i in range(probs.shape[0]):
        valid = np.where(ranker_masks[i])[0]
        if len(valid) < 2:
            continue
        p = probs[i, valid]
        p = p[p > 0]
        entropies.append(float(-np.sum(p * log(p))))
    return np.array(entropies, dtype=np.float64)


def parse_model_epoch(model_path):
    """Extract (model_name, epoch) from a checkpoint path saved by train.py's
    <stem>_epoch<N>.pt convention (see train.py's epoch_save_path). Falls back
    to epoch=None if the filename doesn't match that pattern."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    m = re.match(r"^(.*)_epoch(\d+)$", stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, None


def save_predictions(output_path, model_path, qids, probs, ranker_masks, id_to_ranker):
    """Appends one row per (qid, ranker) with the model's predicted probability
    to a shared CSV, so predictions from multiple model/epoch checkpoints can
    accumulate in the same file across separate evaluate.py runs."""
    model_name, epoch = parse_model_epoch(model_path)
    write_header = not (os.path.exists(output_path) and os.path.getsize(output_path) > 0)
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "epoch", "qid", "ranker", "score"])
        for i, qid in enumerate(qids):
            for r in np.where(ranker_masks[i])[0]:
                writer.writerow([model_name, epoch, qid, id_to_ranker[r], float(probs[i, r])])
    print(f"Appended predictions to {output_path}")


def report(probs, labels, ranker_masks, id_to_ranker, metric="ndcg_cut.100", rbo_p=0.8):
    mask_flat = ranker_masks.reshape(-1)
    preds_flat = probs.reshape(-1)[mask_flat]
    labels_flat = labels.reshape(-1)[mask_flat]

    pr, kt = _corr(preds_flat, labels_flat)
    print(f"\n=== Overall  [{metric}]  (n={len(preds_flat)}) ===")
    print(f"  Pearson : {pr:.4f}")
    print(f"  Kendall : {kt:.4f}")

    num_rankers = probs.shape[1]
    print(f"\n=== Per-Ranker  [{metric}] ===")
    ranker_prs, ranker_kts = [], []
    for r in range(num_rankers):
        m = ranker_masks[:, r]
        rp, rl = probs[m, r], labels[m, r]
        pr, kt = _corr(rp, rl)
        print(f"  {id_to_ranker[r]:40s}  Pearson={pr:.4f}  Kendall={kt:.4f}  n={len(rp)}")
        if not np.isnan(pr):
            ranker_prs.append(pr)
            ranker_kts.append(kt)
    if ranker_prs:
        print(f"  {'--- macro avg ---':40s}  Pearson={np.mean(ranker_prs):.4f}  Kendall={np.mean(ranker_kts):.4f}")

    per_q_pr, per_q_kt, per_q_rbo, per_q_rr = [], [], [], []
    for i in range(probs.shape[0]):
        valid = np.where(ranker_masks[i])[0]
        if len(valid) < 2:
            continue
        pr, kt = _corr(probs[i, valid], labels[i, valid])
        if not np.isnan(pr):
            per_q_pr.append(pr)
            per_q_kt.append(kt)
        per_q_rbo.append(_rbo(probs[i, valid], labels[i, valid], p=rbo_p))
        per_q_rr.append(_reciprocal_rank(probs[i, valid], labels[i, valid]))
    if per_q_pr:
        print(f"\n=== Per-Query macro-average (n_queries={len(per_q_pr)}) ===")
        print(f"  Avg Pearson : {np.mean(per_q_pr):.4f}")
        print(f"  Avg Kendall : {np.mean(per_q_kt):.4f}")
    if per_q_rbo:
        print(f"  Avg RBO(p={rbo_p:.2f}) : {np.mean(per_q_rbo):.4f}")
    if per_q_rr:
        print(f"  MRR             : {np.mean(per_q_rr):.4f}")

    top1, n1 = top1_accuracy(probs, labels, ranker_masks)
    print(f"\n=== Best-Ranker Classification  [{metric}] ===")
    print(f"  Top-1 accuracy : {top1:.4f}  (n={n1})")
    k3 = min(3, num_rankers)
    if k3 > 1:
        top3, n3 = topk_accuracy(probs, labels, ranker_masks, k=k3)
        print(f"  Top-{k3} accuracy : {top3:.4f}  (n={n3})")

    per_class, macro_f1 = classwise_prf1(probs, labels, ranker_masks, id_to_ranker)
    print(f"\n=== Classwise Precision/Recall/F1  [{metric}] ===")
    for c in per_class:
        print(
            f"  {c['ranker']:40s}  P={c['precision']:.4f}  R={c['recall']:.4f}  "
            f"F1={c['f1']:.4f}  support={c['support']}"
        )
    print(f"  {'--- macro avg F1 ---':40s}  {macro_f1:.4f}")

    entropies = prediction_entropy(probs, ranker_masks)
    if len(entropies):
        print("\n=== Prediction Entropy (nats) ===")
        print(f"  Mean : {np.mean(entropies):.4f}")
        print(f"  Std  : {np.std(entropies):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, nargs="+")
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument(
        "--metrics_csv", required=True,
        help="Path to a precomputed per-(ranker, qid) metrics CSV (see "
             "build_metrics_csv.py and guide_docs/PRECOMPUTED_METRICS_GUIDE.md). "
             "Required - there is no pytrec_eval fallback.",
    )
    parser.add_argument(
        "--features", nargs="+", default=None, choices=list(ALL_FEATURE_BLOCKS),
        help="Which input feature blocks the checkpoint was trained with, any "
             f"subset of {list(ALL_FEATURE_BLOCKS)}. Defaults to auto-detecting "
             "'<model_path>.features.json' (saved by train.py) if present, "
             "else falls back to all six with a warning.",
    )
    parser.add_argument(
        "--lexical_cache", default=None,
        help="Path to a precomputed lexical/IDF feature cache (see "
             "build_feature_cache.py). Required if 'lexical' is in the "
             "resolved --features - no live fallback (Lucene is never "
             "touched here; see QPPDataset's require_caches).",
    )
    parser.add_argument(
        "--embedding_cache", default=None,
        help="Path to a precomputed raw embedding feature cache (see "
             "build_embedding_cache.py, built from a raw source like "
             "data/cache/bert-query-embeddings/cls/*.cls.pkl - width is "
             "inferred, not fixed; must match the width used at training "
             "time). Required if 'embedding' is in the resolved --features - "
             "no live fallback.",
    )
    parser.add_argument(
        "--query_type_cache", default=None,
        help="Path to a precomputed query_type feature cache (see "
             "build_query_type_cache.py). Required if 'query_type' is in "
             "the resolved --features - no live fallback.",
    )
    parser.add_argument(
        "--entity_count_cache", default=None,
        help="Path to a precomputed entity_count feature cache (see "
             "build_entity_count_cache.py) - the 1-dim precomputed "
             "named-entity count per query. Required if 'entity_count' is "
             "in the resolved --features - no live fallback.",
    )
    parser.add_argument(
        "--scs_pmi_cache", default=None,
        help="Path to a precomputed scs_pmi feature cache (see "
             "build_scs_pmi_cache.py) - the 3-dim precomputed "
             "(scs, avg_pmi, max_pmi) triple per query. Required if "
             "'scs_pmi' is in the resolved --features - no live fallback.",
    )
    parser.add_argument(
        "--doc_feature_cache", default=None,
        help="Path to a precomputed doc-content feature cache (see "
             "build_doc_feature_cache.py) - the 8-dim Lucene-derived "
             "per-(qid, doc_id) term features (features.DOC_TERM_FEATURE_NAMES). "
             "Required if 'doc_feats' is in the resolved --features - no "
             "live fallback. `score` itself always comes from the run file, "
             "never cached.",
    )
    parser.add_argument("--model_path", default="model_best.pt")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS,
        help="Must match the hidden layer sizes used during training.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--score_scale", type=float, default=None)
    parser.add_argument(
        "--score_norm",
        default="global",
        choices=["global", "per_query", "per_ranker"],
        help="Must match the value used during training.",
    )
    parser.add_argument(
        "--no_embedding_reduction", action="store_true",
        help="Must match the value used during training - changes the "
             "model's architecture (embedding_proj presence and MLP input "
             "width), so a mismatch will fail to load the checkpoint.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ndcg_cut.100"],
        help="Metrics to report correlations for (pytrec_eval-style dotted "
             "names, e.g. ndcg_cut.100 ndcg_cut.10 map_cut.50) - each must be "
             "a column (underscore form) in --metrics_csv.",
    )
    parser.add_argument(
        "--rbo_p",
        type=float,
        default=0.8,
        help="Persistence parameter for the Per-Query macro-average RBO "
             "(Rank-Biased Overlap) - higher values weight agreement further "
             "down each query's ranker ranking more heavily.",
    )
    parser.add_argument(
        "--ranker_map",
        default=None,
        help="Path to the ranker->id json saved during training. Required for "
             "per_ranker so eval rankers map to the same ids. Defaults to "
             "'<model_path>.rankers.json' if that file exists.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path for per-(qid, ranker) predicted probabilities, with "
             "model/epoch columns parsed from --model_path. Rows are appended, "
             "so runs against different checkpoints accumulate in one file. "
             "Defaults to 'predictions.csv' next to --model_path (i.e. the "
             "same run directory as the checkpoint). Pass an empty string to "
             "skip writing.",
    )
    parser.add_argument(
        "--split_path",
        default=None,
        help="Path to the dev/test split json saved during training. Restricts "
             "evaluation to the held-out test topics. Defaults to "
             "'<model_path>.split.json' if that file exists.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Which feature blocks the checkpoint was trained with. Auto-detect the
    # sidecar next to the checkpoint if --features wasn't explicitly given
    # (mirrors the --ranker_map/--split_path auto-detect pattern below).
    default_features_path = args.model_path + ".features.json"
    if args.features is not None:
        features = args.features
    elif os.path.exists(default_features_path):
        with open(default_features_path) as f:
            features = json.load(f)
        print(f"Loaded feature blocks ({features}) from {default_features_path}")
    else:
        features = list(ALL_FEATURE_BLOCKS)
        print("WARNING: no features sidecar found; assuming all six feature "
              "blocks (lexical+embedding+query_type+entity_count+scs_pmi+"
              "doc_feats). Pass --features explicitly "
              "if training used a different subset.")
    try:
        validate_feature_blocks(features)
    except ValueError as e:
        parser.error(str(e))
    if "lexical" in features and args.lexical_cache is None:
        parser.error("--lexical_cache is required when 'lexical' is in the resolved --features")
    if "embedding" in features and args.embedding_cache is None:
        parser.error("--embedding_cache is required when 'embedding' is in the resolved --features")
    if "query_type" in features and args.query_type_cache is None:
        parser.error("--query_type_cache is required when 'query_type' is in the resolved --features")
    if "entity_count" in features and args.entity_count_cache is None:
        parser.error("--entity_count_cache is required when 'entity_count' is in the resolved --features")
    if "scs_pmi" in features and args.scs_pmi_cache is None:
        parser.error("--scs_pmi_cache is required when 'scs_pmi' is in the resolved --features")
    if "doc_feats" in features and args.doc_feature_cache is None:
        parser.error("--doc_feature_cache is required when 'doc_feats' is in the resolved --features")

    # Lucene is never touched here: index_stats always stays None, and
    # require_caches=True below forbids QPPDataset from falling back to it
    # even if it weren't - every selected block must come from a cache.
    index_stats = None
    lexical_cache = load_feature_cache(args.lexical_cache) if "lexical" in features else None
    embedding_cache = load_feature_cache(args.embedding_cache) if "embedding" in features else None
    query_type_cache = load_feature_cache(args.query_type_cache) if "query_type" in features else None
    entity_count_cache = load_feature_cache(args.entity_count_cache) if "entity_count" in features else None
    scs_pmi_cache = load_feature_cache(args.scs_pmi_cache) if "scs_pmi" in features else None
    doc_feature_cache = load_feature_cache(args.doc_feature_cache) if "doc_feats" in features else None
    queries = load_queries(args.queries)

    # Restrict to the held-out test topics saved during training. Auto-detect
    # the sidecar next to the checkpoint if not explicitly provided.
    default_split = args.model_path + ".split.json"
    split_path = args.split_path or (default_split if os.path.exists(default_split) else None)
    if split_path:
        with open(split_path) as f:
            test_qids = set(json.load(f))
        queries = {q: t for q, t in queries.items() if q in test_qids}
        print(f"Loaded test split ({len(queries)} topics) from {split_path}")
    else:
        print("WARNING: no dev/test split file found; evaluating on all topics in "
              "--queries (this may include topics used for checkpoint selection "
              "during training).")

    # Load the training ranker->id map so per-ranker stats/columns line up.
    # Auto-detect the sidecar next to the checkpoint if not explicitly provided.
    ranker_to_id = None
    default_map = args.model_path + ".rankers.json"
    map_path = args.ranker_map or (default_map if os.path.exists(default_map) else None)
    if map_path:
        with open(map_path) as f:
            ranker_to_id = json.load(f)
        print(f"Loaded ranker map ({len(ranker_to_id)} rankers) from {map_path}")
    else:
        print("WARNING: no ranker map found; falling back to sorted run names. "
              "Ranker columns may not match training if the run-file names "
              "differ, which would misalign both per_ranker stats and the "
              "fixed-size classification head.")

    dataset = QPPDataset(
        args.run, args.qrels, queries, index_stats,
        args.top_k, args.score_scale, args.score_norm,
        ranker_to_id=ranker_to_id,
        metrics_csv=args.metrics_csv,
        lexical_cache=lexical_cache,
        embedding_cache=embedding_cache,
        query_type_cache=query_type_cache,
        entity_count_cache=entity_count_cache,
        scs_pmi_cache=scs_pmi_cache,
        doc_feature_cache=doc_feature_cache,
        require_caches=True,
        feature_blocks=features,
    )

    model = QPPMLP(
        doc_feature_dim=dataset.doc_feature_dim,
        list_feature_dim=dataset.list_feature_dim,
        query_feature_dim=dataset.query_feature_dim,
        embedding_slice=dataset.embedding_slice,
        top_k=args.top_k,
        hidden_dims=args.hidden_dims,
        score_norm=args.score_norm,
        num_rankers=dataset.num_rankers,
        reduce_embedding=not args.no_embedding_reduction,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    id_to_ranker = {v: k for k, v in dataset.ranker_to_id.items()}
    qids = [s["qid"] for s in dataset.samples]

    probs, ranker_masks = predict_all(model, dataset, device, args.batch_size)

    if args.output != "":
        output_path = args.output or os.path.join(
            os.path.dirname(args.model_path) or ".", "predictions.csv"
        )
        save_predictions(output_path, args.model_path, qids, probs, ranker_masks, id_to_ranker)

    for metric in args.metrics:
        labels = compute_labels_matrix(qids, id_to_ranker, metric, args.metrics_csv)
        report(probs, labels, ranker_masks, id_to_ranker, metric, rbo_p=args.rbo_p)


if __name__ == "__main__":
    main()
