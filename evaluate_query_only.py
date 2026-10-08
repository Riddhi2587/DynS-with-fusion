"""
Evaluate a trained QueryOnlyMLP: the query-features-only ablation of
QPPMLP (see model.py, train.py). Reports classification accuracy (does
the predicted top ranker match the true best ranker?) plus correlation
between predicted probability and the true metric score. The reporting
code is the same as evaluate.py's, so its output is directly comparable
to evaluate.py's for the full model.

Example:
    python evaluate_query_only.py \
        --index /path/to/msmarco-passage-index \
        --run data/dl20_runs.txt \
        --qrels data/dl20-passage.qrels \
        --queries data/dl20-queries.tsv \
        --model_path model_query_only_best.pt
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, rankdata
from torch.utils.data import DataLoader

from dataset import QPPQueryOnlyDataset, _require_metric_column, load_precomputed_metrics
from feature_cache import load_feature_cache, load_query_embeddings
from features import IndexStats, QueryTypeClassifier
from model import DEFAULT_HIDDEN_DIMS, QueryOnlyMLP
from train import load_queries


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=64):
    """Returns probs, shape (N_queries, num_rankers)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    for query_feats, _labels in loader:
        logits = model(query_feats.to(device))
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


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
    fallback - supply a precomputed metrics CSV.
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
    accumulate in the same file across separate evaluate_query_only.py runs."""
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
    parser.add_argument(
        "--index", default=None,
        help="Path to the Lucene index. Required as the --lexical_cache miss "
             "fallback; omit if --lexical_cache fully covers the eval set's "
             "queries.",
    )
    parser.add_argument(
        "--query_embeddings", default=None,
        help="Path to the eval dataset's precomputed raw query-embeddings pkl "
             "({qid: vector}, see e.g. "
             "data/cache/bert-query-embeddings/cls/*.cls.pkl - width is "
             "inferred, not fixed; must match the width used at training "
             "time). Required as the --embedding_cache miss fallback; omit "
             "if --embedding_cache fully covers the eval set's queries.",
    )
    parser.add_argument("--run", required=True, nargs="+")
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model_path", default="model_query_only_best.pt")
    parser.add_argument(
        "--metrics_csv", required=True,
        help="Path to a precomputed per-(ranker, qid) metrics CSV (see "
             "guide_docs/PRECOMPUTED_METRICS_GUIDE.md). "
             "Required - there is no pytrec_eval fallback.",
    )
    parser.add_argument(
        "--lexical_cache", default=None,
        help="Path to a precomputed lexical/IDF feature cache (see "
             "build_feature_cache.py) - the 5-dim lexical query features are "
             "looked up from it instead of being recomputed via Lucene.",
    )
    parser.add_argument(
        "--embedding_cache", default=None,
        help="Path to a precomputed raw embedding feature cache (see "
             "build_embedding_cache.py) - the raw query representation "
             "(variable width by source) is looked up from it instead of "
             "being recomputed from --query_embeddings.",
    )
    parser.add_argument(
        "--query_type_cache", default=None,
        help="Path to a precomputed query_type feature cache (see "
             "build_query_type_cache.py) - the 1-dim query_type flag is "
             "looked up from it instead of being recomputed by the "
             "classifier.",
    )
    parser.add_argument(
        "--hidden_dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS,
        help="Must match the hidden layer sizes used during training.",
    )
    parser.add_argument(
        "--no_embedding_reduction", action="store_true",
        help="Must match the value used during training - changes the "
             "model's architecture (embedding_proj presence and MLP input "
             "width), so a mismatch will fail to load the checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ndcg_cut.100"],
        help="Metrics to report correlations for (pytrec_eval-style dotted "
             "names, e.g. ndcg_cut.100 ndcg_cut.10 map_cut.50) - each must be "
             "a column (underscore form) in --metrics_csv.",
    )
    parser.add_argument(
        "--ranker_map",
        default=None,
        help="Path to the ranker->id json saved during training. Defaults to "
             "'<model_path>.rankers.json' if that file exists.",
    )
    parser.add_argument(
        "--output",
        default="predictions_query_only.csv",
        help="CSV path for per-(qid, ranker) predicted probabilities, with "
             "model/epoch columns parsed from --model_path. Rows are appended, "
             "so runs against different checkpoints accumulate in one file. "
             "Pass an empty string to skip writing.",
    )
    parser.add_argument(
        "--split_path",
        default=None,
        help="Path to the dev/test split json saved during training. Restricts "
             "evaluation to the held-out test topics. Defaults to "
             "'<model_path>.split.json' if that file exists.",
    )
    args = parser.parse_args()

    if args.index is None and args.lexical_cache is None:
        parser.error("--index or --lexical_cache (or both) must be provided")
    if args.query_embeddings is None and args.embedding_cache is None:
        parser.error("--query_embeddings or --embedding_cache (or both) must be provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.index:
        index_stats = IndexStats(args.index)
    else:
        index_stats = None
        print("No --index given; relying entirely on --lexical_cache "
              "(a cache miss will raise).")

    if args.query_embeddings:
        embedding_lookup = load_query_embeddings(args.query_embeddings)
    else:
        embedding_lookup = None
        print("No --query_embeddings given; relying entirely on "
              "--embedding_cache (a cache miss will raise).")

    # Cheap/lazy to construct (the HF pipeline only loads on first .classify()
    # call), so always available as the --query_type_cache miss fallback.
    query_type_classifier = QueryTypeClassifier()

    lexical_cache = load_feature_cache(args.lexical_cache) if args.lexical_cache else None
    embedding_cache = load_feature_cache(args.embedding_cache) if args.embedding_cache else None
    query_type_cache = load_feature_cache(args.query_type_cache) if args.query_type_cache else None

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

    # Load the training ranker->id map so ranker columns line up. Auto-detect
    # the sidecar next to the checkpoint if not explicitly provided.
    ranker_to_id = None
    default_map = args.model_path + ".rankers.json"
    map_path = args.ranker_map or (default_map if os.path.exists(default_map) else None)
    if map_path:
        with open(map_path) as f:
            ranker_to_id = json.load(f)
        print(f"Loaded ranker map ({len(ranker_to_id)} rankers) from {map_path}")

    dataset = QPPQueryOnlyDataset(
        args.run, args.qrels, queries, index_stats,
        ranker_to_id=ranker_to_id,
        lexical_cache=lexical_cache,
        embedding_cache=embedding_cache,
        query_type_cache=query_type_cache,
        embedding_lookup=embedding_lookup,
        query_type_classifier=query_type_classifier,
        metrics_csv=args.metrics_csv,
    )

    model = QueryOnlyMLP(
        query_feature_dim=dataset.query_feature_dim,
        embedding_slice=dataset.embedding_slice,
        hidden_dims=args.hidden_dims,
        num_rankers=dataset.num_rankers,
        reduce_embedding=not args.no_embedding_reduction,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    id_to_ranker = {v: k for k, v in dataset.ranker_to_id.items()}
    qids = [s["qid"] for s in dataset.samples]

    probs = predict_all(model, dataset, device, args.batch_size)
    # Every ranker is always present in this data (QPPQueryOnlyDataset has no
    # ranker_mask); synthesize an all-True mask for the reporting helpers
    # above, which also handle masked (partial-ranker) data.
    ranker_masks = np.ones((len(qids), dataset.num_rankers), dtype=bool)

    if args.output != "":
        save_predictions(args.output, args.model_path, qids, probs, ranker_masks, id_to_ranker)

    for metric in args.metrics:
        labels = compute_labels_matrix(qids, id_to_ranker, metric, args.metrics_csv)
        report(probs, labels, ranker_masks, id_to_ranker, metric)


if __name__ == "__main__":
    main()
