#!/usr/bin/env python
"""Argmax-QPP fusion metrics, adapted from pipeline/build_fusion_table.py
(classification-head-MLP branch) for mlp-with-feats's open-ended
feature-ablation runs.

The original script has a hardcoded branch/result_dir registry (one entry
per config-driven pipeline branch) and builds a whole multi-branch table in
one run. mlp-with-feats doesn't use that pipeline - each feature-ablation
combo lives in its own RPP-final-results/mlp-with-feats/llm-judged-train/<folder>/
directory, with predictions filenames that vary (hidden-dims tag, epoch,
etc. per your own naming), and there's no fixed list of "branches" - new
combos get added ad hoc. So instead of a big batch run, this computes ONE
(label, dataset) row per invocation and appends it to a shared fusion_table.csv,
matching how evaluate.py/best_ranker_summary.py are run here already - one
command per model/dataset/epoch, not a sweep.

The core metric logic (argmax-QPP) is copied near-verbatim from
build_fusion_table.py - same PYTREC_METRICS (ndcg_cut.10, map_cut.100).

Also adds query_macro_kendall_{nDCG@10,MAP@100} columns - the same
"Per-Query macro-average" Kendall's tau evaluate.py's report() prints
(predicted score vs. true metric value, correlated per query across that
query's rankers, then averaged over queries), recomputed here from the
saved predictions CSV. Needs scipy (already a dependency).

Also adds query_macro_rbo_{nDCG@10,MAP@100} (Rank-Biased Overlap, same
per-query-then-average convention as query_macro_kendall, but weighting
agreement among the top-predicted rankers more heavily - persistence
parameter via --rbo_p, default 0.8) and mrr_{nDCG@10,MAP@100} (Mean
Reciprocal Rank of the true best-labeled ranker within the predicted
ordering, per query, averaged over queries) columns.

CombMNZ fusion has been removed from this script - only argmax-QPP (the
"equivalent of the best ranker selection using predicted score") is kept, so
this script needs neither --run/--qrels nor pytrec_eval.

Usage:
    python build_fusion_row.py \
        --label lex-emb-qt-doc-hist-h1024-512-epoch40 \
        --predictions ~/Documents/Payel/RPP-final-results/mlp-with-feats/llm-judged-train/lex-emb-qt-doc-hist/predictions_dl19_h1024-512_epoch40.csv \
        --epoch 40 \
        --dataset dl19 \
        --metrics_csv ~/Documents/Payel/precise-qpp-all-req-files/relevance-eval/dl19_metrics.csv \
        --output fusion_table.csv

Run once per (label, dataset) pair - rows accumulate in --output across
calls, same append-don't-overwrite convention as evaluate.py's predictions CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import kendalltau, rankdata

# pytrec_eval metric name (dot form) -> (metrics_csv/result column key, relevance_level or None).
PYTREC_METRICS = {
    "ndcg_cut.10": ("ndcg_cut_10", None),
    "map_cut.100": ("map_cut_100", 2),
}
METRIC_LABELS = {"ndcg_cut_10": "nDCG@10", "map_cut_100": "MAP@100"}


def load_metrics_csv(path: str) -> dict[tuple[str, str], dict[str, float]]:
    """{(qid, ranker): {metric_key: value}}."""
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        metric_cols = [c for c in reader.fieldnames if c not in ("qid", "ranker")]
        for row in reader:
            lookup[(row["qid"], row["ranker"])] = {c: float(row[c]) for c in metric_cols}
    return lookup


def load_predictions(path: Path, epoch: int) -> dict[str, dict[str, float]]:
    """Reads a predictions CSV (model, epoch, qid, ranker, score), filters to
    `epoch`, dedupes keep-last per (model, epoch, qid, ranker) - same
    append-only-file dedup hazard as elsewhere in this repo.
    Returns {qid: {ranker: score}}."""
    if not path.exists():
        return {}
    rows: dict[tuple[str, str, str, str], float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["epoch"]) != epoch:
                continue
            key = (row["model"], row["epoch"], row["qid"], row["ranker"])
            rows[key] = float(row["score"])  # keep-last on duplicate key
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    for (_, _, qid, ranker), score in rows.items():
        predictions[qid][ranker] = score
    return dict(predictions)


def argmax_metrics(
    predictions: dict[str, dict[str, float]],
    metrics_lookup: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """Per query, pick the ranker with the max predicted score, look up that
    (qid, ranker)'s already-precomputed metric values, average over queries."""
    per_query: dict[str, list[float]] = {key: [] for key, _ in PYTREC_METRICS.values()}
    for qid, ranker_scores in predictions.items():
        if not ranker_scores:
            continue
        best_ranker = max(ranker_scores, key=ranker_scores.get)
        row = metrics_lookup.get((qid, best_ranker), {})
        for key, _ in PYTREC_METRICS.values():
            per_query[key].append(row.get(key, 0.0))
    return {key: (mean(vals) if vals else float("nan")) for key, vals in per_query.items()}


def query_macro_kendall(
    predictions: dict[str, dict[str, float]],
    metrics_lookup: dict[tuple[str, str], dict[str, float]],
    metric_key: str,
) -> float:
    """Per query: Kendall's tau between the model's predicted scores and the
    true metric_key values, across that query's rankers (>=2 required, else
    skipped) - averaged over queries. Same definition as evaluate.py's
    report()'s "Per-Query macro-average" Kendall, recomputed here from the
    saved predictions CSV (which already has every ranker's predicted score
    per query, not just the argmax winner) instead of live tensors."""
    per_query_taus: list[float] = []
    for qid, ranker_scores in predictions.items():
        if len(ranker_scores) < 2:
            continue
        preds, labels = [], []
        for ranker, score in ranker_scores.items():
            row = metrics_lookup.get((qid, ranker))
            if row is None or metric_key not in row:
                continue
            preds.append(score)
            labels.append(row[metric_key])
        if len(preds) < 2:
            continue
        tau, _ = kendalltau(preds, labels)
        if tau == tau:  # skip NaN (e.g. all-tied predictions or labels)
            per_query_taus.append(tau)
    return mean(per_query_taus) if per_query_taus else float("nan")


def query_macro_rbo(
    predictions: dict[str, dict[str, float]],
    metrics_lookup: dict[tuple[str, str], dict[str, float]],
    metric_key: str,
    p: float = 0.8,
) -> float:
    """Per query: Rank-Biased Overlap (persistence p) between the ranking
    induced by the model's predicted scores and the ranking induced by true
    metric_key values, across that query's rankers (>=2 required, else
    skipped) - averaged over queries. Same per-query-then-average convention
    as query_macro_kendall, but weights agreement among the top-ranked
    rankers more heavily. Both rankings are complete permutations of the same
    ranker set per query, so the exact closed form is used (RBO(p) = (1-p) *
    sum_{d=1}^k p^(d-1) * A_d + p^k), not the extrapolated estimate meant for
    indefinite/truncated rankings.

    Ties (equal predicted scores or equal metric_key values) are handled via
    competition ranking + symmetric overlap, same as evaluate.py's _rbo - see
    its docstring for the full explanation. This reduces to the plain
    |intersection| / d formula when there are no ties."""
    per_query_rbos: list[float] = []
    for qid, ranker_scores in predictions.items():
        if len(ranker_scores) < 2:
            continue
        pairs = []
        for ranker, score in ranker_scores.items():
            row = metrics_lookup.get((qid, ranker))
            if row is None or metric_key not in row:
                continue
            pairs.append((score, row[metric_key]))
        k = len(pairs)
        if k < 2:
            continue
        pred_rank = rankdata([-pair[0] for pair in pairs], method="min")
        true_rank = rankdata([-pair[1] for pair in pairs], method="min")
        total = 0.0
        for d in range(1, k + 1):
            seen_pred = {i for i in range(k) if pred_rank[i] <= d}
            seen_true = {i for i in range(k) if true_rank[i] <= d}
            agreement = 2 * len(seen_pred & seen_true) / (len(seen_pred) + len(seen_true))
            total += (p ** (d - 1)) * agreement
        per_query_rbos.append((1 - p) * total + p ** k)
    return mean(per_query_rbos) if per_query_rbos else float("nan")


def query_mrr(
    predictions: dict[str, dict[str, float]],
    metrics_lookup: dict[tuple[str, str], dict[str, float]],
    metric_key: str,
) -> float:
    """Per query: reciprocal rank of the true best-labeled ranker (by
    metric_key) within the ranking induced by the model's predicted scores,
    across that query's rankers (>=2 required, else skipped) - averaged over
    queries. Ties for true-best are resolved to the earliest (best)
    predicted rank among them, matching evaluate.py's top1_accuracy tie
    convention."""
    per_query_rr: list[float] = []
    for qid, ranker_scores in predictions.items():
        if len(ranker_scores) < 2:
            continue
        pairs = []
        for ranker, score in ranker_scores.items():
            row = metrics_lookup.get((qid, ranker))
            if row is None or metric_key not in row:
                continue
            pairs.append((ranker, score, row[metric_key]))
        if len(pairs) < 2:
            continue
        pred_rank = {
            ranker: r + 1
            for r, (ranker, _, _) in enumerate(sorted(pairs, key=lambda x: -x[1]))
        }
        best_val = max(v for _, _, v in pairs)
        best_ranks = [pred_rank[ranker] for ranker, _, v in pairs if v == best_val]
        per_query_rr.append(1.0 / min(best_ranks))
    return mean(per_query_rr) if per_query_rr else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Row label, e.g. lex-emb-qt-doc-hist-h1024-512-epoch40.")
    parser.add_argument("--dataset", required=True, help="Dataset name for the column group, e.g. dl19.")
    parser.add_argument("--predictions", required=True, help="Path to this run's predictions CSV.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to filter predictions to.")
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--output", default="fusion_table.csv")
    parser.add_argument(
        "--rbo_p", type=float, default=0.8,
        help="Persistence parameter for query_macro_rbo - higher values "
             "weight agreement further down each query's ranker ranking "
             "more heavily.",
    )
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions), args.epoch)
    if not predictions:
        print(f"No predictions found at {args.predictions} for epoch {args.epoch} - nothing to write.")
        return 1

    metrics_lookup = load_metrics_csv(args.metrics_csv)
    argmax_result = argmax_metrics(predictions, metrics_lookup)

    row = {"label": args.label, "dataset": args.dataset}
    for metric_key, _ in PYTREC_METRICS.values():
        row[f"query_macro_kendall_{METRIC_LABELS[metric_key]}"] = query_macro_kendall(
            predictions, metrics_lookup, metric_key
        )
        row[f"query_macro_rbo_{METRIC_LABELS[metric_key]}"] = query_macro_rbo(
            predictions, metrics_lookup, metric_key, p=args.rbo_p
        )
        row[f"mrr_{METRIC_LABELS[metric_key]}"] = query_mrr(
            predictions, metrics_lookup, metric_key
        )
    for metric_key, val in argmax_result.items():
        row[f"argmax_{METRIC_LABELS[metric_key]}"] = val

    output_path = Path(args.output)
    write_header = not (output_path.exists() and output_path.stat().st_size > 0)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended row ({args.label}, {args.dataset}) -> {output_path}")
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
