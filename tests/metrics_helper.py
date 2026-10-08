"""Test-only helper: writes a per-(qid, ranker) metrics CSV from run files and
qrels via pytrec_eval, for fixtures that need a precomputed metrics CSV."""

import csv

import pytrec_eval

from dataset import load_qrels, load_run

# metric -> relevance_level (graded metrics ignore it; binary ones use grade>=2).
METRICS = {
    "ndcg_cut.10": 1,
    "ndcg_cut.100": 1,
    "map_cut.50": 2,
    "map_cut.100": 2,
}


def write_metrics_csv(run_files, qrels_path, output_csv, metrics=METRICS):
    runs = load_run(run_files)
    qrels = load_qrels(qrels_path)

    rows = {}  # (qid, ranker) -> {metric_key: value}
    for metric, rel_level in metrics.items():
        key = metric.replace(".", "_")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric}, relevance_level=rel_level)
        for ranker, run_data in runs.items():
            run_for_eval = {qid: dict(doc_list) for qid, doc_list in run_data.items()}
            for qid, scores in evaluator.evaluate(run_for_eval).items():
                rows.setdefault((qid, ranker), {})[key] = scores.get(key, 0.0)

    metric_keys = [m.replace(".", "_") for m in metrics]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "ranker"] + metric_keys)
        writer.writeheader()
        for (qid, ranker), vals in sorted(rows.items()):
            writer.writerow({"qid": qid, "ranker": ranker, **vals})
