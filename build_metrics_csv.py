"""
Generate a precomputed per-(qid, ranker) metrics CSV for one dataset (see
guide_docs/PRECOMPUTED_METRICS_GUIDE.md). Run once per dataset; rebuild only
if that dataset's run files or qrels change. This is the only script in the
repo that calls pytrec_eval - train.py/evaluate.py and their query-only
counterparts consume the CSV this produces and never call pytrec_eval
themselves.

Example:
    python build_metrics_csv.py \
        --run data/dl19_runs/*.res \
        --qrels data/dl19-passage.qrels \
        --out dl19_metrics.csv
"""

import argparse
import csv

import pytrec_eval

from dataset import load_qrels, load_run

# metric -> relevance_level, matching evaluate.py's compute_labels_matrix
# convention: graded metrics (ndcg) ignore relevance_level; binary metrics
# (map) use the TREC DL convention of grade>=2 counting as relevant.
METRICS = {
    "ndcg_cut.10": 1,
    "ndcg_cut.100": 1,
    "map_cut.50": 2,
    "map_cut.100": 2,
}


def build_metrics_csv(run_files, qrels_path, output_csv, metrics=METRICS):
    runs = load_run(run_files)
    qrels = load_qrels(qrels_path)

    rows = {}  # (qid, ranker) -> {metric_key: value}
    for metric, rel_level in metrics.items():
        key = metric.replace(".", "_")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric}, relevance_level=rel_level)
        for ranker, run_data in runs.items():
            run_for_eval = {qid: dict(doc_list) for qid, doc_list in run_data.items()}
            results = evaluator.evaluate(run_for_eval)
            for qid, scores in results.items():
                rows.setdefault((qid, ranker), {})[key] = scores.get(key, 0.0)

    metric_keys = [m.replace(".", "_") for m in metrics]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "ranker"] + metric_keys)
        writer.writeheader()
        for (qid, ranker), vals in sorted(rows.items()):
            writer.writerow({"qid": qid, "ranker": ranker, **vals})

    print(f"Wrote {len(rows)} rows ({len(runs)} rankers) to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, nargs="+", help="Run files for this dataset.")
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--out", required=True, help="Output metrics CSV path.")
    args = parser.parse_args()

    build_metrics_csv(args.run, args.qrels, args.out)


if __name__ == "__main__":
    main()
