"""
Standalone port of best_ranker_per_query.ipynb's compare_predicted_vs_actual,
for running directly against predictions/metrics CSVs already on this machine
(no need to copy files back to a local notebook).

Example:
    python best_ranker_summary.py \
        --predictions ~/Documents/Payel/RPP-final-results/mlp-with-feats/llm-judged-train/lex-emb-qt/predictions_dl19_epoch40.csv \
        --metrics ~/Documents/Payel/precise-qpp-all-req-files/relevance-eval/dl19_metrics.csv \
        --label "lex-emb-qt epoch40 vs dl19 actual best"
"""

import argparse
import pandas as pd


def best_ranker_per_qid(metrics_df, metric_col="ndcg_cut_100"):
    winner_idx = metrics_df.groupby("qid")[metric_col].idxmax()
    return metrics_df.loc[winner_idx].set_index("qid")["ranker"]


def compare_predicted_vs_actual(predictions_path, metrics_path, label, metric_col="ndcg_cut_100"):
    predictions_df = pd.read_csv(predictions_path, dtype={"qid": str})
    metrics_df = pd.read_csv(metrics_path, dtype={"qid": str})

    predicted_best = best_ranker_per_qid(predictions_df, metric_col="score")
    actual_best = best_ranker_per_qid(metrics_df, metric_col=metric_col)

    comparison = pd.DataFrame({"predicted_best": predicted_best, "actual_best": actual_best})
    correct = comparison["predicted_best"] == comparison["actual_best"]

    all_rankers = sorted(metrics_df["ranker"].unique())
    summary = pd.DataFrame({
        "predicted_best_count": comparison["predicted_best"].value_counts().reindex(all_rankers, fill_value=0),
        "actual_best_count": comparison["actual_best"].value_counts().reindex(all_rankers, fill_value=0),
        "correct_count": comparison.loc[correct, "predicted_best"].value_counts().reindex(all_rankers, fill_value=0),
    })
    summary.index.name = "ranker"

    print(f"\n{label} (total queries: {len(comparison)})")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--metric_col", default="ndcg_cut_100")
    args = parser.parse_args()

    label = args.label or f"{args.predictions} vs {args.metrics}"
    summary = compare_predicted_vs_actual(args.predictions, args.metrics, label, args.metric_col)
    print(summary.to_string())


if __name__ == "__main__":
    main()
