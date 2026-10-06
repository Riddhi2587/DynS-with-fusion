"""
Pivots a long-format fusion_table.csv (one row per (label, dataset), as
written by build_fusion_row.py) into a wide table: one row per label, with
dl19's and dl20's columns aligned side by side for direct comparison.

Column order within each dataset matches build_fusion_row.py's row order:
query_macro_kendall (nDCG@10, MAP@100), then argmax (nDCG@10, MAP@100),
then CombMNZ (nDCG@10, MAP@100) - preserved here via the input CSV's own
column order, not re-sorted.

If a (label, dataset) pair appears more than once in the input (e.g. the
fusion-row script was re-run and appended again), the last occurrence wins -
same keep-last convention used elsewhere in this repo for accumulating CSVs.

Usage:
    python align_fusion_table.py \
        --input fusion_table.csv \
        --output fusion_table_aligned.csv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fusion_table.csv")
    parser.add_argument("--output", default="fusion_table_aligned.csv")
    parser.add_argument(
        "--datasets", nargs="+", default=["dl19", "dl20"],
        help="Dataset names to align as column groups, in order.",
    )
    args = parser.parse_args()
    DATASETS = args.datasets

    df = pd.read_csv(args.input)
    metric_cols = [c for c in df.columns if c not in ("label", "dataset")]

    df = df.drop_duplicates(subset=["label", "dataset"], keep="last")
    labels = list(dict.fromkeys(df["label"]))  # preserve first-seen order

    rows = []
    for label in labels:
        row = {"label": label}
        sub = df[df["label"] == label].set_index("dataset")
        for dataset in DATASETS:
            for col in metric_cols:
                key = f"{dataset}_{col}"
                row[key] = sub.loc[dataset, col] if dataset in sub.index else float("nan")
        rows.append(row)

    out_df = pd.DataFrame(rows).set_index("label")
    out_df.to_csv(args.output)
    print(f"Wrote aligned fusion table -> {args.output} ({len(out_df)} row(s), datasets={DATASETS})")


if __name__ == "__main__":
    main()
