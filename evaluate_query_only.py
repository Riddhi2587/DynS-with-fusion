"""
Evaluate a trained QueryOnlyMLP: the query-features-only ablation of
QPPMLP (see model.py, train_query_only.py). Same reporting as evaluate.py
- classification accuracy (does the predicted top ranker match the true
best ranker?) plus correlation between predicted probability and the true
metric score - so its output is directly comparable to evaluate.py's for
the full model.

Example:
    python evaluate_query_only.py \
        --index /path/to/msmarco-passage-index \
        --run data/dl20_runs.txt \
        --qrels data/dl20-passage.qrels \
        --queries data/dl20-queries.tsv \
        --model_path model_query_only_best.pt
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import QPPQueryOnlyDataset
from evaluate import compute_labels_matrix, report, save_predictions
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
             "build_metrics_csv.py and guide_docs/PRECOMPUTED_METRICS_GUIDE.md). "
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
    # ranker_mask); synthesize an all-True mask to reuse evaluate.py's
    # reporting helpers, which are shared with the full (masked) pipeline.
    ranker_masks = np.ones((len(qids), dataset.num_rankers), dtype=bool)

    if args.output != "":
        save_predictions(args.output, args.model_path, qids, probs, ranker_masks, id_to_ranker)

    for metric in args.metrics:
        labels = compute_labels_matrix(qids, id_to_ranker, metric, args.metrics_csv)
        report(probs, labels, ranker_masks, id_to_ranker, metric)


if __name__ == "__main__":
    main()
