"""
Precompute and save an embedding feature cache (see feature_cache.py) for
every qid in a queries file, so QPPQueryOnlyDataset construction can look the
raw, L2-normalized query representation up instead of recomputing it every
time. The raw width (768 for BERT/Contriever, 384 for MiniLM, etc.) is
inferred from the embedding source, not fixed - see
feature_cache.load_query_embeddings / build_embedding_cache.

Unlike build_feature_cache.py (lexical), this needs no run files or Lucene
index - build_embedding_feature is pure lookup + L2-normalize against the
precomputed query-embeddings pkl, so every qid in --queries gets one
directly.

Example:
    python -m build_caches.build_embedding_cache \
        --queries data/dl19-queries.tsv \
        --query_embeddings ../data/cache/bert-query-embeddings/cls/dl19.cls.pkl \
        --output dl19_embedding_features.pkl
"""

import argparse
import os

from build_caches.feature_cache import build_embedding_cache, load_query_embeddings, save_feature_cache
from train import load_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument(
        "--query_embeddings", required=True,
        help="Path to this dataset's precomputed raw query-embeddings pkl "
             "({qid: vector}, see e.g. data/cache/bert-query-embeddings/cls/*.cls.pkl). "
             "Width is inferred from the file, not fixed.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading query embeddings...")
    embedding_lookup = load_query_embeddings(args.query_embeddings)

    queries = load_queries(args.queries)
    print(f"Building embedding feature cache for {len(queries)} queries...")

    cache = build_embedding_cache(queries, embedding_lookup)
    print(f"  {cache.meta['num_qids']} qids")

    save_feature_cache(cache, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved -> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
