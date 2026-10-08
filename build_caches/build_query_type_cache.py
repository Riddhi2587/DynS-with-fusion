"""
Precompute and save a query_type feature cache (see feature_cache.py) for
every qid in a queries file, so QPPQueryOnlyDataset construction can look the
1-dim query_type flag up instead of re-running the classifier on every
dataset construction.

Unlike build_feature_cache.py (lexical) and build_embedding_cache.py, this
needs no run files - query_type is a pure function of the raw query text, so
every qid in --queries gets classified directly, no Lucene index and no
query-embeddings pkl needed either.

Example:
    python -m build_caches.build_query_type_cache \
        --queries data/dl19-queries.tsv \
        --output dl19_querytype_features.pkl
"""

import argparse
import os

from build_caches.feature_cache import build_query_type_cache, save_feature_cache
from features import QueryTypeClassifier
from train import load_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query_type_classifier = QueryTypeClassifier()

    queries = load_queries(args.queries)
    print(f"Building query_type feature cache for {len(queries)} queries...")

    cache = build_query_type_cache(queries, query_type_classifier)
    print(f"  {cache.meta['num_qids']} qids")

    save_feature_cache(cache, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved -> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
