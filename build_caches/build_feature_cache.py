"""
Precompute and save a lexical/IDF feature cache (see feature_cache.py) for
one dataset (a fixed set of run files + a queries file), so
QPPQueryOnlyDataset construction can look the 5-dim lexical query features up
instead of recomputing them via Lucene every time.

This is the lexical-only cache - see build_embedding_cache.py and
build_query_type_cache.py for the other two (independent) pre-retrieval
feature caches QPPQueryOnlyDataset merges at construction time.

Example:
    python -m build_caches.build_feature_cache \
        --index /path/to/msmarco-passage-index \
        --run data/dl19_runs/*.res \
        --queries data/dl19-queries.tsv \
        --output dl19_lexical_features.pkl
"""

import argparse
import os

from build_caches.feature_cache import build_feature_cache, save_feature_cache
from features import IndexStats
from train import load_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--run", required=True, nargs="+")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading index...")
    index_stats = IndexStats(args.index)

    queries = load_queries(args.queries)
    print(f"Building lexical feature cache from {len(args.run)} run file(s), {len(queries)} queries...")

    cache = build_feature_cache(args.run, queries, index_stats)
    print(f"  {cache.meta['num_qids']} qids")

    save_feature_cache(cache, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved -> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
