# DynS

## Architecture

```
                 +--> embedding model (precomputed vectors) --+
                 |    e.g. BERT / Contriever / MiniLM         |
 query text -----+--> lexical / IDF stats (5-dim, Lucene) ----+--> concat
                 |                                            |
                 +--> query-type classifier (keyword / NL) ---+
                                                               |
                                                               v
                                                              MLP
                                                               |
                                                               v
                                                     8 outputs (one score per ranker)
```

The embedding model is not trained here; only its precomputed vectors are
read. The number of outputs equals the number of rankers (run files), 8 in
the reference setup. Softmax over the outputs gives a distribution over which
ranker is best for a query.

## Setup

```bash
pip install -r requirements.txt
```

- Pyserini needs a JDK and a Lucene index (e.g. MS MARCO passage) to compute
  lexical features. You can skip the index if a lexical cache covers every
  query (see [Feature caches](#feature-caches)).
- The first time the query-type feature is computed, the classifier
  `shahrukhx01/bert-mini-finetune-question-detection` is downloaded from
  Hugging Face.

## Inputs

| Input | Train flag | Eval flag | Format |
|---|---|---|---|
| Run files | `--train_run` | `--run` | One TREC-format file per ranker. The ranker name is the file stem, with year tokens stripped (`BM25.2019.100` -> `BM25.100`) so train and eval names match. |
| Qrels | `--train_qrels` | `--qrels` | Standard TREC qrels. |
| Queries | `--train_queries` | `--queries` | TSV, `qid<TAB>query text`. |
| Metrics CSV | `--train_metrics_csv` | `--metrics_csv` | Required (no pytrec_eval fallback). Columns `qid`, `ranker`, plus one column per metric in underscore form, e.g. `ndcg_cut_100` (the training label). A missing (ranker, qid) row counts as 0.0. |
| Lucene index | `--index` | `--index` | Only needed for lexical features not covered by a cache. |
| Query embeddings | `--query_embeddings` | `--query_embeddings` | Pickle of `{qid: vector}`. The width is inferred (768, 384, ...), and it must match between train and eval. Only needed for embeddings not covered by a cache. |

At least one of `--index` / `--lexical_cache` and one of `--query_embeddings`
/ `--embedding_cache` must be given.

## Feature caches

Caches are optional. There is one per feature, and each can be rebuilt on its
own:

| Feature | Cache | Needs |
|---|---|---|
| lexical (5-dim IDF stats) | `build_caches.build_feature_cache` | Lucene index, run files, queries |
| embedding (raw, L2-normalized) | `build_caches.build_embedding_cache` | queries, query embeddings pkl |
| query_type (1-dim flag) | `build_caches.build_query_type_cache` | queries |

```bash
python -m build_caches.build_feature_cache \
    --index /path/to/index --run data/dl19_runs/*.res \
    --queries data/dl19-queries.tsv --output dl19_lexical.pkl

python -m build_caches.build_embedding_cache \
    --queries data/dl19-queries.tsv \
    --query_embeddings dl19.cls.pkl --output dl19_embedding.pkl

python -m build_caches.build_query_type_cache \
    --queries data/dl19-queries.tsv --output dl19_query_type.pkl
```

Pass them to `train.py` / `evaluate.py` with `--lexical_cache`,
`--embedding_cache` and `--query_type_cache`. A query missing from a cache
falls back to live computation (`--index`, `--query_embeddings`, the
classifier). If no fallback was supplied, a clear error is raised.

## Training

```bash
python train.py \
    --index /path/to/index \
    --query_embeddings dl19.cls.pkl \
    --train_run data/dl19_runs/*.res \
    --train_qrels data/dl19-passage.qrels \
    --train_queries data/dl19-queries.tsv \
    --train_metrics_csv data/dl19_metrics.csv
```

| Flag | Default | Notes |
|---|---|---|
| `--hidden_dims` | `2048 1024` | MLP hidden layer sizes. |
| `--dropout` | `0.2` | |
| `--loss_type` | `cross_entropy` | `cross_entropy` (top-1 best-ranker classification) or `listmle` (listwise over all rankers' nDCG@100). |
| `--lr` | `1e-3` | |
| `--epochs` | `50` | |
| `--batch_size` | `8` | |
| `--seed` | `42` | |
| `--standardize_embedding` | off | Z-score the raw embedding before it is projected. |
| `--no_embedding_reduction` | off | Skip the learned projection to 32-d and feed the full embedding to the MLP. Pass the same flag at eval time. |
| `--save_path` | `model_query_only_best.pt` | |
| `--run_dir` | none | Puts the checkpoints, results and feature report in one folder, and appends to `runs_index_query_only.csv`. |

Outputs:
- A checkpoint after every epoch: `<save_path stem>_epoch<N>.pt`. There is no
  dev-set checkpoint selection, so evaluate each checkpoint you care about.
- `<checkpoint>.rankers.json` beside each checkpoint, holding the ranker -> id
  map that evaluation reuses.
- A feature sanity report (JSON) at `--feature_report_path`, and a results
  JSON at `--results_path`.

## Evaluation

```bash
python evaluate.py \
    --index /path/to/index \
    --query_embeddings dl20.cls.pkl \
    --run data/dl20_runs/*.res \
    --qrels data/dl20-passage.qrels \
    --queries data/dl20-queries.tsv \
    --metrics_csv data/dl20_metrics.csv \
    --model_path model_query_only_best_epoch50.pt
```

- `--hidden_dims` and `--no_embedding_reduction` must match training, or the
  checkpoint will not load.
- The ranker map is read from `<model_path>.rankers.json` if it exists (or
  pass `--ranker_map`). If `<model_path>.split.json` exists, evaluation is
  restricted to those test topics (or pass `--split_path`).
- `--metrics` (default `ndcg_cut.100`) selects the metrics to report. Each
  must be a column in the metrics CSV.

It prints:
- Overall and per-ranker Pearson and Kendall correlations between predicted
  scores and the true metric.
- Per-query average Pearson, Kendall, RBO and MRR.
- Best-ranker classification: top-1 and top-k accuracy, plus per-ranker
  precision, recall and F1.
- Prediction entropy.

Per-(qid, ranker) predicted probabilities are appended to `--output`
(default `predictions_query_only.csv`). Pass `--output ""` to skip writing.
