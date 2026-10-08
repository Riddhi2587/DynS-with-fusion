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

The 8 outputs are one score per ranker (one per run file).

## Inputs

| Input | Train flag | Eval flag | Format |
|---|---|---|---|
| Run files | `--train_run` | `--run` | One TREC-format file per ranker |
| Qrels | `--train_qrels` | `--qrels` | Standard TREC qrels |
| Queries | `--train_queries` | `--queries` | TSV, `qid<TAB>query text` |
| Metrics CSV | `--train_metrics_csv` | `--metrics_csv` | Columns `qid`, `ranker`, plus one column per metric, e.g. `ndcg_cut_100` |
| Lucene index | `--index` | `--index` | Pyserini index, used for lexical features |
| Query embeddings | `--query_embeddings` | `--query_embeddings` | Pickle of `{qid: vector}` |

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

A checkpoint is saved after every epoch as `model_query_only_best_epoch<N>.pt`,
with a `.rankers.json` file beside it.

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

If you changed `--hidden_dims` or used `--no_embedding_reduction` in training,
pass the same flags here.
