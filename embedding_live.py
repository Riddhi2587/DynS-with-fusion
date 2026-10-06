"""
Live model wrappers for the runtime-measurement scripts: the all-MiniLM-L6-v2
query encoder and a GPU-capable QueryTypeClassifier factory.

MiniLMEncoder is ported from the encoding step of
Plan1-query-embedding-classifier-RQ3/Plan1/build_minilm_embedding_cache.py
(same model, same un-normalized output). L2-normalization is NOT done here -
features.build_embedding_feature applies it at feature-construction time,
exactly as in the cached pipeline.
"""

from typing import Optional

import numpy as np
import torch

from features import QueryTypeClassifier

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MiniLMEncoder:
    """Loads eagerly on load() (so model-load time can be timed separately from
    encode time). `model` is an injection point for tests: any object with a
    sentence-transformers-style encode(texts, ...) -> np.ndarray."""

    def __init__(self, model_name: str = MODEL_NAME, device=None, model=None):
        self.model_name = model_name
        self.device = device if device is not None else default_device()
        self.model = model

    def load(self) -> "MiniLMEncoder":
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=str(self.device))
        return self

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        """(len(texts), dim) float32, un-normalized."""
        vecs = self.model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def make_query_type_classifier(device=None, pipeline_fn: Optional[object] = None) -> QueryTypeClassifier:
    """QueryTypeClassifier with its transformers pipeline loaded NOW (not lazily
    on first use) and placed on `device` (GPU if CUDA), so timed calls never
    include model loading. `pipeline_fn` is a test injection point."""
    if pipeline_fn is None:
        from transformers import pipeline

        device = device if device is not None else default_device()
        idx = -1
        if getattr(device, "type", str(device)).startswith("cuda"):
            idx = getattr(device, "index", None) or 0
        pipeline_fn = pipeline(
            "text-classification", model=QueryTypeClassifier.MODEL_NAME, device=idx
        )
    return QueryTypeClassifier(pipeline_fn=pipeline_fn)
