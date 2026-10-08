from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from features import EMBEDDING_DIM

DEFAULT_HIDDEN_DIMS = [2048, 1024]


class QueryOnlyMLP(nn.Module):
    """
    Feedforward classification head for QPP, from query features alone.

    Input is just the query vector (shared across all rankers for a given
    query, so it is passed through the MLP once, not once per ranker);
    output is one logit per ranker —
    ranker identity is encoded purely by output position (a separate output
    unit per ranker), not by any per-ranker input feature.

    Standardization statistics (of query features only) are stored as a
    registered buffer, so they are saved/restored with the model's
    state_dict. Call `fit_standardization` once on the TRAINING data before
    training.

    If `embedding_slice` is given, it locates the raw (variable-width, e.g.
    768/384) query-representation block within `query_feats` - the caller
    passes it explicitly (normally read off the paired dataset instance,
    e.g. `dataset.embedding_slice`), since its offset/width depend on the
    embedding source and which other query blocks are selected; there is no
    auto-detection from `query_feature_dim` alone. That raw slice is reduced
    to EMBEDDING_DIM dims by a trained `nn.Linear` (`self.embedding_proj`),
    spliced back into the same position before the MLP (see
    `_project_embedding`) - unless `reduce_embedding` is False, in which case
    `embedding_proj` is never built and the raw slice is passed through to
    the MLP at its full original width instead. Whether the RAW slice is
    z-scored before that projection is controlled by `standardize_embedding`
    (default False, matching the original PCA-embedding behavior of never
    standardizing it) - the PROJECTED output is never standardized either
    way, since it's a trained internal representation, not a raw input
    statistic.
    """

    def __init__(
        self,
        query_feature_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        num_rankers: int = 1,
        embedding_slice: Optional[slice] = None,
        standardize_embedding: bool = False,
        reduce_embedding: bool = True,
    ):
        super().__init__()
        self.num_rankers = max(int(num_rankers), 1)
        self.query_feature_dim = query_feature_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims else list(DEFAULT_HIDDEN_DIMS)

        self.register_buffer("query_mean", torch.zeros(query_feature_dim))
        self.register_buffer("query_std", torch.ones(query_feature_dim))

        self.embedding_slice = embedding_slice
        self.standardize_embedding = standardize_embedding
        self.reduce_embedding = reduce_embedding
        if embedding_slice is not None:
            raw_dim = embedding_slice.stop - embedding_slice.start
            if reduce_embedding:
                self.embedding_proj = nn.Linear(raw_dim, EMBEDDING_DIM)
                proj_dim = EMBEDDING_DIM
            else:
                self.embedding_proj = None
                proj_dim = raw_dim
        else:
            raw_dim = 0
            proj_dim = 0
            self.embedding_proj = None

        in_dim = query_feature_dim - raw_dim + proj_dim
        layers = []
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, self.num_rankers)]
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_standardization(self, loader: DataLoader, eps: float = 1e-6) -> None:
        """
        Estimate query-feature standardization statistics from TRAINING data
        and store them in a buffer. Call once, before training, on a loader
        over the training set.

        The embedding columns (self.embedding_slice, if set) are excluded
        UNLESS self.standardize_embedding is True: by default their mean/std
        are forced to identity (0, 1) regardless of the fitted statistics,
        since they're L2-normalized per query at feature-construction time;
        setting standardize_embedding=True instead lets them get real
        computed statistics like every other query feature (see class
        docstring).
        """
        device = self.query_mean.device
        q_sum = torch.zeros_like(self.query_mean)
        q_sqsum = torch.zeros_like(self.query_mean)
        q_count = 0

        for query_feats, _labels in loader:
            query_feats = query_feats.to(device)
            q_sum += query_feats.sum(dim=0)
            q_sqsum += (query_feats * query_feats).sum(dim=0)
            q_count += query_feats.shape[0]

        q_mean = q_sum / max(q_count, 1)
        q_var = (q_sqsum / max(q_count, 1)) - q_mean**2
        q_std = q_var.clamp_min(0).sqrt() + eps

        if self.embedding_slice is not None and not self.standardize_embedding:
            q_mean[self.embedding_slice] = 0.0
            q_std[self.embedding_slice] = 1.0

        self.query_mean.copy_(q_mean)
        self.query_std.copy_(q_std)

    def _project_embedding(self, query_feats: Tensor) -> Tensor:
        """Replace the raw embedding_slice columns of query_feats with their
        EMBEDDING_DIM-dim projection through self.embedding_proj, leaving
        every other column untouched. No-op if embedding_slice is None."""
        if self.embedding_proj is None:
            return query_feats
        s = self.embedding_slice
        before, raw, after = query_feats[:, :s.start], query_feats[:, s], query_feats[:, s.stop:]
        return torch.cat([before, self.embedding_proj(raw), after], dim=1)

    def forward(self, query_feats: Tensor) -> Tensor:
        """
        Args:
            query_feats: (B, query_feature_dim)

        Returns:
            (B, num_rankers) per-ranker logits. softmax(dim=-1) gives a
            distribution over "which ranker is best for this query". No
            masking - every ranker is always present in this data.
        """
        query_feats = (query_feats - self.query_mean) / self.query_std
        query_feats = self._project_embedding(query_feats)
        return self.mlp(query_feats)
