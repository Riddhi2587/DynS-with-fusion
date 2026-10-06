from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from features import EMBEDDING_DIM, SCORE_FEATURE_INDICES

DEFAULT_HIDDEN_DIMS = [2048, 1024]


class QPPMLP(nn.Module):
    """
    Feedforward classification head for QPP.

    One "example" is a query together with all `num_rankers` rankers' ranked
    lists for it. For each ranker slot the pipeline is:
      0. Standardize raw doc / list / query features using train-set statistics.
      1. Flatten that ranker's top_k (padded) document feature vectors, in
         rank order, into a single (top_k * doc_feature_dim) vector, and
         prepend that ranker's list_feature_dim-dim LIST features (mean_score,
         var_score - see features.LIST_FEATURE_NAMES) to the START of it, so
         they appear ONCE per ranker rather than being duplicated into every
         one of that ranker's top_k document rows.
    The `num_rankers` resulting (list_feature_dim + top_k*doc_feature_dim)
    blocks are concatenated with each other, in fixed ranker-column order
    (see `ranker_to_id` in dataset.py), and with the query feature vector —
    which is shared across rankers and included ONCE, not duplicated per
    ranker. This gives one
    (num_rankers * (list_feature_dim + top_k * doc_feature_dim) + query_feature_dim)
    input vector per query, which is passed through a plain MLP (Linear +
    ReLU + Dropout blocks; `hidden_dims` sets the number and size of hidden
    layers) that outputs `num_rankers` logits at once, so the head can mix
    information across rankers. A softmax over the logits gives a
    distribution over "which ranker is best for this query", trained with
    cross-entropy against the ranker with the highest true metric score (see
    train.py). Rankers absent for a given query (ranker_mask False) have
    their logit forced to -inf so they get zero probability mass — note
    their (still MLP-computed) logit can still influence the *other*
    rankers' logits through the concatenation, since the head requires a
    fixed `num_rankers` matching the one fixed at construction time.

    Standardization statistics are stored as registered buffers, so they are
    saved and restored automatically with the model's state_dict. Call
    `fit_standardization` once on the TRAINING data before training.

    score_norm controls how the score-derived features (the per-doc `score`
    column, SCORE_FEATURE_INDICES, and the list_feats slot, mean_score/
    var_score) are standardized:
        "global"     : one mean/std per feature, pooled over all rankers.
                       (The score features then partly encode ranker identity,
                       since a ranker's score magnitude differs by system.)
        "per_query"  : the dataset has already made these columns scale-free
                       within each list, so the model just standardizes them
                       globally like everything else (near-identity).
        "per_ranker" : the per-doc `score` column AND the list_feats slot are
                       each standardized using per-ranker mean/std, selected
                       by column position (a ranker's column index is fixed
                       dataset-wide - see `ranker_to_id` in dataset.py). The
                       other 8 doc features and all query features are still
                       global. Requires `num_rankers` to match the dataset's
                       ranker count.

    Because flattening requires a fixed-size input, `top_k` must match the
    top_k used to build the dataset (every sample is padded/truncated to it),
    and `num_rankers` must match the dataset's ranker count.

    doc_feature_dim/list_feature_dim/query_feature_dim are NOT fixed
    constants - they come from whichever input feature blocks the QPPDataset
    this model is paired with was built with (see QPPDataset's
    `feature_blocks` param / features.ALL_FEATURE_BLOCKS: any subset of
    lexical/embedding/query_type/doc_feats). A block that dataset excluded
    has dimension 0 here too - e.g. doc_feature_dim=list_feature_dim=0 if
    "doc_feats" wasn't selected - and the concatenation/standardization math
    below is written generically enough that a 0-width block simply
    contributes nothing, with one exception: per-ranker doc-score
    standardization (see fit_standardization/_standardize) is guarded by
    `doc_feature_dim > 0`, since it indexes a fixed column position that
    doesn't exist on a 0-width doc tensor.

    If the query-side blocks include "embedding", its raw (variable-width,
    e.g. 768/384) query-representation block within `query_feats` is located
    by `embedding_slice`, read straight off the paired QPPDataset instance
    (`dataset.embedding_slice`, from features.resolve_query_feature_layout) -
    its position depends on which OTHER query blocks are also selected, so it
    is never assumed to be at a fixed offset. That raw slice is reduced to
    EMBEDDING_DIM dims by a trained `nn.Linear` (`self.embedding_proj`),
    spliced back into the same position before the MLP (see
    `_project_embedding`) - unless `reduce_embedding` is False, in which case
    `embedding_proj` is never built and the raw slice is passed through to
    the MLP at its full original width instead. Whether the RAW slice is
    z-scored before that projection (same as every other query feature) or
    left as the L2-normalized vector `features.build_embedding_feature`
    produced is controlled by `standardize_embedding` (default False,
    matching the original PCA-embedding behavior of never standardizing it)
    - the PROJECTED output is never standardized either way, since it's a
    trained internal representation, not a raw input statistic.
    """

    def __init__(
        self,
        doc_feature_dim: int,
        list_feature_dim: int,
        query_feature_dim: int,
        embedding_slice: Optional[slice] = None,
        top_k: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        score_norm: str = "global",
        num_rankers: int = 1,
        standardize_embedding: bool = False,
        reduce_embedding: bool = True,
    ):
        super().__init__()
        if score_norm not in ("global", "per_query", "per_ranker"):
            raise ValueError("score_norm must be global | per_query | per_ranker")
        self.score_norm = score_norm
        self.num_rankers = max(int(num_rankers), 1)
        self.score_cols = tuple(SCORE_FEATURE_INDICES)  # (6,)
        self.top_k = top_k
        self.doc_feature_dim = doc_feature_dim
        self.list_feature_dim = list_feature_dim
        self.query_feature_dim = query_feature_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims else list(DEFAULT_HIDDEN_DIMS)

        # --- Global standardization buffers (all features) ---------------------
        # Populated from training data via fit_standardization(). Registered as
        # buffers so they travel inside state_dict. Identity until fitted.
        self.register_buffer("doc_mean", torch.zeros(doc_feature_dim))
        self.register_buffer("doc_std", torch.ones(doc_feature_dim))
        self.register_buffer("list_mean", torch.zeros(list_feature_dim))
        self.register_buffer("list_std", torch.ones(list_feature_dim))
        self.register_buffer("query_mean", torch.zeros(query_feature_dim))
        self.register_buffer("query_std", torch.ones(query_feature_dim))

        # --- Per-ranker standardization buffers -----------------------------
        # doc-side: score column only, shape (num_rankers, len(score_cols)).
        # list-side: both list features, shape (num_rankers, list_feature_dim).
        # Always registered (so state_dict keys are stable) but only used
        # when score_norm == "per_ranker". num_rankers must match between the
        # saved checkpoint and the model at load time.
        n_score = len(self.score_cols)
        self.register_buffer("ranker_score_mean", torch.zeros(self.num_rankers, n_score))
        self.register_buffer("ranker_score_std", torch.ones(self.num_rankers, n_score))
        self.register_buffer("ranker_list_mean", torch.zeros(self.num_rankers, list_feature_dim))
        self.register_buffer("ranker_list_std", torch.ones(self.num_rankers, list_feature_dim))

        # Embedding slice within query_feats - passed in explicitly by the
        # caller (normally dataset.embedding_slice), since its offset depends
        # on which other query blocks are selected. None if "embedding" isn't
        # part of this model's query input at all.
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

        in_dim = (
            self.num_rankers * (top_k * doc_feature_dim + list_feature_dim)
            + query_feature_dim - raw_dim + proj_dim
        )
        layers = []
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, self.num_rankers)]
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_standardization(self, loader: DataLoader, eps: float = 1e-6) -> None:
        """
        Estimate standardization statistics from TRAINING data and store them in
        buffers. Call once, before training, on a loader over the training set.

        Always computes global per-feature mean/std for both doc_feats (padded
        doc positions, and rows for rankers absent for a given query, are
        excluded) and list_feats (rankers absent for a given query are
        excluded, via ranker_mask - list_feats has no per-doc pad_mask, since
        it's already one value per (query, ranker)). When
        score_norm == "per_ranker", additionally computes per-ranker mean/std
        for the doc-side score column(s) and for list_feats, using each
        ranker's fixed column position.

        The embedding columns (self.embedding_slice, if set) are excluded
        UNLESS self.standardize_embedding is True: by default their mean/std
        are forced to identity (0, 1) regardless of the fitted statistics,
        since they're L2-normalized per query at feature-construction time
        and standardizing them on top of that isn't always desired; setting
        standardize_embedding=True instead lets them get real computed
        statistics like every other query feature (see class docstring).
        """
        device = self.doc_mean.device
        idx = torch.tensor(self.score_cols, device=device, dtype=torch.long)

        doc_sum = torch.zeros_like(self.doc_mean)
        doc_sqsum = torch.zeros_like(self.doc_mean)
        doc_count = 0

        list_sum = torch.zeros_like(self.list_mean)
        list_sqsum = torch.zeros_like(self.list_mean)
        list_count = 0

        q_sum = torch.zeros_like(self.query_mean)
        q_sqsum = torch.zeros_like(self.query_mean)
        q_count = 0

        per_ranker = self.score_norm == "per_ranker"
        # The doc-side score column only exists if doc_feats is part of this
        # model's input at all (doc_feature_dim > 0) - guards the
        # index_select below, which would be out of range on a 0-width doc
        # tensor otherwise. list_feats needs no such guard: it's standardized
        # as a whole tensor, not by indexing a fixed column within it, so it
        # stays safe (and a no-op) even at width 0.
        has_doc = self.doc_feature_dim > 0
        if per_ranker:
            if has_doc:
                r_sum = torch.zeros_like(self.ranker_score_mean)      # (R, n_score)
                r_sqsum = torch.zeros_like(self.ranker_score_mean)    # (R, n_score)
                r_count = torch.zeros(self.num_rankers, device=device)

            r_list_sum = torch.zeros_like(self.ranker_list_mean)    # (R, list_feature_dim)
            r_list_sqsum = torch.zeros_like(self.ranker_list_mean)  # (R, list_feature_dim)
            r_list_count = torch.zeros(self.num_rankers, device=device)

        for doc_feats, list_feats, query_feats, pad_mask, ranker_mask, _ in loader:
            doc_feats = doc_feats.to(device)      # (B, R, k, F)
            list_feats = list_feats.to(device)    # (B, R, L)
            query_feats = query_feats.to(device)  # (B, Q)
            pad_mask = pad_mask.to(device)        # (B, R, k)
            ranker_mask = ranker_mask.to(device)  # (B, R)

            valid = ~pad_mask & ranker_mask.unsqueeze(-1)  # (B, R, k)
            vf = doc_feats[valid]                           # (n_valid, F)
            doc_sum += vf.sum(dim=0)
            doc_sqsum += (vf * vf).sum(dim=0)
            doc_count += vf.shape[0]

            lvf = list_feats[ranker_mask]                   # (n_valid_r, L)
            list_sum += lvf.sum(dim=0)
            list_sqsum += (lvf * lvf).sum(dim=0)
            list_count += lvf.shape[0]

            q_sum += query_feats.sum(dim=0)
            q_sqsum += (query_feats * query_feats).sum(dim=0)
            q_count += query_feats.shape[0]

            if per_ranker:
                if has_doc:
                    sc = doc_feats.index_select(-1, idx)          # (B, R, k, n_score)
                    mask_f = valid.unsqueeze(-1).float()           # (B, R, k, 1)
                    r_sum += (sc * mask_f).sum(dim=(0, 2))
                    r_sqsum += (sc * sc * mask_f).sum(dim=(0, 2))
                    r_count += valid.sum(dim=(0, 2)).float()

                rmask_f = ranker_mask.unsqueeze(-1).float()    # (B, R, 1)
                r_list_sum += (list_feats * rmask_f).sum(dim=0)
                r_list_sqsum += (list_feats * list_feats * rmask_f).sum(dim=0)
                r_list_count += ranker_mask.sum(dim=0).float()

        doc_mean = doc_sum / max(doc_count, 1)
        doc_var = (doc_sqsum / max(doc_count, 1)) - doc_mean**2
        list_mean = list_sum / max(list_count, 1)
        list_var = (list_sqsum / max(list_count, 1)) - list_mean**2
        q_mean = q_sum / max(q_count, 1)
        q_var = (q_sqsum / max(q_count, 1)) - q_mean**2
        q_std = q_var.clamp_min(0).sqrt() + eps

        if self.embedding_slice is not None and not self.standardize_embedding:
            q_mean[self.embedding_slice] = 0.0
            q_std[self.embedding_slice] = 1.0

        self.doc_mean.copy_(doc_mean)
        self.doc_std.copy_(doc_var.clamp_min(0).sqrt() + eps)
        self.list_mean.copy_(list_mean)
        self.list_std.copy_(list_var.clamp_min(0).sqrt() + eps)
        self.query_mean.copy_(q_mean)
        self.query_std.copy_(q_std)

        if per_ranker:
            if has_doc:
                cnt = r_count.clamp_min(1).unsqueeze(1)     # (R, 1)
                r_mean = r_sum / cnt
                r_var = (r_sqsum / cnt) - r_mean**2
                self.ranker_score_mean.copy_(r_mean)
                self.ranker_score_std.copy_(r_var.clamp_min(0).sqrt() + eps)

            list_cnt = r_list_count.clamp_min(1).unsqueeze(1)     # (R, 1)
            r_list_mean = r_list_sum / list_cnt
            r_list_var = (r_list_sqsum / list_cnt) - r_list_mean**2
            self.ranker_list_mean.copy_(r_list_mean)
            self.ranker_list_std.copy_(r_list_var.clamp_min(0).sqrt() + eps)

    def _standardize(self, doc_feats: Tensor, list_feats: Tensor, query_feats: Tensor):
        """Standardize doc + list + query features according to score_norm.

        doc_feats: (B, R, top_k, doc_feature_dim); list_feats: (B, R,
        list_feature_dim); query_feats: (B, query_feature_dim).
        """
        query_feats = (query_feats - self.query_mean) / self.query_std
        R = doc_feats.size(1)

        if self.score_norm == "per_ranker" and self.doc_feature_dim > 0:
            idx = torch.tensor(self.score_cols, device=doc_feats.device, dtype=torch.long)
            raw_score = doc_feats.index_select(-1, idx)          # (B, R, k, n_score) RAW
            r_mean = self.ranker_score_mean[:R]                  # (R, n_score)
            r_std = self.ranker_score_std[:R]                    # (R, n_score)
            score_std = (raw_score - r_mean.view(1, R, 1, -1)) / r_std.view(1, R, 1, -1)

            doc_feats = (doc_feats - self.doc_mean) / self.doc_std   # global everywhere
            doc_feats = doc_feats.index_copy(-1, idx, score_std)     # override score cols
        else:
            # "global"/"per_query", or "per_ranker" with no doc_feats at all
            # (nothing to index by ranker column - doc_feature_dim == 0).
            doc_feats = (doc_feats - self.doc_mean) / self.doc_std

        if self.score_norm == "per_ranker":
            r_list_mean = self.ranker_list_mean[:R]              # (R, L)
            r_list_std = self.ranker_list_std[:R]                # (R, L)
            list_feats = (list_feats - r_list_mean.view(1, R, -1)) / r_list_std.view(1, R, -1)
        else:
            list_feats = (list_feats - self.list_mean) / self.list_std

        return doc_feats, list_feats, query_feats

    def _project_embedding(self, query_feats: Tensor) -> Tensor:
        """Replace the raw embedding_slice columns of query_feats with their
        EMBEDDING_DIM-dim projection through self.embedding_proj, leaving
        every other column untouched. No-op if embedding_slice is None."""
        if self.embedding_proj is None:
            return query_feats
        s = self.embedding_slice
        before, raw, after = query_feats[:, :s.start], query_feats[:, s], query_feats[:, s.stop:]
        return torch.cat([before, self.embedding_proj(raw), after], dim=1)

    def forward(
        self,
        doc_feats: Tensor,
        list_feats: Tensor,
        query_feats: Tensor,
        pad_mask: Tensor = None,
        ranker_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            doc_feats:   (B, R, top_k, doc_feature_dim) — R = num_rankers
            list_feats:  (B, R, list_feature_dim) — once per ranker (mean_score,
                         var_score), NOT duplicated across that ranker's top_k docs
            query_feats: (B, query_feature_dim) — shared across rankers
            pad_mask:    (B, R, top_k) bool — True for padding doc positions
            ranker_mask: (B, R) bool — True where a ranker has data for this
                         query; absent rankers' logits are set to -inf.

        Returns:
            (B, R) per-ranker logits. softmax(dim=-1) gives a distribution
            over "which ranker is best for this query".
        """
        B, R, k, _ = doc_feats.shape
        if R != self.num_rankers:
            raise ValueError(
                f"QPPMLP was built with num_rankers={self.num_rankers}, but got "
                f"R={R}; the concatenation MLP head requires a fixed ranker "
                f"count matching construction time."
            )

        doc_feats, list_feats, query_feats = self._standardize(doc_feats, list_feats, query_feats)
        query_feats = self._project_embedding(query_feats)
        # Zero padded rows (including rankers absent for this query, which are
        # fully padded) so they contribute nothing to the flattened vector.
        if pad_mask is not None:
            doc_feats = doc_feats.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        # list_feats has no per-doc pad_mask (one value per ranker, not per
        # doc); zero it directly via ranker_mask so an absent ranker's raw
        # zero doesn't become a non-zero -mean/std after standardization.
        if ranker_mask is not None:
            list_feats = list_feats.masked_fill(~ranker_mask.unsqueeze(-1), 0.0)

        # Put list_feats at the START of each ranker's block, followed by
        # that ranker's flattened top_k doc features - so mean_score/var_score
        # appear ONCE per ranker instead of being duplicated across top_k docs.
        doc_flat = doc_feats.reshape(B, R, k * self.doc_feature_dim)     # (B, R, k*F)
        per_ranker_block = torch.cat([list_feats, doc_flat], dim=-1)     # (B, R, L + k*F)
        flat = per_ranker_block.reshape(B, R * (self.list_feature_dim + k * self.doc_feature_dim))
        x = torch.cat([flat, query_feats], dim=1)

        logits = self.mlp(x)  # (B, R)
        if ranker_mask is not None:
            logits = logits.masked_fill(~ranker_mask, float("-inf"))
        return logits


class QueryOnlyMLP(nn.Module):
    """
    Query-features-only ablation of QPPMLP: same MLP classification head,
    but with the entire per-ranker ranked-list / doc-feature side removed,
    to measure how much of QPPMLP's accuracy comes from the doc-side signal
    versus query characteristics alone.

    Input is just the query vector (shared across all rankers for a given
    query, so it is passed through the MLP once, not once per ranker);
    output is one logit per ranker, matching QPPMLP's output structure —
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
