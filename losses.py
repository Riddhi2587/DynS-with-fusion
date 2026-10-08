"""
Listwise ranking losses for QueryOnlyMLP.

listmle_loss is adapted from allRank's listMLE.py
(https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py).
The only real change is how "missing" ranker slots are marked: allRank pads
slates to a fixed length and flags padding with a sentinel value in y_true
(PADDED_Y_VALUE), whereas here every query already carries a boolean
`ranker_mask` (True = ranker present) from dataset.py/model.py, so that mask
is used directly instead of a sentinel comparison.
"""

import torch

DEFAULT_EPS = 1e-10


def listmle_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """
    ListMLE loss (Xia et al., 2008): the negative log-likelihood of the
    ground-truth ranking (by y_true) under a Plackett-Luce model parameterized
    by y_pred.

    Args:
        y_pred: (B, R) predicted scores (e.g. QueryOnlyMLP's per-ranker logits).
        y_true: (B, R) ground-truth relevance, e.g. per-ranker NDCG - higher is better.
        mask:   (B, R) bool, True where the ranker is present/valid for that query.
        eps:    numerical-stability epsilon for the log.

    Returns:
        Scalar loss (mean over the batch of the per-query listwise NLL).
    """
    B, R = y_pred.shape

    # Randomized tie-breaking, as in the reference implementation: ties in
    # y_true (e.g. two rankers with identical NDCG) get a random relative
    # order instead of a fixed positional bias from stable sort.
    random_indices = torch.randperm(R, device=y_pred.device)
    y_pred = y_pred[:, random_indices]
    y_true = y_true[:, random_indices]
    mask = mask[:, random_indices]

    # Sort each row by descending true relevance; absent rankers (mask=False)
    # are pushed to the end via -inf so they land after all real entries.
    y_true = y_true.masked_fill(~mask, float("-inf"))
    y_true_sorted, indices = y_true.sort(descending=True, dim=-1)
    pad = y_true_sorted == float("-inf")

    preds_sorted_by_true = torch.gather(y_pred, dim=1, index=indices)
    preds_sorted_by_true = preds_sorted_by_true.masked_fill(pad, float("-inf"))

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    observation_loss = observation_loss.masked_fill(pad, 0.0)

    return observation_loss.sum(dim=1).mean()
