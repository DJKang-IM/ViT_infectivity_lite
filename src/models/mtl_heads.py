# -*- coding: utf-8 -*-
"""Multi-task heads and masked loss for CXR MTL (D1..D5).

Head types:
  reg (D1, D3, D4): single raw Linear -> MSE loss (targets in [0,1]).
  clf (D2, D5):     single raw Linear (logit) -> BCEWithLogits loss.

Outputs are raw (no activation); apply sigmoid at inference for clf heads
and clamp reg heads to [0,1] for metrics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default head layout for D1..D5
DEFAULT_HEAD_TYPES = ["reg", "clf", "reg", "reg", "clf"]


class MTLHeads(nn.Module):
    def __init__(self, feat_dim: int, head_types: list[str] | None = None) -> None:
        super().__init__()
        self.head_types = list(head_types or DEFAULT_HEAD_TYPES)
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in self.head_types])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        outs = [h(feat) for h in self.heads]
        return torch.cat(outs, dim=1)  # (B, H) raw


def masked_mtl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    head_types: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of per-head masked losses (MSE for reg, BCEWithLogits for clf).

    pred/target/mask: (B, H). Returns (total_loss, per_head_loss_dict).
    """
    total = pred.new_zeros(())
    per_head: dict[str, float] = {}
    for j, ht in enumerate(head_types):
        m = mask[:, j]
        denom = m.sum().clamp(min=1.0)
        p = pred[:, j]
        t = target[:, j]
        if ht == "clf":
            loss_j = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        else:
            loss_j = (p - t) ** 2
        loss_j = (loss_j * m).sum() / denom
        total = total + loss_j
        per_head[f"D{j + 1}"] = float(loss_j.detach())
    return total, per_head
