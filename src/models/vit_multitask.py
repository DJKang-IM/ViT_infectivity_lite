# -*- coding: utf-8 -*-
"""ViT multi-head regression model for D1-D5."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ViTMultiTask(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        *,
        pretrained: bool = True,
        num_heads: int = 5,
        img_size: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        feat_dim = self.backbone.num_features
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        outs = [h(feat) for h in self.heads]
        return torch.cat(outs, dim=1)  # (B, num_heads)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target/mask: (B, H). mask=1 where label present."""
    diff = (pred - target) ** 2
    weighted = diff * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom
