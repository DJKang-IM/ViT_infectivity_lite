# -*- coding: utf-8 -*-
"""Backbone factory for CXR MTL baselines (ViT-Small, Swin-Tiny)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.models.mtl_heads import DEFAULT_HEAD_TYPES, MTLHeads

BACKBONES = {
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


class MTLModel(nn.Module):
    """timm backbone (feature only) + multi-task heads."""

    def __init__(
        self,
        backbone: str = "vit_small",
        *,
        pretrained: bool = True,
        img_size: int = 256,
        head_types: list[str] | None = None,
    ) -> None:
        super().__init__()
        model_name = BACKBONES.get(backbone, backbone)
        kwargs = dict(pretrained=pretrained, num_classes=0)
        # ViT supports flexible img_size via pos-embed interpolation; Swin needs it too.
        try:
            self.backbone = timm.create_model(model_name, img_size=img_size, **kwargs)
        except TypeError:
            self.backbone = timm.create_model(model_name, **kwargs)
        feat_dim = self.backbone.num_features
        self.heads = MTLHeads(feat_dim, head_types or DEFAULT_HEAD_TYPES)

    @property
    def head_types(self) -> list[str]:
        return self.heads.head_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = feat.mean(dim=tuple(range(1, feat.ndim - 1))) if feat.ndim == 4 else feat
        return self.heads(feat)


def create_mtl_model(
    backbone: str,
    *,
    pretrained: bool = True,
    img_size: int = 256,
    head_types: list[str] | None = None,
) -> MTLModel:
    return MTLModel(
        backbone,
        pretrained=pretrained,
        img_size=img_size,
        head_types=head_types,
    )
