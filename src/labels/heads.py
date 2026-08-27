# -*- coding: utf-8 -*-
"""Label head definitions for Infectivity_ViT v1."""
from __future__ import annotations

# Active training / eval heads (CSV columns)
TRAIN_HEADS = ["D1", "D2", "D3", "D4", "D5", "D7"]

# Phase III private-tag convention (not used in v1 training)
D6_NTM_RESERVED = "D6"  # NTM flag — embed only, v1 미학습

HEAD_MEANINGS = {
    "D1": "AFB smear (graded)",
    "D2": "TB-PCR (soft)",
    "D3": "Solid culture TTP (loginv)",
    "D4": "Liquid culture TTP (twostep-C)",
    "D5": "Cavity (binary)",
    "D6": "NTM (reserved)",
    "D7": "RIF resistance PCR (soft, same as D2)",
}
