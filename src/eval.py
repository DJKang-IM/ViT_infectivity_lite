# -*- coding: utf-8 -*-
"""Per-head evaluation metrics for graded labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.labels.heads import TRAIN_HEADS

HEADS = TRAIN_HEADS


def aggregate_by_study(
    study_nos: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse slice-level preds to one row per study (labels are study-level)."""
    if method not in ("mean", "median", "max"):
        raise ValueError(f"method must be mean|median|max, got {method!r}")

    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for study in sorted({int(s) for s in study_nos}):
        idx = np.array([i for i, s in enumerate(study_nos) if int(s) == study])
        p = y_pred[idx]
        t = y_true[idx]
        if method == "mean":
            p_agg = np.mean(p, axis=0)
        elif method == "median":
            p_agg = np.median(p, axis=0)
        else:
            p_agg = np.max(p, axis=0)
        targets.append(t[0])
        preds.append(p_agg)
    return np.stack(targets), np.stack(preds)


def per_head_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    binary_threshold: float = 0.5,
) -> dict[str, dict]:
    """y_true/y_pred: (N, num_heads). NaN in y_true => missing."""
    out: dict[str, dict] = {}
    for i, h in enumerate(HEADS):
        t = y_true[:, i]
        p = y_pred[:, i]
        valid = ~np.isnan(t)
        n = int(valid.sum())
        if n == 0:
            out[h] = {"n": 0}
            continue
        tv = t[valid]
        pv = p[valid]
        mse = float(np.mean((pv - tv) ** 2))
        sp = float(spearmanr(tv, pv).statistic) if n >= 2 and np.std(tv) > 0 and np.std(pv) > 0 else float("nan")
        metrics: dict = {"n": n, "mse": mse, "spearman": sp, "mean_target": float(tv.mean()), "mean_pred": float(pv.mean())}
        # binary AUROC for rough Phase III comparison
        bin_t = (tv >= binary_threshold).astype(int)
        if len(np.unique(bin_t)) == 2:
            try:
                metrics["auroc"] = float(roc_auc_score(bin_t, pv))
            except ValueError:
                metrics["auroc"] = float("nan")
        else:
            metrics["auroc"] = float("nan")
        out[h] = metrics
    return out


def macro_average(metrics: dict[str, dict], key: str) -> float:
    vals = [m[key] for m in metrics.values() if m.get("n", 0) > 0 and key in m and not np.isnan(m[key])]
    return float(np.mean(vals)) if vals else float("nan")


def save_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
