# -*- coding: utf-8 -*-
"""Layer-2 AFB grade calibration: ridit, PLS(CXR), polychoric, optimal scaling, isotonic."""
from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression

from src.labels.encoders import encode_afb_grade_v1

_V1_ROOT = Path(__file__).resolve().parents[2]
_TB3_POLY = Path(r"D:\TBC_CAD_INFECTIVITY\TB Phase III")
if str(_TB3_POLY) not in sys.path:
    sys.path.insert(0, str(_TB3_POLY))

CANONICAL_RAW_STRINGS: list[tuple[str, str]] = [
    ("No AFB seen", "negative"),
    ("Negative for AFB", "negative"),
    ("Doubtful Repeat test (Trace)", "trace"),
    ("Rare (1+)", "1+"),
    ("Few (2+)", "2+"),
    ("Moderate (3+)", "3+"),
    ("Numerous (4+)", "4+"),
    ("Positive for AFB", "4+_ungraded"),
]

D1_TO_ORDINAL: dict[float, int] = {
    0.0: 0,
    0.125: 1,
    0.25: 2,
    0.5: 3,
    0.75: 4,
    1.0: 5,
}

ORDINAL_LABELS = ["0_neg", "1_trace", "2_1plus", "3_2plus", "4_3plus", "5_4plus"]

NPZ_DEFAULT = Path(
    r"D:\TB Phase III\artifacts\phase3_v7_011_densenet121_clahe_1024_rf_missing"
    r"\phase3_v7.011_features_d1d5_missing.npz"
)


def study_of(path: str) -> int:
    m = re.match(r"^(\d+)", Path(str(path)).stem)
    if not m:
        raise ValueError(f"Cannot parse study id from path: {path!r}")
    return int(m.group(1))


def d1_to_ordinal(d1: float) -> int | None:
    if d1 is None or (isinstance(d1, float) and np.isnan(d1)):
        return None
    key = round(float(d1), 6)
    for k, v in D1_TO_ORDINAL.items():
        if abs(key - k) < 1e-6:
            return v
    return None


def raw_to_ordinal(raw: str) -> int | None:
    v = encode_afb_grade_v1(raw)
    if v is None:
        return None
    return d1_to_ordinal(v)


def ordinal_to_grade_v1(ordinal: int) -> float:
    inv = {v: k for k, v in D1_TO_ORDINAL.items()}
    return float(inv[ordinal])


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def culture_anchor(d3: np.ndarray, d4: np.ndarray) -> np.ndarray:
    return _zscore(d3) + _zscore(d4)


def _normalize_grade_scores(scores: dict[int, float], fix_zero: bool = True) -> dict[int, float]:
    """Affine map observed grades to [0, 1]; optionally pin grade 0 at 0."""
    if not scores:
        return scores
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    out: dict[int, float] = {}
    for g, v in scores.items():
        if hi - lo < 1e-12:
            out[g] = 0.0 if g == 0 else 1.0
        else:
            out[g] = (float(v) - lo) / (hi - lo)
    if fix_zero and 0 in out:
        out[0] = 0.0
    if 5 in out:
        out[5] = max(out.get(5, 1.0), out.get(4, 0.0))
        out[5] = min(max(out[5], 0.0), 1.0)
    return out


def check_monotone(scores: dict[int, float]) -> bool:
    keys = sorted(scores.keys())
    return all(scores[keys[i]] <= scores[keys[i + 1]] + 1e-9 for i in range(len(keys) - 1))


def fit_ridit(ordinals: np.ndarray) -> tuple[dict[int, float], dict[str, Any]]:
    ctr = Counter(int(o) for o in ordinals)
    n = sum(ctr.values())
    probs = {g: ctr.get(g, 0) / n for g in range(6)}
    scores: dict[int, float] = {}
    for g in range(6):
        below = sum(probs.get(gp, 0.0) for gp in range(g))
        scores[g] = below + 0.5 * probs.get(g, 0.0)
    meta = {"n": n, "grade_counts": {str(k): ctr.get(k, 0) for k in range(6)}}
    return scores, meta


def _study_features_from_npz(npz_path: Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    paths = np.asarray(d["paths"], dtype=object)
    study_feats: dict[int, list[np.ndarray]] = {}
    for i, p in enumerate(paths):
        sid = study_of(str(p))
        study_feats.setdefault(sid, []).append(X[i])
    out = {s: np.mean(np.stack(v, axis=0), axis=0) for s, v in study_feats.items()}
    meta = {"n_images": int(len(paths)), "n_studies": len(out), "feature_dim": int(X.shape[1])}
    return out, meta


def fit_pls_cxr(
    ordinals: np.ndarray,
    study_ids: np.ndarray,
    study_features: dict[int, np.ndarray],
    *,
    positive_only: bool = True,
) -> tuple[dict[int, float], dict[str, Any]]:
    rows = []
    for sid, ord_g in zip(study_ids, ordinals):
        if sid not in study_features:
            continue
        if positive_only and int(ord_g) <= 0:
            continue
        rows.append((int(sid), int(ord_g), study_features[sid]))
    if len(rows) < 10:
        raise ValueError(f"PLS CXR: too few joined studies (n={len(rows)})")

    y = np.array([r[1] for r in rows], dtype=float)
    X = np.stack([r[2] for r in rows], axis=0)
    if len(np.unique(y)) < 2:
        raise ValueError("PLS CXR: ordinal Y has <2 unique levels in AFB+ subset")

    pls = PLSRegression(n_components=1, scale=True)
    pls.fit(X, y)
    lv1 = pls.transform(X).ravel()

    centroids: dict[int, float] = {0: 0.0}
    for g in range(1, 6):
        mask = y == g
        if mask.any():
            centroids[g] = float(np.mean(lv1[mask]))

    present = [g for g in range(1, 6) if g in centroids]
    if len(present) < 2:
        raise ValueError("PLS CXR: <2 positive grade centroids available")

    pos_centroids = {g: centroids[g] for g in present}
    xs = np.array(sorted(pos_centroids.keys()), dtype=float)
    ys = np.array([pos_centroids[int(g)] for g in xs], dtype=float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(xs, ys)
    mono_centroids = {int(g): float(iso.predict(np.array([g], dtype=float))[0]) for g in xs}
    normed = _normalize_grade_scores(mono_centroids, fix_zero=False)
    scores: dict[int, float] = {0: 0.0}
    for g in present:
        scores[g] = normed[g]
    for g in range(6):
        scores.setdefault(g, scores.get(max(present), 1.0))
    if 5 in present:
        scores[5] = 1.0

    meta = {
        "n_used": len(rows),
        "n_studies_with_cxr": len(study_features),
        "grade_counts": {str(g): int((y == g).sum()) for g in range(1, 6)},
        "lv1_centroids_raw": {str(k): centroids[k] for k in present},
        "lv1_centroids_isotonic": {str(k): mono_centroids[k] for k in present},
        "monotonicity_corrected": True,
        "pls_x_variance_explained": float(pls.score(X, y)),
    }
    return scores, meta


def _quantile_bin(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    s = pd.Series(x)
    try:
        b = pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        b = pd.cut(s, bins=n_bins, labels=False, duplicates="drop")
    return np.asarray(b.fillna(0).astype(int))


def fit_polychoric_culture(
    ordinals: np.ndarray,
    d3: np.ndarray,
    d4: np.ndarray,
) -> tuple[dict[int, float], dict[str, Any]]:
    mask = np.isfinite(d3) & np.isfinite(d4) & np.isfinite(ordinals)
    o = ordinals[mask].astype(int)
    anchor = culture_anchor(d3[mask], d4[mask])
    d3b = _quantile_bin(d3[mask], 5)
    d4b = _quantile_bin(d4[mask], 5)

    cond_means: dict[int, float] = {}
    for g in range(6):
        m = o == g
        if m.any():
            cond_means[g] = float(np.mean(anchor[m]))
    scores = _normalize_grade_scores(cond_means, fix_zero=True)

    rho_afb_d3 = rho_afb_d4 = None
    try:
        from analyze_infectivity_polychoric import polychoric_corr_matrix  # noqa: WPS433

        Xpoly = np.column_stack([o, d3b, d4b])
        R, _ = polychoric_corr_matrix(Xpoly, ["AFB_ord", "D3_bin", "D4_bin"])
        rho_afb_d3 = float(R[0, 1])
        rho_afb_d4 = float(R[0, 2])
    except Exception as exc:  # pragma: no cover
        rho_afb_d3 = rho_afb_d4 = None
        poly_err = str(exc)
    else:
        poly_err = None

    meta = {
        "n_used": int(mask.sum()),
        "rho_afb_d3": rho_afb_d3,
        "rho_afb_d4": rho_afb_d4,
        "polychoric_error": poly_err,
        "cond_mean_anchor": {str(k): cond_means[k] for k in sorted(cond_means)},
    }
    return scores, meta


def fit_optimal_scaling(
    ordinals: np.ndarray,
    d3: np.ndarray,
    d4: np.ndarray,
) -> tuple[dict[int, float], dict[str, Any]]:
    mask = np.isfinite(d3) & np.isfinite(d4) & np.isfinite(ordinals)
    o = ordinals[mask].astype(int)
    anchor = culture_anchor(d3[mask], d4[mask])
    if len(o) < 20:
        raise ValueError(f"Optimal scaling: too few complete cases (n={len(o)})")

    def neg_corr(params: np.ndarray) -> float:
        s = np.array([0.0, params[0], params[1], params[2], params[3], 1.0], dtype=float)
        if np.any(np.diff(s) < -1e-9):
            return 1e6 + float(np.sum(np.maximum(0.0, -np.diff(s)) ** 2) * 1e4)
        mapped = s[o]
        if np.std(mapped) < 1e-12 or np.std(anchor) < 1e-12:
            return 0.0
        return float(-np.corrcoef(mapped, anchor)[0, 1])

    x0 = np.linspace(0.1, 0.9, 4)
    res = optimize.minimize(neg_corr, x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * 4)
    s_opt = np.array([0.0, res.x[0], res.x[1], res.x[2], res.x[3], 1.0], dtype=float)
    for i in range(1, 5):
        s_opt[i] = max(s_opt[i], s_opt[i - 1])
    s_opt[5] = 1.0
    scores = {g: float(s_opt[g]) for g in range(6)}
    meta = {
        "n_used": int(mask.sum()),
        "correlation": float(-res.fun) if res.fun < 1e5 else None,
        "optimizer_success": bool(res.success),
    }
    return scores, meta


def fit_isotonic_anchor(
    ordinals: np.ndarray,
    d3: np.ndarray,
    d4: np.ndarray,
) -> tuple[dict[int, float], dict[str, Any]]:
    mask = np.isfinite(d3) & np.isfinite(d4) & np.isfinite(ordinals)
    o = ordinals[mask].astype(int)
    anchor = culture_anchor(d3[mask], d4[mask])

    medians: dict[int, float] = {}
    for g in range(6):
        m = o == g
        if m.any():
            medians[g] = float(np.median(anchor[m]))

    xs = np.array(sorted(medians.keys()), dtype=float)
    ys = np.array([medians[int(g)] for g in xs], dtype=float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(xs, ys)
    pred = iso.predict(np.arange(6, dtype=float))
    lo, hi = float(pred.min()), float(pred.max())
    scores: dict[int, float] = {}
    for g in range(6):
        if hi - lo < 1e-12:
            scores[g] = 0.0 if g == 0 else 1.0
        else:
            scores[g] = float((pred[g] - lo) / (hi - lo))
    scores[0] = 0.0
    meta = {
        "n_used": int(mask.sum()),
        "grade_median_anchor": {str(k): medians[k] for k in sorted(medians)},
    }
    return scores, meta


@dataclass
class Layer2Comparison:
    grade_scores: dict[str, dict[int, float]] = field(default_factory=dict)
    method_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    monotone: dict[str, bool] = field(default_factory=dict)
    raw_table: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade_scores": {k: {str(g): v for g, v in d.items()} for k, d in self.grade_scores.items()},
            "method_meta": self.method_meta,
            "monotone": self.monotone,
            "raw_table": self.raw_table,
        }


def load_labels_table(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path, encoding="utf-8-sig")
    df["Study No."] = df["Study No."].astype(int)
    for col in ("D1", "D3", "D4"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ordinal"] = df["D1"].map(d1_to_ordinal)
    return df


def run_all_methods(
    labels_path: Path,
    *,
    npz_path: Path = NPZ_DEFAULT,
) -> Layer2Comparison:
    df = load_labels_table(labels_path)
    labeled = df[df["ordinal"].notna()].copy()
    ordinals = labeled["ordinal"].astype(int).to_numpy()
    study_ids = labeled["Study No."].to_numpy()
    d3 = labeled["D3"].to_numpy(dtype=float)
    d4 = labeled["D4"].to_numpy(dtype=float)

    cmp = Layer2Comparison()
    baseline = {g: ordinal_to_grade_v1(g) for g in range(6)}
    cmp.grade_scores["afb_grade_v1"] = baseline
    cmp.monotone["afb_grade_v1"] = check_monotone(baseline)

    ridit_scores, ridit_meta = fit_ridit(ordinals)
    cmp.grade_scores["ridit"] = ridit_scores
    cmp.method_meta["ridit"] = ridit_meta
    cmp.monotone["ridit"] = check_monotone(ridit_scores)

    poly_scores, poly_meta = fit_polychoric_culture(ordinals, d3, d4)
    cmp.grade_scores["polychoric"] = poly_scores
    cmp.method_meta["polychoric"] = poly_meta
    cmp.monotone["polychoric"] = check_monotone(poly_scores)

    opt_scores, opt_meta = fit_optimal_scaling(ordinals, d3, d4)
    cmp.grade_scores["optimal_scaling"] = opt_scores
    cmp.method_meta["optimal_scaling"] = opt_meta
    cmp.monotone["optimal_scaling"] = check_monotone(opt_scores)

    iso_scores, iso_meta = fit_isotonic_anchor(ordinals, d3, d4)
    cmp.grade_scores["isotonic_anchor"] = iso_scores
    cmp.method_meta["isotonic_anchor"] = iso_meta
    cmp.monotone["isotonic_anchor"] = check_monotone(iso_scores)

    pls_err = None
    try:
        study_feats, npz_meta = _study_features_from_npz(npz_path)
        pls_scores, pls_meta = fit_pls_cxr(ordinals, study_ids, study_feats)
        pls_meta.update(npz_meta)
        cmp.grade_scores["pls_cxr"] = pls_scores
        cmp.method_meta["pls_cxr"] = pls_meta
        cmp.monotone["pls_cxr"] = check_monotone(pls_scores)
    except Exception as exc:
        pls_err = str(exc)
        cmp.method_meta["pls_cxr"] = {"error": pls_err, "npz": str(npz_path)}

    methods = ["afb_grade_v1", "ridit", "pls_cxr", "polychoric", "optimal_scaling", "isotonic_anchor"]
    for raw, _tag in CANONICAL_RAW_STRINGS:
        ord_g = raw_to_ordinal(raw)
        if ord_g is None:
            continue
        row: dict[str, Any] = {
            "raw_text": raw,
            "ordinal": ord_g,
            "ordinal_label": ORDINAL_LABELS[ord_g],
        }
        for m in methods:
            scores = cmp.grade_scores.get(m)
            if scores is None:
                row[m] = None
            else:
                row[m] = round(float(scores[ord_g]), 6)
        cmp.raw_table.append(row)

    return cmp


def format_comparison_md(cmp: Layer2Comparison) -> str:
    lines = [
        "# AFB Layer-2 Score Comparison (empirical)",
        "",
        "## Raw text → score by method",
        "",
        "| Raw text | ord | grade_v1 | ridit | pls_cxr | polychoric | optimal | isotonic |",
        "|----------|----:|---------:|------:|--------:|-----------:|--------:|---------:|",
    ]
    for row in cmp.raw_table:
        def _f(key: str) -> str:
            v = row.get(key)
            return "—" if v is None else f"{v:.4f}"

        lines.append(
            f"| `{row['raw_text']}` | {row['ordinal']} | "
            f"{_f('afb_grade_v1')} | {_f('ridit')} | {_f('pls_cxr')} | "
            f"{_f('polychoric')} | {_f('optimal_scaling')} | {_f('isotonic_anchor')} |"
        )

    lines.extend(["", "## Grade-level f(g)", ""])
    lines.append("| grade | ord | " + " | ".join(cmp.grade_scores.keys()) + " |")
    lines.append("|-------|----:|" + "|".join([":---:"] * len(cmp.grade_scores)) + "|")
    for g in range(6):
        cells = [ORDINAL_LABELS[g], str(g)]
        for m, scores in cmp.grade_scores.items():
            cells.append(f"{scores.get(g, float('nan')):.4f}" if g in scores else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Monotonicity", ""])
    for m, ok in cmp.monotone.items():
        lines.append(f"- **{m}**: {'OK' if ok else 'VIOLATION'}")

    lines.extend(["", "## Method metadata", ""])
    for m, meta in cmp.method_meta.items():
        lines.append(f"### {m}")
        for k, v in meta.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines)
