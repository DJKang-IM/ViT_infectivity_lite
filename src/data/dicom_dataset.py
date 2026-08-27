# -*- coding: utf-8 -*-
"""DICOM dataset: study-level labels, slice-level samples (CV stays at study)."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import build_transform, load_tensor_from_dicom

from src.labels.heads import TRAIN_HEADS

HEADS = TRAIN_HEADS
STUDY_RE = re.compile(r"^(\d+)")


def study_no_from_path(p: Path) -> int | None:
    m = STUDY_RE.match(p.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def index_dicoms(dicom_dir: Path, *, gangnam_only: bool = True) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    for p in dicom_dir.rglob("*.dcm"):
        s = study_no_from_path(p)
        if s is None:
            continue
        if gangnam_only and not (10000 <= s < 20000):
            continue
        out.setdefault(s, []).append(p)
    for s in out:
        out[s] = sorted(out[s])
    return out


def load_labels_csv(labels_csv: Path) -> dict[int, np.ndarray]:
    """Return study_no -> float array shape (num_heads,) with nan for missing."""
    labels: dict[int, np.ndarray] = {}
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(float(row["Study No."]))
            except (KeyError, ValueError):
                continue
            vals = []
            for h in HEADS:
                v = row.get(h, "").strip()
                if v == "":
                    vals.append(np.nan)
                else:
                    vals.append(float(v))
            labels[s] = np.array(vals, dtype=np.float32)
    return labels


def build_slice_samples(
    study_ids: list[int],
    dicom_index: dict[int, list[Path]],
    *,
    slice_mode: str = "all",
) -> list[tuple[int, Path]]:
    """Expand study IDs to (study_no, dcm_path) rows for the DataLoader."""
    if slice_mode not in ("all", "first"):
        raise ValueError(f"slice_mode must be 'all' or 'first', got {slice_mode!r}")
    samples: list[tuple[int, Path]] = []
    for study in study_ids:
        paths = dicom_index.get(study, [])
        if not paths:
            continue
        use_paths = paths[:1] if slice_mode == "first" else paths
        for p in use_paths:
            samples.append((study, p))
    return samples


def study_balanced_weights(samples: list[tuple[int, Path]]) -> list[float]:
    """Per-slice weights so each study contributes equally in expectation."""
    counts = Counter(study for study, _ in samples)
    return [1.0 / counts[study] for study, _ in samples]


class InfectivityDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[int, Path]],
        labels: dict[int, np.ndarray],
        *,
        image_size: int = 256,
        clahe: bool = True,
        clahe_clip_limit: float = 0.03,
        resize_mode: str = "stretch",
    ) -> None:
        self.samples = list(samples)
        self.labels = labels
        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.transform = build_transform(image_size, resize_mode=resize_mode)

    @classmethod
    def from_studies(
        cls,
        study_ids: list[int],
        dicom_index: dict[int, list[Path]],
        labels: dict[int, np.ndarray],
        *,
        slice_mode: str = "all",
        **kwargs,
    ) -> InfectivityDataset:
        samples = build_slice_samples(study_ids, dicom_index, slice_mode=slice_mode)
        return cls(samples, labels, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        study, dcm_path = self.samples[idx]
        x = load_tensor_from_dicom(
            dcm_path,
            self.transform,
            clahe=self.clahe,
            clahe_clip_limit=self.clahe_clip_limit,
        )
        y = self.labels.get(study, np.full(len(HEADS), np.nan, dtype=np.float32))
        mask = ~np.isnan(y)
        y_filled = np.nan_to_num(y, nan=0.0).astype(np.float32)
        return {
            "x": x,
            "y": torch.from_numpy(y_filled),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "study_no": study,
            "dcm_path": str(dcm_path),
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "study_no": [b["study_no"] for b in batch],
    }
