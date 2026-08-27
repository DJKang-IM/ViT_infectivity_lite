# -*- coding: utf-8 -*-
"""PyTorch Dataset for CXR Multi-task Learning (D1..D5).

Sampling is per-image (each JPEG is one sample) but labels are shared at the
study level. Missing targets carry value -1 and mask 0 so the loss is skipped.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cxr_jpeg_index import load_manifest
from src.data.preprocess import build_transform, jpeg_to_float01

HEADS = ["D1", "D2", "D3", "D4", "D5"]
MASK_COLS = ["m1", "m2", "m3", "m4", "m5"]


def load_cxr_labels(labels_csv: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """study_no -> (y[5], mask[5]) float32."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            s = int(row["Study No."])
            y = np.array([float(row[h]) for h in HEADS], dtype=np.float32)
            m = np.array([float(row[c]) for c in MASK_COLS], dtype=np.float32)
            # defensive: force masked-out targets to 0.0 (not -1) for the model
            y = np.where(m > 0, y, 0.0).astype(np.float32)
            out[s] = (y, m)
    return out


def load_split(split_json: Path, fold: str) -> list[int]:
    obj = json.loads(split_json.read_text(encoding="utf-8"))
    if fold not in ("train", "val", "test"):
        raise ValueError(f"fold must be train/val/test, got {fold!r}")
    return list(obj[fold])


class CXRMultiTaskDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[int, str]],
        labels: dict[int, tuple[np.ndarray, np.ndarray]],
        *,
        image_size: int = 256,
        clahe: bool = True,
        clahe_clip_limit: float = 0.03,
        resize_mode: str = "letterbox",
        brightness_norm: bool = True,
        cache_dir: Path | None = None,
        cache_max_side: int = 512,
    ) -> None:
        self.samples = list(samples)
        self.labels = labels
        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.brightness_norm = brightness_norm
        self.transform = build_transform(image_size, resize_mode=resize_mode)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_max_side = int(cache_max_side)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # cache key salt (params that change the CLAHE'd array)
        self._cache_tag = (
            f"c{int(clahe)}_l{clahe_clip_limit}_b{int(brightness_norm)}"
            f"_m{self.cache_max_side}"
        )

    def _cache_path(self, jpeg_path: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.md5((jpeg_path + "|" + self._cache_tag).encode()).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _clahe_array(self, jpeg_path: str) -> np.ndarray:
        cp = self._cache_path(jpeg_path)
        if cp is not None and cp.exists():
            return np.load(cp).astype(np.float32) / 255.0
        # downscale BEFORE CLAHE: equalize_adapthist cost scales with pixel count
        arr = jpeg_to_float01(
            Path(jpeg_path),
            clahe=self.clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            brightness_norm=self.brightness_norm,
            resize_max_side=self.cache_max_side,
        )
        if cp is not None:
            u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            np.save(cp, u8)
        return arr.astype(np.float32)

    @classmethod
    def from_manifest(
        cls,
        study_ids: list[int],
        manifest_csv: Path,
        labels: dict[int, tuple[np.ndarray, np.ndarray]],
        **kwargs,
    ) -> "CXRMultiTaskDataset":
        manifest = load_manifest(manifest_csv)
        wanted = set(study_ids)
        samples: list[tuple[int, str]] = []
        for s in study_ids:
            for row in manifest.get(s, []):
                if s in labels:
                    samples.append((s, row["jpeg_path"]))
        _ = wanted
        return cls(samples, labels, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        study, jpeg_path = self.samples[idx]
        arr = self._clahe_array(jpeg_path)
        x = self.transform(arr)
        y, mask = self.labels[study]
        return {
            "x": x,
            "y": torch.from_numpy(y.copy()),
            "mask": torch.from_numpy(mask.copy()),
            "study_no": study,
            "jpeg_path": jpeg_path,
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "study_no": [b["study_no"] for b in batch],
    }
