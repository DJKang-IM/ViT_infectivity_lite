# -*- coding: utf-8 -*-
"""Pre-compute CLAHE arrays for all manifest images into the disk cache.

CLAHE (equalize_adapthist) is the slow part of the pipeline; computing it once
here means every training epoch (and both baselines) reads cheap .npy files.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[1]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.data.cxr_multitask_dataset import CXRMultiTaskDataset

MANIFEST_DEFAULT = _V1_ROOT / "artifacts" / "cxr_image_manifest.csv"


def _worker(args) -> int:
    jpeg_path, clip = args
    ds = CXRMultiTaskDataset(
        [], {}, image_size=256, clahe=True, clahe_clip_limit=clip,
        resize_mode="letterbox", brightness_norm=True,
        cache_dir=_V1_ROOT / "artifacts" / "clahe_cache" / f"clip{clip}",
        cache_max_side=512,
    )
    ds._clahe_array(jpeg_path)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-warm CLAHE disk cache")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    ap.add_argument("--clip", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        paths = [row["jpeg_path"] for row in csv.DictReader(f)]
    print(f"images to warm: {len(paths)} (workers={args.workers}, clip={args.clip})",
          flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, (p, args.clip)) for p in paths]
        for fut in as_completed(futs):
            done += fut.result()
            if done % 500 == 0:
                print(f"  {done}/{len(paths)}", flush=True)
    print(f"DONE {done}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
