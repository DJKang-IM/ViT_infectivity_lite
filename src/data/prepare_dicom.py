# -*- coding: utf-8 -*-
"""Copy gangnam (1xxxx) DICOM from RAW to [260626] ViT_Infectivity.

ViT v1 reads labels from CSV, not DICOM private tags — this step only
organizes imaging data (exclude 나은 3xxxx).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[2]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.labels.registry import is_gangnam_study

STUDY_RE = re.compile(r"^(\d+)")


def study_from_dcm_name(name: str) -> int | None:
    m = STUDY_RE.match(Path(name).stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(r"<REDACTED_PATH> CXR_Active Image"))
    ap.add_argument("--dst", type=Path, default=Path(r"<REDACTED_PATH> ViT_Infectivity"))
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="hardlink")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise FileNotFoundError(f"Source not found: {args.src}")

    copied = skipped_other = skipped_exists = 0
    studies: set[int] = set()

    for src in sorted(args.src.rglob("*.dcm")):
        s = study_from_dcm_name(src.name)
        if s is None or not is_gangnam_study(s):
            skipped_other += 1
            continue
        dst = args.dst / src.name
        studies.add(s)
        if dst.exists():
            skipped_exists += 1
            continue
        if not args.dry_run:
            link_or_copy(src, dst, args.mode)
        copied += 1

    print(f"src: {args.src}")
    print(f"dst: {args.dst}")
    print(f"mode: {args.mode} dry_run={args.dry_run}")
    print(f"studies (1xxxx): {len(studies)}")
    print(f"files prepared: {copied} | already present: {skipped_exists} | excluded (non-1xxxx): {skipped_other}")


if __name__ == "__main__":
    main()
