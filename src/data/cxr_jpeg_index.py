# -*- coding: utf-8 -*-
"""Index collected CXR JPEG files into a study-linked manifest.

Filenames look like: ``{study_no}_{patient_reg_no}_{view...}_{YYYY-MM-DD}.jpg``
e.g. ``10001_200143953_Chest_AP_2020-05-23.jpg``.

Only Gangnam studies (10000 <= study_no < 20000) are kept; Naeun (3xxxx) is
excluded (separate project).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from src.labels.registry import is_gangnam_study

_V1_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# study _ reg _ <view...> _ date.ext  (date = YYYY-MM-DD, optional trailing suffix)
NAME_RE = re.compile(
    r"^(?P<study>\d+)_(?P<reg>\d+)_(?P<view>.+)_(?P<date>\d{4}-\d{2}-\d{2})",
)


def parse_jpeg_name(name: str) -> dict | None:
    stem = name.rsplit(".", 1)[0]
    m = NAME_RE.match(stem)
    if not m:
        return None
    return {
        "study_no": int(m.group("study")),
        "patient_reg_no": m.group("reg"),
        "view": m.group("view"),
        "exam_date": m.group("date"),
    }


def build_manifest(
    jpeg_dir: Path,
    *,
    gangnam_only: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    stats = {"scanned": 0, "unparsed": 0, "non_gangnam": 0, "kept": 0}
    for p in jpeg_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stats["scanned"] += 1
        meta = parse_jpeg_name(p.name)
        if meta is None:
            stats["unparsed"] += 1
            continue
        if gangnam_only and not is_gangnam_study(meta["study_no"]):
            stats["non_gangnam"] += 1
            continue
        rows.append({
            "study_no": meta["study_no"],
            "patient_reg_no": meta["patient_reg_no"],
            "jpeg_path": str(p.resolve()),
            "exam_date": meta["exam_date"],
            "view": meta["view"],
            "filename": p.name,
        })
        stats["kept"] += 1
    rows.sort(key=lambda r: (r["study_no"], r["exam_date"], r["filename"]))
    return rows, stats


def load_manifest(manifest_csv: Path) -> dict[int, list[dict]]:
    """study_no -> list of image row dicts."""
    out: dict[int, list[dict]] = {}
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            s = int(row["study_no"])
            out.setdefault(s, []).append(row)
    return out


FIELDS = ["study_no", "patient_reg_no", "jpeg_path", "exam_date", "view", "filename"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Index CXR JPEG files -> manifest CSV")
    ap.add_argument("--jpeg-dir", type=Path,
                    default=Path(r"<REDACTED_PATH> Collecting Folder (download)"))
    ap.add_argument("--out", type=Path,
                    default=_V1_ROOT / "artifacts" / "cxr_image_manifest.csv")
    ap.add_argument("--all-hospitals", action="store_true",
                    help="disable Gangnam-only filter")
    args = ap.parse_args()

    rows, stats = build_manifest(args.jpeg_dir, gangnam_only=not args.all_hospitals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_studies = len({r["study_no"] for r in rows})
    print(f"scanned={stats['scanned']} unparsed={stats['unparsed']} "
          f"non_gangnam={stats['non_gangnam']} kept={stats['kept']}")
    print(f"studies with >=1 image: {n_studies}")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    import sys
    if str(_V1_ROOT) not in sys.path:
        sys.path.insert(0, str(_V1_ROOT))
    main()
