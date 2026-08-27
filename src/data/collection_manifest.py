# -*- coding: utf-8 -*-
"""DICOM collection manifest: labeled studies vs imaging availability."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[2]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.data.dicom_dataset import index_dicoms, load_labels_csv
from src.labels.registry import load_ct_studies, load_sputum_studies


def count_labeled_heads(row: dict[str, str], heads: list[str]) -> int:
    n = 0
    for h in heads:
        v = row.get(h, "").strip()
        if v:
            n += 1
    return n


def build_manifest(
    *,
    labels_csv: Path,
    work_dir: Path,
    raw_dir: Path,
    sputum_dir: Path,
    ct_dir: Path,
) -> list[dict[str, str]]:
    labels = load_labels_csv(labels_csv)
    work_idx = index_dicoms(work_dir)
    raw_idx = index_dicoms(raw_dir)
    sputum = load_sputum_studies(sputum_dir)
    ct = load_ct_studies(ct_dir)

    rows_by_study: dict[int, dict[str, str]] = {}
    with labels_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        field_heads = [h for h in reader.fieldnames or [] if h.startswith("D")]
        for row in reader:
            try:
                study = int(float(row["Study No."]))
            except (KeyError, ValueError):
                continue
            rows_by_study[study] = row

    studies = sorted(rows_by_study.keys())
    out: list[dict[str, str]] = []
    for study in studies:
        n_work = len(work_idx.get(study, []))
        n_raw = len(raw_idx.get(study, []))
        if n_work > 0:
            status = "ready"
        elif n_raw > 0:
            status = "pending_export"
        else:
            status = "pending_collect"

        row = rows_by_study[study]
        out.append({
            "study_no": str(study),
            "status": status,
            "n_slices_work": str(n_work),
            "n_slices_raw": str(n_raw),
            "has_sputum": "1" if study in sputum else "0",
            "has_ct": "1" if study in ct else "0",
            "n_labeled_heads": str(count_labeled_heads(row, field_heads)),
            "note": "",
        })
    return out


def write_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "study_no", "status", "n_slices_work", "n_slices_raw",
        "has_sputum", "has_ct", "n_labeled_heads", "note",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DICOM collection manifest from labels + dirs")
    ap.add_argument("--labels-csv", type=Path, default=_V1_ROOT / "artifacts" / "labels_v1.csv")
    ap.add_argument("--work-dir", type=Path, default=Path(r"<REDACTED_PATH> ViT_Infectivity"))
    ap.add_argument("--raw-dir", type=Path, default=Path(r"<REDACTED_PATH> CXR_Active Image"))
    ap.add_argument("--sputum-dir", type=Path, default=Path(r"<REDACTED_PATH> SPUTUM DATA"))
    ap.add_argument("--ct-dir", type=Path, default=Path(r"<REDACTED_PATH> CT Reading Collection GUI"))
    ap.add_argument("--out", type=Path, default=_V1_ROOT / "artifacts" / "dicom_collection_manifest.csv")
    args = ap.parse_args()

    rows = build_manifest(
        labels_csv=args.labels_csv,
        work_dir=args.work_dir,
        raw_dir=args.raw_dir,
        sputum_dir=args.sputum_dir,
        ct_dir=args.ct_dir,
    )
    write_manifest(rows, args.out)
    stats = summarize(rows)
    print(f"WROTE {args.out} ({len(rows)} studies)")
    for k in ("ready", "pending_export", "pending_collect"):
        print(f"  {k}: {stats.get(k, 0)}")


if __name__ == "__main__":
    main()
