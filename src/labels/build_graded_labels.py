# -*- coding: utf-8 -*-
"""Build study-level graded label CSV for Infectivity_ViT v1."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[2]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.labels.encoders import LabelSchemes, apply_rank_transform
from src.labels.heads import TRAIN_HEADS
from src.labels.parse_cavity import derive_d5, load_ct_blocks
from src.labels.parse_sputum import apply_culture_transform, parse_sputum_blocks
from src.labels.registry import build_study_registry, load_sputum_studies
from src.labels.utils import load_report_dates

NOTE_COLS = ["d1_note", "d2_note", "d3_note", "d4_note", "d7_note", "d5_source", "d5_note"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build graded D1-D5,D7 labels CSV")
    ap.add_argument("--sputum-dir", type=Path, default=Path(r"<REDACTED_PATH> SPUTUM DATA"))
    ap.add_argument("--ct-dir", type=Path, default=Path(r"<REDACTED_PATH> CT Reading Collection GUI"))
    ap.add_argument("--registry-mode", default="union", choices=["union", "sputum", "intersection"])
    ap.add_argument("--dicom-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=_V1_ROOT / "artifacts" / "labels_v1.csv")
    ap.add_argument("--audit-out", type=Path, default=_V1_ROOT / "artifacts" / "labels_v1_audit.csv")
    ap.add_argument("--cxr-mtl-out", type=Path, default=_V1_ROOT / "artifacts" / "labels_cxr_mtl.csv",
                    help="CXR MTL 5-head CSV (D1-D5 only, D2 binarized)")
    ap.add_argument("--afb-scheme", default="afb_polychoric_culture_v1")
    ap.add_argument("--pcr-scheme", default="pcr_soft_v1")
    ap.add_argument("--solid-transform", default="loginv", choices=["inv", "loginv", "rank"])
    ap.add_argument("--liquid-transform", default="twostep", choices=["inv", "loginv", "twostep", "rank"])
    ap.add_argument("--window-months", type=int, default=2)
    args = ap.parse_args()

    schemes = LabelSchemes(
        afb=args.afb_scheme,
        pcr=args.pcr_scheme,
        solid_transform=args.solid_transform,
        liquid_transform=args.liquid_transform,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rdates = load_report_dates()

    dicom_studies: set[int] | None = None
    if args.dicom_dir is not None and args.dicom_dir.is_dir():
        from src.data.dicom_dataset import index_dicoms
        dicom_studies = set(index_dicoms(args.dicom_dir).keys())

    registry = build_study_registry(
        args.sputum_dir, args.ct_dir, mode=args.registry_mode, dicom_studies=dicom_studies,
    )
    sputum_files = load_sputum_studies(args.sputum_dir)
    ctb = load_ct_blocks(args.ct_dir)

    parsed_by_study: dict[int, object] = {}
    solid_ttp: dict[int, int] = {}
    liquid_ttp: dict[int, int] = {}

    for s, p in sputum_files.items():
        if s not in registry:
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ps = parse_sputum_blocks(
            o.get("test_result_blocks", []), rdates.get(s), schemes, window_months=args.window_months,
        )
        parsed_by_study[s] = ps
        if ps.solid_ttp_days is not None:
            solid_ttp[s] = ps.solid_ttp_days
        if ps.liquid_ttp_days is not None:
            liquid_ttp[s] = ps.liquid_ttp_days

    solid_rank = apply_rank_transform(solid_ttp) if schemes.solid_transform == "rank" else {}
    liquid_rank = apply_rank_transform(liquid_ttp) if schemes.liquid_transform == "rank" else {}

    out_rows = [["Study No."] + TRAIN_HEADS + NOTE_COLS]
    audit_rows = [["Study No.", "issue", "detail"]]
    # CXR MTL 5-head table: D1..D5 with masks. Missing target -> -1, mask 0.
    cxr_header = ["Study No.", "D1", "D2", "D3", "D4", "D5",
                  "m1", "m2", "m3", "m4", "m5"]
    cxr_rows = [cxr_header]

    for s in sorted(registry):
        ps = parsed_by_study.get(s)
        if ps is not None:
            ps = apply_culture_transform(
                ps, schemes,
                solid_rank_score=solid_rank.get(s),
                liquid_rank_score=liquid_rank.get(s),
            )
            d1, d2, d3, d4, d7 = ps.d1, ps.d2, ps.d3, ps.d4, ps.d7
            notes = [ps.d1_note, ps.d2_note, ps.d3_note, ps.d4_note, ps.d7_note]
            if ps.d1 is None:
                audit_rows.append([s, "d1_missing", "sputum parsed but no AFB in window"])
        elif s in sputum_files:
            d1 = d2 = d3 = d4 = d7 = None
            notes = ["sputum_empty_parse"] * 5
            audit_rows.append([s, "sputum_empty_parse", ""])
        else:
            d1 = d2 = d3 = d4 = d7 = None
            notes = ["no_sputum_json"] * 5
            if args.registry_mode != "sputum":
                audit_rows.append([s, "no_sputum_json", "CT-only study in registry"])

        d5, d5_src, d5_note = derive_d5(ctb.get(s, []), rdates.get(s))
        if d5 is None:
            d5_src = "missing"
            audit_rows.append([s, "d5_missing", d5_note])

        def fmt(v: float | None) -> str:
            if v is None:
                return ""
            return f"{v:.6g}"

        out_rows.append([
            s, fmt(d1), fmt(d2), fmt(d3), fmt(d4), fmt(d5), fmt(d7),
            notes[0], notes[1], notes[2], notes[3], notes[4], d5_src, d5_note,
        ])

        # --- CXR MTL 5-head row (D1..D5) ---
        # D2 binarized: not-detected(0) stays 0; indeterminate(0.5)/positive(1) -> 1
        d2_bin = None if d2 is None else (1.0 if d2 > 0 else 0.0)
        cxr_vals = [d1, d2_bin, d3, d4, d5]

        def cell(v: float | None) -> str:
            return "-1" if v is None else f"{v:.6g}"

        masks = [0 if v is None else 1 for v in cxr_vals]
        cxr_rows.append([s] + [cell(v) for v in cxr_vals] + masks)

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(out_rows)
    with args.audit_out.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(audit_rows)
    args.cxr_mtl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.cxr_mtl_out.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(cxr_rows)

    n = len([r for r in out_rows[1:]])
    labeled = {h: sum(1 for r in out_rows[1:] if r[TRAIN_HEADS.index(h) + 1] != "") for h in TRAIN_HEADS}
    print(f"registry ({args.registry_mode}): {n} studies")
    print(f"  sputum JSON: {len(sputum_files)} | CT: {len(ctb)}")
    for h in TRAIN_HEADS:
        print(f"  {h} labeled: {labeled[h]}")
    cxr_heads = ["D1", "D2", "D3", "D4", "D5"]
    cxr_labeled = {h: sum(1 for r in cxr_rows[1:] if r[cxr_header.index(h)] != "-1")
                   for h in cxr_heads}
    print("CXR MTL table:")
    for h in cxr_heads:
        print(f"  {h} labeled: {cxr_labeled[h]}")
    print(f"WROTE {args.out}")
    print(f"WROTE {args.audit_out}")
    print(f"WROTE {args.cxr_mtl_out}")


if __name__ == "__main__":
    main()
