# -*- coding: utf-8 -*-
"""Run AFB layer-2 parallel calibration and write comparison artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[1]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.labels.afb_layer2 import (  # noqa: E402
    NPZ_DEFAULT,
    format_comparison_md,
    run_all_methods,
)

LABELS_DEFAULT = _V1_ROOT / "artifacts" / "labels_v1.csv"
OUT_JSON_DEFAULT = _V1_ROOT / "artifacts" / "afb_layer2_comparison.json"
OUT_MD_DEFAULT = _V1_ROOT / "artifacts" / "afb_layer2_comparison.md"


def main() -> None:
    ap = argparse.ArgumentParser(description="AFB layer-2 parallel score comparison")
    ap.add_argument("--labels", type=Path, default=LABELS_DEFAULT)
    ap.add_argument("--npz", type=Path, default=NPZ_DEFAULT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON_DEFAULT)
    ap.add_argument("--out-md", type=Path, default=OUT_MD_DEFAULT)
    args = ap.parse_args()

    cmp = run_all_methods(args.labels, npz_path=args.npz)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(cmp.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.out_md.write_text(format_comparison_md(cmp), encoding="utf-8")

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    for m, ok in cmp.monotone.items():
        print(f"  monotone {m}: {'OK' if ok else 'VIOLATION'}")
    if "error" in cmp.method_meta.get("pls_cxr", {}):
        print(f"  pls_cxr: SKIPPED ({cmp.method_meta['pls_cxr']['error']})")


if __name__ == "__main__":
    main()
