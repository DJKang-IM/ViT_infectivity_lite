# -*- coding: utf-8 -*-
"""Run label scheme ablation: AFB variants x culture transforms (fold 0 screening)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[1]

AFB_SCHEMES = ["afb_grade_v1", "afb_binary", "afb_ordinal_5"]
CULTURE_TRANSFORMS = ["inv", "loginv", "rank"]


def main() -> None:
    py = sys.executable
    build = _V1_ROOT / "src" / "labels" / "build_graded_labels.py"
    train = _V1_ROOT / "src" / "train.py"

    for afb in AFB_SCHEMES:
        for cult in CULTURE_TRANSFORMS:
            tag = f"abl_{afb}_{cult}"
            labels_out = _V1_ROOT / "artifacts" / f"labels_{tag}.csv"
            print(f"\n========== {tag} ==========")
            subprocess.run([
                py, str(build),
                "--afb-scheme", afb,
                "--culture-transform", cult,
                "--out", str(labels_out),
                "--audit-out", str(_V1_ROOT / "artifacts" / f"labels_{tag}_audit.csv"),
            ], check=True, cwd=str(_V1_ROOT))

            subprocess.run([
                py, str(train),
                "--labels-csv", str(labels_out),
                "--tag", tag,
                "--fold", "0",
                "--epochs", "2",
            ], check=True, cwd=str(_V1_ROOT))

    # summary table
    rows = ["| scheme | macro_mse | macro_spearman | macro_auroc |", "|---|---:|---:|---:|"]
    for afb in AFB_SCHEMES:
        for cult in CULTURE_TRANSFORMS:
            tag = f"abl_{afb}_{cult}"
            metrics_path = _V1_ROOT / "artifacts" / f"v1_{tag}" / "metrics.json"
            if not metrics_path.exists():
                continue
            import json
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            fold0 = m["folds"][0] if m.get("folds") else {}
            rows.append(
                f"| {tag} | {fold0.get('macro_mse', 'nan'):.4f} | "
                f"{fold0.get('macro_spearman', 'nan'):.4f} | "
                f"{fold0.get('macro_auroc', 'nan'):.4f} |"
            )
    report = _V1_ROOT / "artifacts" / "ablation_summary.md"
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nWROTE {report}")


if __name__ == "__main__":
    main()
