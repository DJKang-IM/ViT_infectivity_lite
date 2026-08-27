# -*- coding: utf-8 -*-
"""Audit raw sputum JSON vs graded label encodings (run BEFORE ViT training)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

_V1_ROOT = Path(__file__).resolve().parents[2]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.labels.encoders import (
    AFB_ENCODERS,
    CULTURE_TRANSFORMS,
    PCR_ENCODERS,
    encode_afb_grade_v1,
)
from src.labels.utils import AFB_STAIN_RE, ORDER_RE, REPORT_RE, TB_NAME, is_sputum_specimen, norm

FINAL_DEFAULT = Path(r"<REDACTED_PATH> SPUTUM DATA")


def cult_pos(line: str) -> bool:
    tl = line.lower()
    if "no growth" in tl or "no afb" in tl:
        return False
    if re.search(r"non[-\s]?tuberculosis|nontuberculous|비결핵|\bntm\b", line, re.I):
        return False
    return ("isolated" in tl) or ("comp" in tl and "tuberculosis" in tl) \
        or ("가능성 높음" in line) or ("positive" in tl)


def audit_afb(sputum_dir: Path) -> dict:
    raw = Counter()
    by_encoder: dict[str, Counter] = {k: Counter() for k in AFB_ENCODERS}
    examples: dict[str, list] = defaultdict(list)

    for p in sputum_dir.glob("*.json"):
        if p.name.startswith("[") or p.name.startswith("_"):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        for b in o.get("test_result_blocks", []):
            for line in str(b.get("test_result", "")).splitlines():
                m = AFB_STAIN_RE.search(line)
                if not m:
                    continue
                v = m.group(1).strip()
                raw[v] += 1
                if len(examples[v]) < 2:
                    examples[v].append(o.get("study_no"))
                for name, fn in AFB_ENCODERS.items():
                    enc = fn(v)
                    by_encoder[name][enc if enc is not None else "MISSING"] += 1

    return {"raw": raw, "by_encoder": by_encoder, "examples": dict(examples)}


def audit_pcr(sputum_dir: Path) -> dict:
    cats = Counter()
    by_encoder: dict[str, Counter] = {k: Counter() for k in PCR_ENCODERS}
    for p in sputum_dir.glob("*.json"):
        if p.name.startswith("[") or p.name.startswith("_"):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        for b in o.get("test_result_blocks", []):
            name = norm(str(b.get("test_name", "")))
            if not TB_NAME.search(name) or not is_sputum_specimen(name):
                continue
            text = str(b.get("test_result", ""))
            tl = text.lower()
            if "indeterminate" in tl:
                cats["indeterminate"] += 1
            elif "not detected" in tl or ("negative" in tl and "positive" not in tl):
                cats["negative"] += 1
            elif "positive" in tl or "detected" in tl:
                cats["positive"] += 1
            else:
                cats["other"] += 1
            for enc_name, fn in PCR_ENCODERS.items():
                v = fn(text)
                by_encoder[enc_name][v if v is not None else "MISSING"] += 1
    return {"categories": cats, "by_encoder": by_encoder}


def audit_culture_ttp(sputum_dir: Path) -> dict:
    solid_days: list[int] = []
    liquid_days: list[int] = []
    pos_line_samples: list[tuple] = []

    for p in sputum_dir.glob("*.json"):
        if p.name.startswith("[") or p.name.startswith("_"):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        for b in o.get("test_result_blocks", []):
            text = str(b.get("test_result", ""))
            if "afb stain" not in text.lower():
                continue
            om = ORDER_RE.search(text)
            if not om:
                continue
            order_d = om.group(1)
            lines = text.splitlines()
            cur = om.group(2)
            for i, line in enumerate(lines):
                rm = REPORT_RE.search(line)
                if rm:
                    cur = rm.group(1).replace("-", "")
                ls = line.strip()
                is_liquid = "액체배지" in ls or "액체배양" in ls or "(액체)" in ls
                is_solid = "고체배지" in ls or "고체배양" in ls or "(고체)" in ls
                if not (is_liquid or is_solid) or not cult_pos(ls):
                    continue
                pos_date = cur
                for j in range(i, -1, -1):
                    rm2 = REPORT_RE.search(lines[j])
                    if rm2:
                        pos_date = rm2.group(1).replace("-", "")
                        break
                try:
                    d0 = date(int(order_d[:4]), int(order_d[4:6]), int(order_d[6:8]))
                    d1 = date(int(pos_date[:4]), int(pos_date[4:6]), int(pos_date[6:8]))
                    days = (d1 - d0).days
                except (ValueError, TypeError):
                    continue
                if days < 0:
                    continue
                if is_liquid:
                    liquid_days.append(days)
                else:
                    solid_days.append(days)
                if len(pos_line_samples) < 10:
                    pos_line_samples.append(("liquid" if is_liquid else "solid", days, ls[:100]))

    def summarize(days: list[int]) -> dict:
        if not days:
            return {"n": 0}
        s = sorted(days)
        n = len(s)
        return {
            "n": n,
            "min": s[0],
            "p25": s[n // 4],
            "median": s[n // 2],
            "p75": s[3 * n // 4],
            "max": s[-1],
            "top_days": Counter(days).most_common(10),
        }

    transforms = {}
    for tname, fn in CULTURE_TRANSFORMS.items():
        transforms[tname] = {
            "solid": {d: round(fn(d), 4) for d in [1, 3, 7, 14, 18, 21, 30, 60]},
            "liquid": {d: round(fn(d), 4) for d in [0, 1, 2, 3, 7, 14, 22, 30]},
        }

    return {
        "solid": summarize(solid_days),
        "liquid": summarize(liquid_days),
        "transform_preview": transforms,
        "samples": pos_line_samples,
    }


def audit_built_csv(labels_csv: Path) -> dict:
    if not labels_csv.exists():
        return {}
    out = {}
    rows = list(csv.DictReader(labels_csv.open(encoding="utf-8-sig")))
    for h in ["D1", "D2", "D3", "D4", "D5"]:
        vals = [float(r[h]) for r in rows if r.get(h, "").strip()]
        out[h] = {
            "n": len(vals),
            "unique": len(set(round(v, 4) for v in vals)),
            "top": Counter(round(v, 4) for v in vals).most_common(10),
        }
    return out


def write_report(out_path: Path, sections: dict) -> None:
    lines = ["# Label Audit Report\n"]
    lines.append("> Run BEFORE ViT training. Goal: validate continuous label design.\n")

    lines.append("## 1. AFB (D1) — raw strings in JSON\n")
    raw = sections["afb"]["raw"]
    lines.append(f"Unique raw strings: **{len(raw)}** | Total AFB stain lines: **{sum(raw.values())}**\n")
    lines.append("| Count | Raw text | Example study |")
    lines.append("|---:|---|---|")
    for v, c in raw.most_common():
        ex = sections["afb"]["examples"].get(v, [])
        lines.append(f"| {c} | `{v}` | {ex} |")

    lines.append("\n### AFB encoding comparison\n")
    lines.append("| Scheme | Score | Count |")
    lines.append("|---|---:|---:|")
    for scheme, ctr in sections["afb"]["by_encoder"].items():
        for score, n in sorted(ctr.items(), key=lambda x: (str(x[0]) == "MISSING", x[0])):
            lines.append(f"| {scheme} | {score} | {n} |")

    lines.append("\n## 2. TB-PCR (D2)\n")
    pcr = sections["pcr"]
    lines.append("| Category | Blocks |")
    lines.append("|---|---:|")
    for k, v in pcr["categories"].most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("\n**Note:** `pcr_soft_v1` uses 0.5 for indeterminate — check if any exist in cohort.\n")
    for scheme, ctr in pcr["by_encoder"].items():
        lines.append(f"\n**{scheme}:** " + ", ".join(f"{k}={v}" for k, v in sorted(ctr.items())))

    lines.append("\n## 3. Culture TTP (D3 solid / D4 liquid)\n")
    cult = sections["culture"]
    for kind in ("solid", "liquid"):
        s = cult[kind]
        lines.append(f"\n### {kind}\n")
        if s.get("n", 0) == 0:
            lines.append("No positives parsed.\n")
            continue
        lines.append(
            f"n={s['n']} | min={s['min']} | p25={s['p25']} | median={s['median']} | "
            f"p75={s['p75']} | max={s['max']}\n"
        )
        lines.append("Top day counts: " + ", ".join(f"{d}d×{c}" for d, c in s["top_days"][:8]))

    lines.append("\n### Transform preview (days → score)\n")
    for tname, prev in cult["transform_preview"].items():
        lines.append(f"\n**{tname}**")
        lines.append("| days | solid score | liquid score |")
        lines.append("|---:|---:|---:|")
        all_days = sorted(set(prev["solid"]) | set(prev["liquid"]))
        for d in all_days:
            lines.append(
                f"| {d} | {prev['solid'].get(d, '-')} | {prev['liquid'].get(d, '-')} |"
            )

    if sections.get("built"):
        lines.append("\n## 4. Current labels_v1.csv (study-level, after window)\n")
        for h, info in sections["built"].items():
            lines.append(f"\n**{h}**: n={info['n']} unique={info['unique']}")
            lines.append("Top scores: " + ", ".join(f"{s}×{c}" for s, c in info["top"]))

    lines.append("\n## 5. Open decisions (before ViT)\n")
    lines.append("""
1. **AFB trace** (`Doubtful Repeat test (Trace)`): 0.125 vs 0.25 vs merge to binary-positive?
2. **PCR 0.5 rule**: cohort has **0 indeterminate** blocks — rule is ready but unused until data appears.
3. **Culture transform**: liquid median ~3d (scores cluster near 1.0 under loginv); solid median ~18d (scores ~0.35).
   - `inv` vs `loginv` vs `rank` vs capped linear — compare Spearman vs clinical intuition.
4. **ViT is step 2** — freeze label scheme first, then train.
""")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sputum-dir", type=Path, default=FINAL_DEFAULT)
    ap.add_argument("--labels-csv", type=Path, default=_V1_ROOT / "artifacts" / "labels_v1.csv")
    ap.add_argument("--out", type=Path, default=_V1_ROOT / "artifacts" / "label_audit_report.md")
    args = ap.parse_args()

    sections = {
        "afb": audit_afb(args.sputum_dir),
        "pcr": audit_pcr(args.sputum_dir),
        "culture": audit_culture_ttp(args.sputum_dir),
        "built": audit_built_csv(args.labels_csv),
    }
    write_report(args.out, sections)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
