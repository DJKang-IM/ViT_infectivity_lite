# -*- coding: utf-8 -*-
"""Study registry from collected JSON sources (no CRAWLFIX CSV)."""
from __future__ import annotations

import json
import re
from pathlib import Path

# Study ID ranges by hospital
STUDY_ID_GANGNAM_MIN = 10000   # 강남성심병원 (included)
STUDY_ID_GANGNAM_MAX = 20000   # exclusive upper bound
# 나은병원 3xxxx — excluded from v1; separate X-ray recrawl project

STUDY_JSON_RE = re.compile(r"^(\d+)_")


def is_gangnam_study(study_no: int) -> bool:
    return STUDY_ID_GANGNAM_MIN <= study_no < STUDY_ID_GANGNAM_MAX


def study_from_json_path(p: Path) -> int | None:
    m = STUDY_JSON_RE.match(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_sputum_studies(sputum_dir: Path) -> dict[int, Path]:
    """study_no -> JSON path from [FINAL] SPUTUM DATA."""
    out: dict[int, Path] = {}
    for p in sputum_dir.glob("*.json"):
        if p.name.startswith("[") or p.name.startswith("_"):
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
            s = int(o["study_no"])
        except Exception:
            s = study_from_json_path(p)
        if s is None or not is_gangnam_study(s):
            continue
        out[s] = p
    return out


def load_ct_studies(ct_dir: Path) -> set[int]:
    """study_no set from CT reading collection JSON files."""
    studies: set[int] = set()
    for p in ct_dir.glob("*.json"):
        if not (p.name.endswith("_chest_ct.json") or p.name.endswith("_EXTERNAL.json")
                or p.name.endswith("_ct_reading.json")):
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
            s = int(o["study_no"])
        except Exception:
            s = study_from_json_path(p)
        if s is not None and is_gangnam_study(s):
            studies.add(s)
    return studies


def build_study_registry(
    sputum_dir: Path,
    ct_dir: Path,
    *,
    mode: str = "union",
    dicom_studies: set[int] | None = None,
) -> set[int]:
    """
    Build study ID registry from collected sources.

    mode:
      - union: sputum JSON ∪ CT JSON (label building cohort)
      - sputum: sputum JSON only (D1-D4 primary)
      - intersection: sputum ∩ CT ∩ dicom (if dicom_studies given)
    """
    sputum = set(load_sputum_studies(sputum_dir).keys())
    ct = load_ct_studies(ct_dir)
    m = (mode or "union").strip().lower()

    if m == "sputum":
        reg = sputum
    elif m == "intersection":
        reg = sputum & ct
        if dicom_studies is not None:
            reg &= dicom_studies
    else:
        reg = sputum | ct

    return reg
