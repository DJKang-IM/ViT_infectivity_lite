# -*- coding: utf-8 -*-
"""Shared date/window helpers for label parsing."""
from __future__ import annotations

import calendar
import glob
import re
from datetime import date, datetime
from pathlib import Path

ORDER_RE = re.compile(r"처방일/보고일】\s*(\d{8})\s*/\s*(\d{8})")
REPORT_RE = re.compile(r">>\s*(?:중간|최종)검사결과\s*\((\d{4}-\d{2}-\d{2})")
AFB_STAIN_RE = re.compile(r"AFB\s*stain[^:\n]*:\s*([^\r\n]+)", re.I)
TB_NAME = re.compile(r"mycobacterium\s+tuberculosis\s*pcr|\btb\s*pcr\b", re.I)
RIF_NAME = re.compile(
    r"rifampin\s+resistance|rif\s+resistance|rifampicin\s+resistance",
    re.I,
)
SPUTUM_RE = re.compile(r"sputum|객담", re.I)
NTM_CULT = re.compile(r"non[-\s]?tuberculosis|nontuberculous|비결핵|\bntm\b", re.I)


def norm(s: str) -> str:
    return " ".join((s or "").split()).lower().replace("<", "〈").replace(">", "〉")


def is_sputum_specimen(name: str) -> bool:
    return bool(SPUTUM_RE.search(norm(str(name))))


def add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def parse_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if re.fullmatch(r"\d{8}", s):
        try:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            return None
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    return None


def in_window(d: date | None, anchor: date | None, lo_m: int, hi_m: int) -> bool:
    if d is None or anchor is None:
        return False
    return add_months(anchor, lo_m) <= d <= add_months(anchor, hi_m)


def load_report_dates() -> dict[int, date]:
    """Load study_no -> 신고일자 from FINAL META xlsx."""
    try:
        from python_calamine import CalamineWorkbook
    except ImportError as e:
        raise RuntimeError(
            "python-calamine required for anchor dates. pip install python-calamine"
        ) from e

    cands = [g for g in glob.glob(r"<REDACTED_PATH>") if "260509" in g]
    if not cands:
        raise FileNotFoundError("Could not find *260509*FINAL*META*.xlsx on D:\\")
    xlsx = cands[0]
    wb = CalamineWorkbook.from_path(xlsx)
    rows = wb.get_sheet_by_index(0).to_python()
    hdr = rows[0]
    ci_s = hdr.index("Study No.") if "Study No." in hdr else 4
    ci_d = hdr.index("신고일자") if "신고일자" in hdr else 1
    out: dict[int, date] = {}
    for r in rows[1:]:
        if ci_s >= len(r) or r[ci_s] in (None, ""):
            continue
        try:
            s = int(float(r[ci_s]))
        except (ValueError, TypeError):
            continue
        pd_ = parse_date(r[ci_d] if ci_d < len(r) else None)
        if pd_:
            out[s] = pd_
    return out


def cell_str(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", ""):
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s
