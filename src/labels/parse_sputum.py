# -*- coding: utf-8 -*-
"""Parse sputum JSON blocks into raw D1–D4, D7 graded label components."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .encoders import LabelSchemes
from .utils import (
    AFB_STAIN_RE,
    NTM_CULT,
    ORDER_RE,
    REPORT_RE,
    RIF_NAME,
    TB_NAME,
    in_window,
    is_sputum_specimen,
    norm,
    parse_date,
)


def cult_is_positive(line: str) -> bool:
    tl = line.lower()
    if "no growth" in tl or "no afb" in tl:
        return False
    if NTM_CULT.search(line):
        return False
    return ("isolated" in tl) or ("comp" in tl and "tuberculosis" in tl) \
        or ("가능성 높음" in line) or ("positive" in tl)


@dataclass
class HeadState:
    afb_values: list[float] = field(default_factory=list)
    pcr_values: list[float] = field(default_factory=list)
    rif_values: list[float] = field(default_factory=list)
    solid_neg: bool = False
    liquid_neg: bool = False
    solid_pos_days: list[int] = field(default_factory=list)
    liquid_pos_days: list[int] = field(default_factory=list)


@dataclass
class ParsedSputum:
    d1: float | None = None
    d2: float | None = None
    d3: float | None = None
    d4: float | None = None
    d7: float | None = None
    d1_note: str = ""
    d2_note: str = ""
    d3_note: str = ""
    d4_note: str = ""
    d7_note: str = ""
    solid_ttp_days: int | None = None
    liquid_ttp_days: int | None = None


def _parse_block_order_dates(text: str) -> tuple[date | None, date | None]:
    om = ORDER_RE.search(text)
    if not om:
        return None, None
    return parse_date(om.group(1)), parse_date(om.group(2))


def _culture_pos_days_in_line(
    lines: list[str],
    line_idx: int,
    cur_report: date | None,
    order_date: date | None,
) -> int | None:
    ls = lines[line_idx].strip()
    if not cult_is_positive(ls):
        return None
    pos_date = cur_report
    for j in range(line_idx, -1, -1):
        rm = REPORT_RE.search(lines[j])
        if rm:
            pos_date = parse_date(rm.group(1))
            break
    if pos_date is None or order_date is None:
        return None
    days = (pos_date - order_date).days
    return days if days >= 0 else None


def parse_sputum_blocks(
    blocks: list[dict],
    anchor: date | None,
    schemes: LabelSchemes,
    *,
    window_months: int = 2,
) -> ParsedSputum:
    st = HeadState()
    afb_fn = schemes.afb_fn()
    pcr_fn = schemes.pcr_fn()
    lo_m, hi_m = -window_months, window_months

    for b in blocks:
        name = norm(str(b.get("test_name", "")))
        text = str(b.get("test_result", ""))
        order_d, report_d = _parse_block_order_dates(text)

        def ev(cur: date | None) -> bool:
            ds = [d for d in (order_d, report_d, cur) if d is not None]
            return any(in_window(d, anchor, lo_m, hi_m) for d in ds)

        # D7 — RIF resistance PCR (Expert); same soft rule as D2
        if RIF_NAME.search(name):
            if not is_sputum_specimen(name):
                continue
            if ev(report_d):
                rv = pcr_fn(text)
                if rv is not None:
                    st.rif_values.append(rv)
            continue

        # D2 — TB-PCR (exclude RIF blocks matched above)
        if TB_NAME.search(name):
            if not is_sputum_specimen(name):
                continue
            if ev(report_d):
                pv = pcr_fn(text)
                if pv is not None:
                    st.pcr_values.append(pv)
            continue

        is_combo = ("afb stain" in name and "culture" in name) or "고체+액체" in name
        if not is_combo and "afb stain" not in text.lower():
            continue
        if not is_sputum_specimen(name):
            continue

        cur = report_d
        lines = text.splitlines()
        for i, line in enumerate(lines):
            rm = REPORT_RE.search(line)
            if rm:
                cur = parse_date(rm.group(1))
                continue
            ls = line.strip()
            if not ls:
                continue

            am = AFB_STAIN_RE.search(ls)
            if am:
                val = afb_fn(am.group(1))
                if val is not None and ev(cur):
                    st.afb_values.append(val)
                continue

            is_liquid = "액체배지" in ls or "액체배양" in ls or "(액체)" in ls
            is_solid = "고체배지" in ls or "고체배양" in ls or "(고체)" in ls
            if not (is_liquid or is_solid):
                continue
            if not ev(cur):
                continue

            if cult_is_positive(ls):
                days = _culture_pos_days_in_line(lines, i, cur, order_d)
                if days is not None:
                    if is_liquid:
                        st.liquid_pos_days.append(days)
                    else:
                        st.solid_pos_days.append(days)
            else:
                if is_liquid:
                    st.liquid_neg = True
                else:
                    st.solid_neg = True

    out = ParsedSputum()
    if st.afb_values:
        out.d1 = max(st.afb_values)
        out.d1_note = f"afb_max={out.d1}"
    if st.pcr_values:
        out.d2 = max(st.pcr_values)
        out.d2_note = f"pcr_max={out.d2}"
    if st.rif_values:
        out.d7 = max(st.rif_values)
        out.d7_note = f"rif_max={out.d7}"
    if st.solid_pos_days:
        out.solid_ttp_days = min(st.solid_pos_days)
        out.d3_note = f"solid_ttp={out.solid_ttp_days}d"
    elif st.solid_neg:
        out.d3 = 0.0
        out.d3_note = "solid_neg"
    if st.liquid_pos_days:
        out.liquid_ttp_days = min(st.liquid_pos_days)
        out.d4_note = f"liquid_ttp={out.liquid_ttp_days}d"
    elif st.liquid_neg:
        out.d4 = 0.0
        out.d4_note = "liquid_neg"

    return out


def apply_culture_transform(
    parsed: ParsedSputum,
    schemes: LabelSchemes,
    *,
    solid_rank_score: float | None = None,
    liquid_rank_score: float | None = None,
) -> ParsedSputum:
    """D3 solid and D4 liquid use separate transforms."""
    if schemes.solid_transform == "rank":
        if parsed.solid_ttp_days is not None and solid_rank_score is not None:
            parsed.d3 = solid_rank_score
    else:
        if parsed.solid_ttp_days is not None:
            parsed.d3 = schemes.solid_fn()(parsed.solid_ttp_days)

    if schemes.liquid_transform == "rank":
        if parsed.liquid_ttp_days is not None and liquid_rank_score is not None:
            parsed.d4 = liquid_rank_score
    else:
        if parsed.liquid_ttp_days is not None:
            parsed.d4 = schemes.liquid_fn()(parsed.liquid_ttp_days)

    return parsed
