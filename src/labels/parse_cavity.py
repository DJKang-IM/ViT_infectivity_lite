# -*- coding: utf-8 -*-
"""Parse CT reading JSON for D5 cavity labels."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from .utils import ORDER_RE, in_window, parse_date

CAVITY = re.compile(
    r"(공동성|공동|cavitary|cavitation|cavity|caviat|cavit|cativary|cavitate|cavitated|cavitating)",
    re.I,
)
PRESENT_MIX = re.compile(
    r"(with\s*/\s*without\s+cavit"
    r"|without\s*/\s*with\s+cavit"
    r"|with or without cavit"
    r"|cavitary\s*&\s*non[-\s]?cavitary"
    r"|non[-\s]?cavitary\s*&\s*cavitary"
    r"|cavitary\s+or\s+non[-\s]?cavitary)",
    re.I,
)
PURE_NEG = re.compile(
    r"(no\s+(definite\s+|evidence\s+of\s+|significant\s+)?cavit"
    r"|without\s+cavit"
    r"|cavit\w*\s*\(\s*-\s*\)"
    r"|absence of\s+cavit"
    r"|no cavitation"
    r"|not\s+cavit"
    r"|공동[^.]{0,12}없"
    r"|cavit\w*[^.]{0,12}없)",
    re.I,
)
PAREN_DATE = re.compile(r"\((\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\)")


def split_sentences(t: str) -> list[str]:
    t = t.replace("\r", " ")
    parts = re.split(r"(?<![0-9])[.\n;]+", t)
    return [p.strip() for p in parts if p.strip()]


def block_cavity(txt: str) -> str:
    """Return 'pos' | 'neg' | 'none'."""
    cav_sents = [s for s in split_sentences(txt) if CAVITY.search(s)]
    if not cav_sents:
        return "none"
    if any(PRESENT_MIX.search(s) or not PURE_NEG.search(s) for s in cav_sents):
        return "pos"
    return "neg"


def ct_event_dates(txt: str) -> list[date]:
    ds: list[date] = []
    om = ORDER_RE.search(txt)
    if om:
        for g in (om.group(1), om.group(2)):
            d = parse_date(g)
            if d:
                ds.append(d)
    head = re.split(r"compared", txt, 1, flags=re.I)[0]
    for m in PAREN_DATE.finditer(head):
        try:
            ds.append(date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            pass
    return ds


def load_ct_blocks(ct_dir: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for p in ct_dir.glob("*.json"):
        if not (
            p.name.endswith("_chest_ct.json")
            or p.name.endswith("_EXTERNAL.json")
            or p.name.endswith("_ct_reading.json")
        ):
            continue
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
            s = int(o["study_no"])
        except Exception:
            continue
        if not (10000 <= s < 20000):
            continue
        for b in o.get("exam_blocks", []):
            out[s].append(str(b.get("exam_text", "")))
        for b in o.get("external_blocks", []):
            out[s].append(str(b.get("external_text", "")))
        # legacy GUI format
        for b in o.get("ct_reading_blocks", []):
            out[s].append(str(b.get("ct_reading_text", "")))
    return dict(out)


def derive_d5(
    blocks: list[str],
    anchor: date | None,
    *,
    lo_m: int = -3,
    hi_m: int = 1,
) -> tuple[float | None, str, str]:
    """
    Return (label 0.0/1.0/None, source, note).
    None => use CSV fallback.
    """
    in_win = []
    for txt in blocks:
        ds = ct_event_dates(txt)
        if any(in_window(d, anchor, lo_m, hi_m) for d in ds):
            in_win.append(txt)
    if not in_win:
        return None, "fallback", "no_inwindow_ct"
    statuses = [block_cavity(t) for t in in_win]
    if "pos" in statuses:
        for t in in_win:
            if block_cavity(t) == "pos":
                cs = [s for s in split_sentences(t) if CAVITY.search(s)]
                return 1.0, "ct", f"ct_pos:{cs[0][:120] if cs else ''}"
    if "neg" in statuses:
        return 0.0, "ct", "ct_cavity_negated"
    return 0.0, "ct", "ct_no_cavity_mention"
