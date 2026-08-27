# -*- coding: utf-8 -*-
"""Label encoding schemes for D1–D7 graded targets in [0, 1]."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

POS_AFB = ["rare (1+)", "few (2+)", "moderate (3+)", "numerous (4+)",
           "1+", "2+", "3+", "4+"]
TRACE_AFB = ["doubtful", "trace"]

GRADE_MAP = {
    "1+": 0.25, "rare (1+)": 0.25, "rare": 0.25,
    "2+": 0.50, "few (2+)": 0.50, "few": 0.50,
    "3+": 0.75, "moderate (3+)": 0.75, "moderate": 0.75,
    "4+": 1.0, "numerous (4+)": 1.0, "numerous": 1.0,
}


def _afb_negative(t: str) -> bool:
    return any(k in t for k in ("no afb", "not seen", "negative"))


def encode_afb_grade_v1(text: str) -> float | None:
    t = text.lower().strip()
    if not t:
        return None
    if _afb_negative(t):
        return 0.0
    if any(k in t for k in TRACE_AFB):
        return 0.125
    if "positive for afb" in t:
        return 1.0
    for key, val in GRADE_MAP.items():
        if key in t:
            return val
    if "positive" in t:
        return 1.0
    return None


def encode_afb_binary(text: str) -> float | None:
    v = encode_afb_grade_v1(text)
    if v is None:
        return None
    return 1.0 if v > 0 else 0.0


def encode_afb_ordinal_5(text: str) -> float | None:
    v = encode_afb_grade_v1(text)
    if v is None:
        return None
    if v == 0.0:
        return 0.0
    if v <= 0.125:
        return 0.25
    return v


def encode_afb_raw_div4(text: str) -> float | None:
    t = text.lower().strip()
    if not t:
        return None
    if _afb_negative(t):
        return 0.0
    if any(k in t for k in TRACE_AFB):
        return 0.25
    m = re.search(r"(\d)\+", t)
    if m:
        return min(int(m.group(1)) / 4.0, 1.0)
    if "positive" in t:
        return 1.0
    return None


# AFB Layer-2 calibration: polychoric (culture-anchored) grade scores in [0,1].
# Fitted by scripts/run_afb_layer2_comparison.py; see artifacts/afb_layer2_comparison.json
# and [중요]AFB_Scoring.md. Ordinal grade g -> f_afb(g).
#   0 = negative, 1 = trace/doubtful, 2 = 1+, 3 = 2+, 4 = 3+, 5 = 4+/positive
AFB_POLYCHORIC_SCORES: dict[int, float] = {
    0: 0.0,
    1: 0.4739773378038599,
    2: 0.6025870154308465,
    3: 0.6859208105093068,
    4: 0.9398078449561844,
    5: 1.0,
}

# encode_afb_grade_v1 output -> ordinal grade index
_GRADE_V1_TO_ORDINAL: dict[float, int] = {
    0.0: 0, 0.125: 1, 0.25: 2, 0.5: 3, 0.75: 4, 1.0: 5,
}


def encode_afb_polychoric_culture_v1(text: str) -> float | None:
    """D1 AFB smear -> culture-anchored polychoric score in [0, 1] (MSE target)."""
    v = encode_afb_grade_v1(text)
    if v is None:
        return None
    ordinal = _GRADE_V1_TO_ORDINAL.get(round(v * 8) / 8)
    if ordinal is None:
        return None
    return AFB_POLYCHORIC_SCORES[ordinal]


AFB_ENCODERS: dict[str, Callable[[str], float | None]] = {
    "afb_grade_v1": encode_afb_grade_v1,
    "afb_binary": encode_afb_binary,
    "afb_ordinal_5": encode_afb_ordinal_5,
    "afb_raw_div4": encode_afb_raw_div4,
    "afb_polychoric_culture_v1": encode_afb_polychoric_culture_v1,
}


def encode_pcr_soft_v1(text: str) -> float | None:
    """TB-PCR (D2) and RIF resistance PCR (D7) share this rule."""
    tl = text.lower()
    if "indeterminate" in tl:
        return 0.5
    if "not detected" in tl or ("negative" in tl and "positive" not in tl):
        return 0.0
    if "positive" in tl or "detected" in tl:
        return 1.0
    return None


def encode_pcr_binary_v9(text: str) -> float | None:
    tl = text.lower()
    if "indeterminate" in tl:
        return 1.0
    if "not detected" in tl or ("negative" in tl and "positive" not in tl):
        return 0.0
    if "positive" in tl or "detected" in tl:
        return 1.0
    return None


PCR_ENCODERS: dict[str, Callable[[str], float | None]] = {
    "pcr_soft_v1": encode_pcr_soft_v1,
    "pcr_binary_v9": encode_pcr_binary_v9,
}


def culture_transform_inv(days: int) -> float:
    d = max(int(days), 1)
    return min(1.0 / d, 1.0)


def culture_transform_loginv(days: int) -> float:
    d = max(int(days), 1)
    return min(1.0 / math.log1p(d), 1.0)


def culture_transform_liquid_twostep(days: int) -> float:
    """
    Option C (v1 default for D4 liquid):
      d <= 3  -> 1.0  (fast liquid positivity = high infectivity proxy)
      d > 3   -> loginv(d)
    """
    d = max(int(days), 1)
    if d <= 3:
        return 1.0
    return culture_transform_loginv(d)


SOLID_TRANSFORMS: dict[str, Callable[[int], float]] = {
    "inv": culture_transform_inv,
    "loginv": culture_transform_loginv,
}

LIQUID_TRANSFORMS: dict[str, Callable[[int], float]] = {
    "inv": culture_transform_inv,
    "loginv": culture_transform_loginv,
    "twostep": culture_transform_liquid_twostep,
}


@dataclass
class LabelSchemes:
    afb: str = "afb_grade_v1"
    pcr: str = "pcr_soft_v1"
    solid_transform: str = "loginv"
    liquid_transform: str = "twostep"

    def afb_fn(self) -> Callable[[str], float | None]:
        return AFB_ENCODERS[self.afb]

    def pcr_fn(self) -> Callable[[str], float | None]:
        return PCR_ENCODERS[self.pcr]

    def solid_fn(self) -> Callable[[int], float]:
        return SOLID_TRANSFORMS[self.solid_transform]

    def liquid_fn(self) -> Callable[[int], float]:
        if self.liquid_transform == "rank":
            raise ValueError("rank transform applied cohort-wide after parsing")
        return LIQUID_TRANSFORMS[self.liquid_transform]


def apply_rank_transform(positive_days: dict[int, int]) -> dict[int, float]:
    if not positive_days:
        return {}
    items = sorted(positive_days.items(), key=lambda x: x[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 1.0}
    out: dict[int, float] = {}
    for rank, (study, _d) in enumerate(items):
        out[study] = 1.0 - rank / (n - 1)
    return out
