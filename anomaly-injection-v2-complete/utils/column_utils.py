from __future__ import annotations
import re
from typing import Iterable, Optional

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns); normalized = {_norm(c): c for c in cols}
    for cand in candidates:
        if _norm(cand) in normalized: return normalized[_norm(cand)]
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            cn = _norm(c)
            if n and (n in cn or cn in n): return c
    return None

def phase_candidates(kind: str, phase: int):
    p = str(phase)
    mapping = {
        "V": [f"V_L{p}",f"VL{p}",f"Voltage_L{p}",f"Voltage{p}",f"V{p}"],
        "I": [f"I_L{p}",f"IL{p}",f"Current_L{p}",f"Current{p}",f"I{p}"],
        "PF": [f"PF_L{p}",f"PFL{p}",f"PowerFactor_L{p}",f"PowerFactor{p}",f"PF{p}"],
        "P": [f"P_L{p}",f"PL{p}",f"ActivePower_L{p}",f"ActivePower{p}",f"P{p}"],
        "Q": [f"Q_L{p}",f"QL{p}",f"ReactivePower_L{p}",f"ReactivePower{p}",f"Q{p}"],
        "S": [f"S_L{p}",f"SL{p}",f"ApparentPower_L{p}",f"ApparentPower{p}",f"S{p}"],
    }
    return mapping.get(kind, [])
