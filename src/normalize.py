import re
from typing import Dict, Any

def _pct_to_float(s: str):
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", str(s))
    if m:
        return float(m.group(1).replace(",", "."))
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return None

def _eur_to_int(s: str):
    if s is None:
        return None
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else None

def _int_from_text(s: str):
    if s is None:
        return None
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None

def compute_derived_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    data.setdefault("Financial Instrument", "contrat de prêt")
    data.setdefault("Spv", "LookandFin Finance")
    data.setdefault("Timetable Type", "Amortissable")

    amt = _eur_to_int(data.get("Loan Amount Borrow"))
    if amt is not None:
        data["Loan Amount Borrow"] = amt
        data["Loan Min Amount to raise"] = int(0.8 * amt)
        data["Loan Amount Start"] = amt
        data["Loan Amount Max"] = amt

    rate = _pct_to_float(data.get("Loan Interest Rate"))
    if rate is not None:
        data["Loan Interest Rate"] = rate

    dur = _int_from_text(data.get("Loan Duration"))
    if dur is not None:
        data["Loan Duration"] = dur
        data["Loan Franchise duration"] = max(dur - 1, 0)

    cr = _pct_to_float(data.get("Commission Rate"))
    if cr is not None:
        data["Commission Rate"] = cr
    if "Exit Fee Commission Rate" in data:
        data["Exit Fee Commission Rate"] = _pct_to_float(data.get("Exit Fee Commission Rate"))

    if not data.get("Periodicity Running Commission Rate"):
        data["Periodicity Running Commission Rate"] = "Mensuelle (0,1%)"

    ltv = _pct_to_float(data.get("LTV Max Ratio"))
    if ltv is not None:
        data["LTV Max Ratio"] = ltv
    ltv_pre = _pct_to_float(data.get("LTV Max Pre construction Ratio"))
    if ltv_pre is not None:
        data["LTV Max Pre construction Ratio"] = ltv_pre

    for f in ["Caution personnelle","GAPD","Co-debitor","Subordinated Creditors"]:
        data.setdefault(f, "Non")

    if data.get("Loan Anticipated Refund") is None:
        data["Loan Anticipated Refund"] = "Non"

    return data
