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
    out = dict(data)

    # Constantes (déjà injectées par l’app, on les confirme)
    out.setdefault("Financial Instrument", "contrat de prêt")
    out.setdefault("Spv", "LookandFin Finance")
    out.setdefault("Timetable Type", "Amortissable")

    # Montant + dérivés
    amt = _eur_to_int(out.get("Loan Amount Borrow"))
    if amt is not None:
        out["Loan Amount Borrow"] = amt
        out["Loan Min Amount to raise"] = int(0.8 * amt)
        out["Loan Amount Start"] = out.get("Loan Amount Start", amt)
        out["Loan Amount Max"] = out.get("Loan Amount Max", amt)

    # Taux (%)
    rate = _pct_to_float(out.get("Loan Interest Rate"))
    if rate is not None:
        out["Loan Interest Rate"] = rate

    # Durées
    dur = _int_from_text(out.get("Loan Duration"))
    if dur is not None:
        out["Loan Duration"] = dur
        out["Loan Franchise duration"] = out.get("Loan Franchise duration", max(dur - 1, 0))

    # Commissions -> PAS de valeur par défaut si manquantes
    cr = _pct_to_float(out.get("Commission Rate"))
    if cr is not None:
        out["Commission Rate"] = cr

    run_cr = _pct_to_float(out.get("Periodicity Running Commission Rate"))
    if run_cr is not None:
        out["Periodicity Running Commission Rate"] = run_cr

    # LTV
    ltv = _pct_to_float(out.get("LTV Max Ratio"))
    if ltv is not None:
        out["LTV Max Ratio"] = ltv
    ltv_pre = _pct_to_float(out.get("LTV Max Pre construction Ratio"))
    if ltv_pre is not None:
        out["LTV Max Pre construction Ratio"] = ltv_pre

    # Booléens par défaut si absents
    for f in ["Caution personnelle","GAPD","Co-debitor","Subordinated Creditors"]:
        out.setdefault(f, "Non")

    # Remboursement anticipé: valeur par défaut = Non si absent
    out.setdefault("Loan Anticipated Refund", "Non")

    return out
