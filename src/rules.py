import re
from typing import Dict, Any, List
from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parents[1]
CLAUSES_CFG = BASE / "clauses" / "clauses_patterns.yaml"

def _load_clause_patterns() -> Dict[str, List[str]]:
    """YAML: code_de_clause -> liste de regex (insensibles à la casse)."""
    if CLAUSES_CFG.exists():
        return yaml.safe_load(CLAUSES_CFG.read_text(encoding="utf-8")) or {}
    # fallback minimal si fichier manquant
    return {
        "CAC_NOMINATION": [r"\bcommissaire[s]?\s+aux\s+comptes?\b", r"\bnomination\s+du\s+cac\b"],
        "CAC_REPORT": [r"\brapport\s+du\s+commissaire\s+aux\s+comptes?\b"],
        "PV_AG": [r"\bpv\s+(d['e ]|de\s+)assembl[ée]e?\s+g[ée]n[ée]rale\b"],
        "URBANISM_OK": [r"\burbanisme\s+ok\b", r"\bbien\s+en\s+ordre\s+d[’']urbanisme\b"],
        "CENTRALIZED_ACCOUNT": [r"\baircap\b", r"\bibanfirst\b", r"\bcompte\s+centralisateur\b"],
        "WARRANTY_TRUST_CASH": [r"\bfiducie[-\s]?s[uû]ret[eé]\b", r"\bcompte\s+centralisateur\s+nanti\b"],
        "MANUAL": [r"\bdevis\s+sign[é]s?\b", r"\bkbis\b", r"\bstatuts?\b", r"\bcr[ée]ation\s+de\s+la\s+soci[eé]t[eé]\b", r"\blib[ée]ration\s+du\s+capital\b"],
    }

CLAUSE_PATTERNS = _load_clause_patterns()

def _contains(text: str, pat: str) -> bool:
    return re.search(pat, text, flags=re.I|re.S) is not None

def infer_periodicity(text: str) -> str | None:
    t = text.lower()
    if "mensuel" in t or "mensuelle" in t or "mensualit" in t:
        return "Mensuelle"
    if "trimestriel" in t or "trimestrielle" in t:
        return "Trimestrielle"
    if "semestriel" in t or "semestrielle" in t:
        return "Semestrielle"
    if "annuel" in t or "annuelle" in t:
        return "Annuelle"
    return None

def infer_loan_type(text: str, warranties: str | None, rate: float | None) -> str | None:
    # Règle DOCX: si taux ≠ {7, 7.5, 8} -> risk class A+ -> Assuré
    if rate is not None and (rate not in {7.0, 7.5, 8.0}):
        return "Assuré"
    # Sinon: déduction sur la base des sûretés
    w = (warranties or "") + "\n" + text
    w = w.lower()
    if "fiducie" in w:
        return "Fiducie-sûreté"
    if "hypoth" in w:
        return "Garantie hypothécaire de premier rang"
    if "pmv" in w:
        return "PMV"
    if "wininlening" in w:
        return "Wininlening"
    if "proxi" in w:
        return "Prêt proxi"
    if "coup de pouce" in w:
        return "Prêt coup de pouce"
    # fallback
    return "Non assuré"

def detect_anticipated_refund(text: str) -> tuple[str | None, int | None]:
    t = text.lower()
    if "remboursement anticip" in t and "sans frais" in t:
        # cherche éventuellement un palier minimum (ex: 6 mois)
        m = re.search(r"(?:apr[eè]s|après)\s+(\d+)\s+mois", t)
        min_months = int(m.group(1)) if m else None
        # Si fenêtre conditionnelle : on marque "Partially" sinon "Oui"
        kind = "Partially" if min_months else "Oui"
        return kind, min_months
    return None, None

def detect_centralizer(text: str) -> str:
    t = text.lower()
    if "ibanfirst" in t or "iban first" in t:
        return "IbanFirst"
    if "aircap" in t:
        return "Aircap"
    # fallback (DOCX)
    return "Aircap"

def detect_clauses(text: str) -> List[str]:
    found = []
    for code, regexes in CLAUSE_PATTERNS.items():
        for pat in regexes:
            if _contains(text, pat):
                found.append(code)
                break
    # unicité et ordre de config
    seen = set(); uniq = []
    for c in found:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

def detect_covenants(text: str) -> List[str]:
    t = text.lower()
    cov = []
    if "shareholding" in t or "actionnariat" in t:
        cov.append("Shareholding clause")
    if "negative pledge" in t or "nantissement suppl" in t or "pas de nantissement" in t:
        cov.append("Negative Pledge")
    if "no additional debt" in t or "pas d'endettement suppl" in t:
        cov.append("No Additional Debt")
    if "cost overrun" in t or "dépassement de coûts" in t:
        cov.append("Cost Overrun")
    if "cash deficiency" in t or "insuffisance de trésorerie" in t:
        cov.append("Cash Deficiency")
    return cov

def apply_rules(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    """Applique toutes les règles du prompt DOCX sur les données brutes extraites."""
    out = dict(data)

    # Périodicité (pas de sidebar)
    per = infer_periodicity(full_text)
    if per:
        out["Loan Refund Periodicity"] = per

    # Remboursement anticipé
    ar, min_months = detect_anticipated_refund(full_text)
    if ar:
        out["Loan Anticipated Refund"] = ar
    if min_months is not None:
        out["Loan Min Duration Before Early Repayment"] = min_months

    # Centralisateur (et clause)
    central = detect_centralizer(full_text)
    out["Centralising Account Provider"] = central
    clauses = detect_clauses(full_text)
    if "CENTRALIZED_ACCOUNT" not in clauses and central:
        clauses.append("CENTRALIZED_ACCOUNT")
    if clauses:
        out["Suspensives clauses"] = "; ".join(clauses)

    # Covenants (si tu veux les stocker en champ texte)
    covs = detect_covenants(full_text)
    if covs:
        out["Covenants"] = "; ".join(covs)

    # Déduction Loan Type (sans valeurs par défaut pour commissions)
    # On convertit le taux si déjà normalisé ailleurs -> on tente quand même
    rate = None
    try:
        rate = float(str(out.get("Loan Interest Rate","")).replace(",", "."))
    except Exception:
        pass

    out["Loan Type"] = infer_loan_type(full_text, out.get("Loan Warranties"), rate)

    # NE PAS remplir Commission Rate / Periodicity Running Commission Rate si absents
    # (Ils resteront vides si la TS ne les contient pas)

    return out
