import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # racine du repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import pdfplumber
import yaml
import re

from src.parser.pdf_parser import parse_pdf
from src.normalize import compute_derived_fields

# -----------------------------
# Config
# -----------------------------
CONFIG = ROOT / "src" / "config"
CLAUSES = ROOT / "src" / "clauses"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="TS Parser", layout="wide")
st.title("📄 Term Sheet Parser (Upload → Auto-remplissage)")

# Champs à afficher (ton ordre)
FIELDS_ORDER = [
    "Loan Amount Borrow",
    "Financial Instrument",
    "Spv",
    "Loan Type",
    "Loan Refund Periodicity",
    "Loan Min Amount to raise",
    "Loan Interest Rate",
    "Timetable Type",
    "Loan Duration",
    "Loan Franchise duration",
    "Commission Rate",
    "Exit Fee Commission Rate",
    "Periodicity Running Commission Rate",
    "Fund Usage Description",
    "LTV Max Ratio",
    "LTV Max Pre construction Ratio",
    "Financement de l'acquisition",
    "Suspensives clauses",
    "Loan Warranties",
    "Caution personnelle",
    "GAPD",
    "Co-debitor",
    "Subordinated Creditors",
    "Loan Amount Start",
    "Loan Amount Max",
    "Loan Anticipated Refund",
    "Loan Min Duration Before Early Repayment",
]

# Champs constants (affichés dans la zone d’édition mais désactivés)
CONSTANT_FIELDS = {
    "Financial Instrument": "contrat de prêt",
    "Spv": "LookandFin Finance",
    "Timetable Type": "Amortissable",
}

# Clauses suspensives (catalogue YAML optionnel)
def load_clauses_catalog():
    yml = CLAUSES / "clauses.yaml"
    if yml.exists():
        obj = yaml.safe_load(yml.read_text(encoding="utf-8"))
        # aplatissement (liste ou catégories)
        if isinstance(obj, list):
            out = [str(x).strip() for x in obj if str(x).strip()]
        elif isinstance(obj, dict):
            out = []
            for v in obj.values():
                if isinstance(v, list):
                    out.extend([str(x).strip() for x in v if str(x).strip()])
                elif isinstance(v, str) and v.strip():
                    out.append(v.strip())
        else:
            out = []
        # unicité
        seen = set(); uniq = []
        for c in out:
            if c not in seen:
                uniq.append(c); seen.add(c)
        return uniq
    # fallback minimal si pas de YAML
    return [
        "Expertise indépendante (en l'état, après travaux, liquidative)",
        "Respect des covenants LTV (en l'état / post-travaux)",
        "Mise en place des sûretés (fiducie-sûreté / hypothèque)",
        "Libération des fonds via notaire",
        "Décaissements travaux via compte centralisateur nanti",
        "Tranches min. 5 000 €; acomptes plafonnés 20%",
        "Signature de la convention de prêt et PV nécessaires",
    ]

CLAUSES_CATALOG = load_clauses_catalog()

# -----------------------------
# Heuristiques (aucun choix utilisateur avant upload)
# -----------------------------
def guess_loan_type(text: str) -> str | None:
    t = text.lower()
    if "fiducie" in t or "fiducie-sûreté" in t or "fiducie surete" in t:
        return "Fiducie-sûreté"
    if "garantie hypothéc" in t:
        return "Garantie hypothécaire de premier rang"
    if "non assur" in t:
        return "Non assuré"
    if "assur" in t:
        return "Assuré"
    if "pmv" in t:
        return "PMV"
    if "wininlening" in t:
        return "Wininlening"
    if "proxi" in t:
        return "Prêt proxi"
    if "coup de pouce" in t:
        return "Prêt coup de pouce"
    return None

def guess_periodicity(text: str) -> str | None:
    t = text.lower()
    # repérage mots-clés
    if "mensuel" in t or "mensuelle" in t or "mensualit" in t:
        return "Mensuelle"
    if "trimestriel" in t or "trimestrielle" in t:
        return "Trimestrielle"
    if "semestriel" in t or "semestrielle" in t:
        return "Semestrielle"
    if "annuel" in t or "annuelle" in t:
        return "Annuelle"
    return None

def detect_clauses(text: str, catalog: list[str]) -> list[str]:
    """Détecte les clauses présentes dans le texte du PDF (substring insensible à la casse).
       Simple et robuste. Tu pourras raffiner plus tard (regex/keywords)."""
    t = re.sub(r"\s+", " ", text or "").lower()
    found = []
    for clause in catalog:
        c = re.sub(r"\s+", " ", clause).lower()
        if len(c) >= 6 and c in t:
            found.append(clause)
    # unicité, ordre d’apparition dans catalog
    seen = set(); uniq = []
    for c in found:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

# -----------------------------
# UI minimale : juste un uploader
# -----------------------------
uploaded = st.file_uploader("Dépose ton PDF de Term Sheet", type=["pdf"])

def to_kv_dataframe(rec: dict) -> pd.DataFrame:
    """Sécurise l’affichage contre pyarrow/Overflow: tout en texte, 2 colonnes."""
    rows = []
    for k in FIELDS_ORDER:
        if k in rec:
            v = rec.get(k, "")
            # stringify propre
            if v is None:
                s = ""
            elif isinstance(v, (int, float, bool)):
                s = str(v)
            else:
                s = str(v)
            rows.append({"Field": k, "Value": s})
    # Ajoute les clés non listées (au cas où)
    for k, v in rec.items():
        if k not in FIELDS_ORDER:
            rows.append({"Field": k, "Value": str(v)})
    df = pd.DataFrame(rows)
    # tout en string pour éviter toute conversion Arrow hasardeuse
    return df.astype({"Field": "string", "Value": "string"})

if uploaded:
    pdf_path = OUTPUTS / uploaded.name
    pdf_path.write_bytes(uploaded.getvalue())

    # 1) extraction brut (clé/valeur) via parseur
    raw = parse_pdf(str(pdf_path))

    # 2) lecture texte complet pour heuristiques (type, périodicité, clauses)
    with pdfplumber.open(str(pdf_path)) as pdf:
        full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])

    # 3) constantes imposées (visibles mais non modifiables + utilisées au calcul)
    for k, v in CONSTANT_FIELDS.items():
        raw[k] = v

    # 4) heuristiques automatiques (pas de sidebar)
    lt = guess_loan_type(full_text)
    if lt:
        raw["Loan Type"] = lt
    pr = guess_periodicity(full_text)
    if pr:
        raw["Loan Refund Periodicity"] = pr

    # 5) détection automatique des clauses
    clauses_found = detect_clauses(full_text, CLAUSES_CATALOG)
    if clauses_found:
        raw["Suspensives clauses"] = "; ".join(clauses_found)

    # 6) normalisation + règles dérivées (min à lever, franchise, etc.)
    rec = compute_derived_fields(raw)

    # 7) Formulaire d’édition (les champs constants sont désactivés)
    st.subheader("Champs pré-remplis")
    with st.form("edit_form"):
        col1, col2 = st.columns(2)
        edited = {}

        def text_input_disabled(label, value):
            # champ 'readonly' visuellement cohérent
            st.text_input(label, value=value, disabled=True)

        # Colonne 1
        with col1:
            # constants (affichés mais désactivés)
            for ck in ["Financial Instrument", "Spv", "Timetable Type"]:
                text_input_disabled(ck, str(rec.get(ck, CONSTANT_FIELDS.get(ck, ""))))
            # éditables
            edited["Loan Amount Borrow"] = st.text_input("Loan Amount Borrow (EUR)", str(rec.get("Loan Amount Borrow", "")))
            edited["Loan Interest Rate"] = st.text_input("Loan Interest Rate (%)", str(rec.get("Loan Interest Rate", "")))
            edited["Loan Duration"] = st.text_input("Loan Duration (months)", str(rec.get("Loan Duration", "")))
            edited["Loan Franchise duration"] = st.text_input("Loan Franchise duration (months)", str(rec.get("Loan Franchise duration", "")))
            edited["Commission Rate"] = st.text_input("Commission Rate (%)", str(rec.get("Commission Rate", "5")))
            edited["Exit Fee Commission Rate"] = st.text_input("Exit Fee Commission Rate (%)", str(rec.get("Exit Fee Commission Rate", "0")))

        # Colonne 2
        with col2:
            edited["Loan Type"] = st.text_input("Loan Type", str(rec.get("Loan Type", "")))
            edited["Loan Refund Periodicity"] = st.text_input("Loan Refund Periodicity", str(rec.get("Loan Refund Periodicity", "")))
            edited["LTV Max Ratio"] = st.text_input("LTV Max Ratio (%)", str(rec.get("LTV Max Ratio", "")))
            edited["LTV Max Pre construction Ratio"] = st.text_input("LTV Max Pre construction Ratio (%)", str(rec.get("LTV Max Pre construction Ratio", "")))
            edited["Fund Usage Description"] = st.text_area("Fund Usage Description", str(rec.get("Fund Usage Description", "")), height=120)
            edited["Periodicity Running Commission Rate"] = st.text_input("Periodicity Running Commission Rate", str(rec.get("Periodicity Running Commission Rate", "Mensuelle (0,1%)")))

        # Ligne 3 (pleine largeur) pour autres drapeaux
        edited["Financement de l'acquisition"] = st.text_input("Financement de l'acquisition", str(rec.get("Financement de l'acquisition", "")))
        edited["Loan Warranties"] = st.text_input("Loan Warranties", str(rec.get("Loan Warranties", "")))
        edited["Loan Anticipated Refund"] = st.text_input("Loan Anticipated Refund", str(rec.get("Loan Anticipated Refund", "")))
        edited["Loan Min Duration Before Early Repayment"] = st.text_input("Loan Min Duration Before Early Repayment (months)", str(rec.get("Loan Min Duration Before Early Repayment", "")))
        # Clauses détectées : visibles mais éditables aussi si tu veux corriger
        edited["Suspensives clauses"] = st.text_area("Suspensives clauses (auto-détectées)", str(rec.get("Suspensives clauses", "")), height=100)

        submitted = st.form_submit_button("Appliquer")
        if submitted:
            rec.update(edited)
            rec = compute_derived_fields(rec)
            st.success("Champs mis à jour")

    st.markdown("### Résumé (clé / valeur)")
    df = to_kv_dataframe(rec)
    # data en texte => pas d'overflow pyarrow
    st.dataframe(df, use_container_width=True)

else:
    st.info("Dépose un fichier PDF pour démarrer.")
