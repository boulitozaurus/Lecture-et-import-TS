# --- MUST BE FIRST: make project root importable ---
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
from src.rules import apply_rules  # NOUVEAU: nos règles métier
from src.normalize import compute_derived_fields

CONFIG = ROOT / "src" / "config"
CLAUSES = ROOT / "src" / "clauses"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="TS Parser", layout="wide")
st.title("📄 Term Sheet Parser (Upload → Auto-remplissage)")

# Ordre d’affichage des champs (ta liste)
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

# Constantes visibles mais non modifiables
CONSTANT_FIELDS = {
    "Financial Instrument": "contrat de prêt",
    "Spv": "LookandFin Finance",
    "Timetable Type": "Amortissable",
}

def to_kv_dataframe(rec: dict) -> pd.DataFrame:
    """Affiche en 2 colonnes texte pour éviter Overflow/pyarrow."""
    rows = []
    added = set()
    for k in FIELDS_ORDER:
        v = rec.get(k, "")
        rows.append({"Field": k, "Value": "" if v is None else str(v)})
        added.add(k)
    for k, v in rec.items():
        if k not in added:
            rows.append({"Field": k, "Value": "" if v is None else str(v)})
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    return df.astype({"Field": "string", "Value": "string"})

uploaded = st.file_uploader("Dépose ton PDF de Term Sheet", type=["pdf"])

if uploaded:
    pdf_path = OUTPUTS / uploaded.name
    pdf_path.write_by
