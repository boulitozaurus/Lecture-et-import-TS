# --- MUST BE FIRST: make 'src' importable no matter the launch dir ---
import sys, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve()

def _ensure_src_on_path() -> Path | None:
    # essaie parent, grand-parent, cwd, puis remonte l'arbo jusqu'à trouver 'src'
    candidates = [HERE.parent, HERE.parent.parent, HERE.parent.parent.parent, Path.cwd()]
    for base in candidates + list(HERE.parents):
        if (base / "src").exists():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            return base
    return None

PROJECT_ROOT = _ensure_src_on_path()

import streamlit as st
import pandas as pd
import pdfplumber

# 1er essai: import package "src.rules"
try:
    from src.rules import apply_rules
except Exception:
    # 2e essai: charger directement le fichier src/rules.py s'il existe
    rules_path = None
    for base in [HERE.parent, HERE.parent.parent, HERE.parent.parent.parent, Path.cwd()] + list(HERE.parents):
        cand = base / "src" / "rules.py"
        if cand.exists():
            rules_path = cand
            break
    if rules_path is not None:
        spec = importlib.util.spec_from_file_location("rules", rules_path)
        rules_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules_mod)  # type: ignore
        apply_rules = rules_mod.apply_rules  # type: ignore
    else:
        # Fallback neutre: l'app tourne, mais sans appliquer les règles métier
        def apply_rules(d, full_text): 
            return dict(d)
        st.warning("⚠️ Règles métier introuvables (src/rules.py). Fallback neutre activé.")

from src.parser.pdf_parser import parse_pdf
from src.normalize import compute_derived_fields

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="TS Parser", layout="wide")
st.title("📄 Term Sheet Parser (Upload → Auto-remplissage)")

# Ordre d’affichage (ta liste)
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

# Constantes visibles mais verrouillées
CONSTANT_FIELDS = {
    "Financial Instrument": "contrat de prêt",
    "Spv": "LookandFin Finance",
    "Timetable Type": "Amortissable",
}

def to_kv_dataframe(rec: dict) -> pd.DataFrame:
    """2 colonnes texte -> pas de conversion numérique -> pas d'Overflow/pyarrow."""
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
    pdf_path.write_bytes(uploaded.getvalue())

    # 1) Extraction brute clé/valeur
    raw = parse_pdf(str(pdf_path))

    # 2) Texte complet pour la détection (périodicité, clauses…)
    with pdfplumber.open(str(pdf_path)) as pdf:
        full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])

    # 3) Constantes
    raw.update(CONSTANT_FIELDS)

    # 4) Règles métier (DOCX) -> type, périodicité, clauses, centralisateur…
    rec = apply_rules(raw, full_text)

    # 5) Normalisation + champs dérivés
    rec = compute_derived_fields(rec)

    # 6) Formulaire (constantes disabled)
    st.subheader("Champs pré-remplis (éditables)")
    with st.form("edit_form"):
        c1, c2 = st.columns(2)
        ed = {}

        def disabled(label, value):
            st.text_input(label, value=value, disabled=True)

        with c1:
            for ck in ["Financial Instrument", "Spv", "Timetable Type"]:
                disabled(ck, str(rec.get(ck, CONSTANT_FIELDS.get(ck, ""))))

            ed["Loan Amount Borrow"] = st.text_input("Loan Amount Borrow (EUR)", str(rec.get("Loan Amount Borrow","")))
            ed["Loan Interest Rate"] = st.text_input("Loan Interest Rate (%)", str(rec.get("Loan Interest Rate","")))
            ed["Loan Duration"] = st.text_input("Loan Duration (months)", str(rec.get("Loan Duration","")))
            ed["Loan Franchise duration"] = st.text_input("Loan Franchise duration (months)", str(rec.get("Loan Franchise duration","")))
            ed["Commission Rate"] = st.text_input("Commission Rate (%)", str(rec.get("Commission Rate","")))
            ed["Exit Fee Commission Rate"] = st.text_input("Exit Fee Commission Rate (%)", str(rec.get("Exit Fee Commission Rate","")))

        with c2:
            ed["Loan Type"] = st.text_input("Loan Type", str(rec.get("Loan Type","")))
            ed["Loan Refund Periodicity"] = st.text_input("Loan Refund Periodicity", str(rec.get("Loan Refund Periodicity","")))
            ed["LTV Max Ratio"] = st.text_input("LTV Max Ratio (%)", str(rec.get("LTV Max Ratio","")))
            ed["LTV Max Pre construction Ratio"] = st.text_input("LTV Max Pre construction Ratio (%)", str(rec.get("LTV Max Pre construction Ratio","")))
            ed["Fund Usage Description"] = st.text_area("Fund Usage Description", str(rec.get("Fund Usage Description","")), height=120)
            ed["Periodicity Running Commission Rate"] = st.text_input("Periodicity Running Commission Rate (%)", str(rec.get("Periodicity Running Commission Rate","")))

        ed["Financement de l'acquisition"] = st.text_input("Financement de l'acquisition", str(rec.get("Financement de l'acquisition","")))
        ed["Loan Warranties"] = st.text_input("Loan Warranties", str(rec.get("Loan Warranties","")))
        ed["Loan Anticipated Refund"] = st.text_input("Loan Anticipated Refund", str(rec.get("Loan Anticipated Refund","")))
        ed["Loan Min Duration Before Early Repayment"] = st.text_input("Loan Min Duration Before Early Repayment (months)", str(rec.get("Loan Min Duration Before Early Repayment","")))
        ed["Suspensives clauses"] = st.text_area("Suspensives clauses (auto-détectées)", str(rec.get("Suspensives clauses","")), height=100)

        if st.form_submit_button("Appliquer"):
            rec.update(ed)
            rec = compute_derived_fields(rec)
            st.success("Champs mis à jour")

    st.markdown("### Résumé (clé / valeur)")
    st.dataframe(to_kv_dataframe(rec), use_container_width=True)

else:
    st.info("Dépose un fichier PDF pour démarrer.")
