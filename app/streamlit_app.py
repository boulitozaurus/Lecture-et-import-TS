import io
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml

from src.parser.pdf_parser import parse_pdf
from src.normalize import compute_derived_fields

BASE = Path(__file__).resolve().parents[1]
CONFIG = BASE / "src" / "config"
CLAUSES = BASE / "src" / "clauses"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="Term Sheet Parser", layout="wide")

st.title("📄 Term Sheet PDF Parser – LookandFin")
st.caption("Uploader un PDF → extraction automatique → correction → export CSV/JSON")

# --- Load choices ---
with open(CONFIG / "choices.yaml", "r", encoding="utf-8") as f:
    CHOICES = yaml.safe_load(f)
loan_types = CHOICES.get("LoanType", [])
periodicities = CHOICES.get("LoanRefundPeriodicity", [])

# --- Load clauses from YAML (preferred) or fallback JSON ---
def _flatten_clauses(obj):
    if isinstance(obj, list):
        return [str(x) for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, list):
                out.extend([str(x) for x in v if str(x).strip()])
            elif isinstance(v, str) and v.strip():
                out.append(v.strip())
        # dedupe while preserving order
        seen = set(); uniq = []
        for c in out:
            if c not in seen:
                uniq.append(c); seen.add(c)
        return uniq
    return []

clauses_catalog = []
yml = CLAUSES / "clauses.yaml"
if yml.exists():
    try:
        obj = yaml.safe_load(yml.read_text(encoding="utf-8"))
        clauses_catalog = _flatten_clauses(obj)
    except Exception as e:
        st.error(f"Erreur de lecture YAML: {e}")
elif (CLAUSES / "clauses.json").exists():
    clauses_catalog = json.loads((CLAUSES / "clauses.json").read_text(encoding="utf-8"))
else:
    clauses_catalog = [
        "Expertise indépendante (en l'état, après travaux, liquidative)",
        "Respect des covenants LTV (en l'état / post-travaux)",
        "Mise en place des sûretés (fiducie-sûreté / hypothèque)",
        "Libération des fonds via notaire",
        "Décaissements travaux via compte centralisateur nanti",
        "Tranches min. 5 000 €; acomptes plafonnés 20%",
        "Signature de la convention de prêt et PV nécessaires",
    ]

# --- Sidebar controls ---
with st.sidebar:
    st.header("Contrôles")
    default_loan_type = st.selectbox("Loan Type", loan_types, index=loan_types.index("Fiducie-sûreté") if "Fiducie-sûreté" in loan_types else 0)
    default_periodicity = st.selectbox("Loan Refund Periodicity", periodicities, index=0)
    st.caption("Clauses suspensives (sélection multiple)")
    sel_clauses = st.multiselect("Ajouter au Term Sheet", clauses_catalog, default=[])
    st.divider()
    st.caption("Champs constants")
    st.text_input("Financial Instrument", value="contrat de prêt", key="const_financial_instrument", disabled=True)
    st.text_input("SPV", value="LookandFin Finance", key="const_spv", disabled=True)
    st.text_input("Timetable Type", value="Amortissable", key="const_timetable", disabled=True)

uploaded = st.file_uploader("Déposez un PDF de proposition de financement", type=["pdf"])

def _record_to_df(rec: dict) -> pd.DataFrame:
    return pd.DataFrame([rec])

if uploaded:
    pdf_path = OUTPUTS / uploaded.name
    pdf_path.write_bytes(uploaded.getvalue())

    with st.spinner("Extraction en cours..."):
        raw = parse_pdf(str(pdf_path))
        raw.setdefault("Loan Type", default_loan_type)
        raw.setdefault("Loan Refund Periodicity", default_periodicity)
        if sel_clauses:
            raw["Suspensives clauses"] = "; ".join(sel_clauses)
        raw["Financial Instrument"] = "contrat de prêt"
        raw["Spv"] = "LookandFin Finance"
        raw["Timetable Type"] = "Amortissable"
        rec = compute_derived_fields(raw)

    st.subheader("Champs extraits (éditables)")
    with st.form("edit_form"):
        col1, col2 = st.columns(2)
        editable = {}

        with col1:
            editable["Deal Name"] = st.text_input("Deal Name", value=uploaded.name.replace(".pdf",""))
            editable["Loan Amount Borrow"] = st.text_input("Loan Amount Borrow (EUR)", value=str(rec.get("Loan Amount Borrow","")))
            editable["Loan Interest Rate"] = st.text_input("Loan Interest Rate (%)", value=str(rec.get("Loan Interest Rate","")))
            editable["Loan Duration"] = st.text_input("Loan Duration (months)", value=str(rec.get("Loan Duration","")))
            editable["Loan Franchise duration"] = st.text_input("Loan Franchise duration (months)", value=str(rec.get("Loan Franchise duration","")))
            editable["Commission Rate"] = st.text_input("Commission Rate (%)", value=str(rec.get("Commission Rate","5")))
            editable["Exit Fee Commission Rate"] = st.text_input("Exit Fee Commission Rate (%)", value=str(rec.get("Exit Fee Commission Rate","0")))
            editable["Periodicity Running Commission Rate"] = st.text_input("Periodicity Running Commission Rate", value=str(rec.get("Periodicity Running Commission Rate","Mensuelle (0,1%)")))

        with col2:
            editable["Loan Type"] = st.selectbox("Loan Type", loan_types, index=loan_types.index(rec.get("Loan Type", default_loan_type)) if rec.get("Loan Type", default_loan_type) in loan_types else 0)
            editable["Loan Refund Periodicity"] = st.selectbox("Loan Refund Periodicity", periodicities, index=periodicities.index(rec.get("Loan Refund Periodicity", default_periodicity)) if rec.get("Loan Refund Periodicity", default_periodicity) in periodicities else 0)
            editable["LTV Max Ratio"] = st.text_input("LTV Max Ratio (%)", value=str(rec.get("LTV Max Ratio","")))
            editable["LTV Max Pre construction Ratio"] = st.text_input("LTV Max Pre construction Ratio (%)", value=str(rec.get("LTV Max Pre construction Ratio","")))
            editable["Fund Usage Description"] = st.text_area("Fund Usage Description", value=str(rec.get("Fund Usage Description","")), height=120)
            editable["Financement de l'acquisition"] = st.selectbox("Financement de l'acquisition", ["Oui","Non"], index=0 if str(rec.get("Financement de l'acquisition","Oui")).lower().startswith("o") else 1)
            editable["Loan Anticipated Refund"] = st.selectbox("Loan Anticipated Refund", ["Oui","Non"], index=0 if str(rec.get("Loan Anticipated Refund","Non")).lower().startswith("o") else 1)
            editable["Loan Warranties"] = st.text_area("Loan Warranties", value=str(rec.get("Loan Warranties","")), height=60)

        submitted = st.form_submit_button("Valider les modifications")
        if submitted:
            rec.update(editable)
            rec = compute_derived_fields(rec)
            st.success("Champs mis à jour")

    st.markdown("### Résumé")
    st.dataframe(_record_to_df(rec), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        csv_buf = io.StringIO()
        pd.DataFrame([rec]).to_csv(csv_buf, index=False)
        st.download_button("⬇️ Télécharger CSV", csv_buf.getvalue(), file_name=uploaded.name.replace(".pdf",".csv"), mime="text/csv")
    with c2:
        st.download_button("⬇️ Télécharger JSON", json.dumps(rec, ensure_ascii=False, indent=2), file_name=uploaded.name.replace(".pdf",".json"), mime="application/json")

else:
    st.info("Déposez un fichier PDF pour démarrer.")
