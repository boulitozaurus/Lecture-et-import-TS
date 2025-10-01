# --- MUST BE FIRST: project root for outputs (no src imports needed here) ---
from pathlib import Path
import streamlit as st
import pandas as pd
import pdfplumber
import re

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]           # racine du repo (contient app/ et src/)
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="TS – Lecteur de tuples", layout="wide")
st.title("📄 Lecteur PDF → Tuples (Term Sheet) — mode lecture uniquement")

# -------------------------------------------------------------------
# Utilitaires d'extraction (aucun mapping, aucun fields.yaml requis)
# -------------------------------------------------------------------
_PAIR_LINE = re.compile(r"(.{3,120}?)\s*[:\-–]\s*(.+)$")

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _extract_tables(page: pdfplumber.page.Page):
    """Renvoie une liste de dicts (tuples) pour les tableaux 2 colonnes."""
    out = []
    try:
        tables = page.extract_tables() or []
    except Exception:
        tables = []
    for r_idx, row in enumerate(tables):
        if not row or len(row) < 2:
            continue
        left = _normalize_space(str(row[0] or ""))
        right = _normalize_space(str(row[1] or ""))
        if not left or not right:
            continue
        out.append({
            "page": str(page.page_number),
            "source": "table",
            "label": left,
            "value": right,
            "row": str(r_idx),
            "col": "0",
            "bbox": "",          # non utilisé ici (on reste simple/fiable)
        })
    return out

def _extract_text_pairs(page: pdfplumber.page.Page):
    """Renvoie des dicts (tuples) pour les lignes TEXT de type 'label : valeur'."""
    out = []
    text = page.extract_text() or ""
    for line in text.splitlines():
        line = _normalize_space(line)
        m = _PAIR_LINE.match(line)
        if not m:
            continue
        left, right = m.group(1), m.group(2)
        out.append({
            "page": str(page.page_number),
            "source": "text",
            "label": _normalize_space(left),
            "value": _normalize_space(right),
            "row": "",
            "col": "",
            "bbox": "",
        })
    return out

def _tuples_to_df(tuples):
    """DataFrame 100% string → pas de conversion numérique → pas d’Overflow pyarrow."""
    if not tuples:
        return pd.DataFrame(columns=["page","source","label","value","row","col","bbox"])
    df = pd.DataFrame(tuples)
    # forcer string partout
    for c in df.columns:
        df[c] = df[c].astype("string")
    # colonnes dans un ordre lisible
    cols = ["page","source","label","value","row","col","bbox"]
    df = df[[c for c in cols if c in df.columns]]
    return df

def _summary_by_page(tuples):
    """Petit récap : volume de tuples par page et par source."""
    if not tuples:
        return pd.DataFrame(columns=["page","tables","text","total"])
    rows = {}
    for t in tuples:
        p = t["page"]
        src = t["source"]
        rows.setdefault(p, {"tables":0, "text":0})
        rows[p][src] += 1
    data = []
    for p, d in sorted(rows.items(), key=lambda kv: int(kv[0])):
        total = d["tables"] + d["text"]
        data.append({"page": str(p), "tables": str(d["tables"]), "text": str(d["text"]), "total": str(total)})
    df = pd.DataFrame(data, columns=["page","tables","text","total"]).astype("string")
    return df

# -------------------------------------------------------------------
# UI minimale : upload → tuples (aucune règle, aucun mapping)
# -------------------------------------------------------------------
uploaded = st.file_uploader("Dépose un PDF de Term Sheet", type=["pdf"])

if uploaded:
    pdf_path = OUTPUTS / uploaded.name
    pdf_path.write_bytes(uploaded.getvalue())

    with st.spinner("Lecture du PDF…"):
        tuples = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                tuples.extend(_extract_tables(page))
                tuples.extend(_extract_text_pairs(page))

    # Récap volumes
    st.subheader("Résumé de l’extraction")
    st.dataframe(_summary_by_page(tuples), use_container_width=True)

    # Tuples bruts
    st.subheader("Tuples extraits (bruts, sans mapping)")
    st.caption("Chaque ligne = (page, source, label brut, valeur brute, meta).")
    st.dataframe(_tuples_to_df(tuples), use_container_width=True)

else:
    st.info("Dépose un PDF pour démarrer.")
