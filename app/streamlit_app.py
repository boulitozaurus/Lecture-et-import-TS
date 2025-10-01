# --- MUST BE FIRST: reader-only app (no src imports) ---
from pathlib import Path
import streamlit as st
import pandas as pd
import pdfplumber
import re
import statistics

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="TS – Lecteur robuste → Tuples", layout="wide")
st.title("📄 Lecteur PDF → Tuples (robuste, sans mapping)")

# ---------------------------------------------
# Utils (aucun mapping; reconstruction 2 colonnes)
# ---------------------------------------------
BULLET_RE = re.compile(r"^[•●▪\-–]")   # puces courantes
PAIR_SEP_RE = re.compile(r"(.{3,120}?)\s*[:\-–]\s*(.+)$")  # fallback "label: valeur"

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _build_lines_from_words(words, line_tol=3.0):
    """Regroupe les mots par 'lignes' (y proches), et renvoie [(y, text, x0_min, x1_max), ...]."""
    if not words:
        return []
    # tri par y puis x
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines = []
    cur = {"top": words[0]["top"], "bottom": words[0]["bottom"], "text": [], "x0": words[0]["x0"], "x1": words[0]["x1"]}
    for w in words:
        if abs(w["top"] - cur["top"]) <= line_tol:
            cur["text"].append(w["text"])
            cur["x0"] = min(cur["x0"], w["x0"])
            cur["x1"] = max(cur["x1"], w["x1"])
            cur["bottom"] = max(cur["bottom"], w["bottom"])
        else:
            y = (cur["top"] + cur["bottom"]) / 2.0
            lines.append((y, _norm(" ".join(cur["text"])), cur["x0"], cur["x1"]))
            cur = {"top": w["top"], "bottom": w["bottom"], "text": [w["text"]], "x0": w["x0"], "x1": w["x1"]}
    y = (cur["top"] + cur["bottom"]) / 2.0
    lines.append((y, _norm(" ".join(cur["text"])), cur["x0"], cur["x1"]))
    return lines

def _pair_two_columns(page, y_tol=5.0):
    """
    Reconstruit la table 2 colonnes en se basant sur la position X des mots.
    Robuste aux versions de pdfplumber : si extract_words(...) lève TypeError,
    on réessaie sans paramètres ; si pas de mots -> fallback texte (label: valeur).
    """
    # 1) Récupération des mots (robuste aux versions)
    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False) or []
    except TypeError:
        # Anciennes versions : certains kwargs ne sont pas supportés
        try:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5) or []
        except Exception:
            words = page.extract_words() or []

    # 2) Si aucun mot (PDF scanné, ou extraction impossible) -> fallback texte simple
    if not words:
        return _extract_text_pairs(page)

    # ---------- reconstruction 2 colonnes ----------
    import statistics
    centers = [ (w["x0"] + w["x1"]) / 2.0 for w in words ]
    mid_x = statistics.median(centers)

    def _build_lines_from_words(words, line_tol=3.0):
        words = sorted(words, key=lambda w: (w["top"], w["x0"]))
        if not words:
            return []
        lines = []
        cur = {"top": words[0]["top"], "bottom": words[0]["bottom"], "text": [], "x0": words[0]["x0"], "x1": words[0]["x1"]}
        for w in words:
            if abs(w["top"] - cur["top"]) <= line_tol:
                cur["text"].append(w["text"])
                cur["x0"] = min(cur["x0"], w["x0"])
                cur["x1"] = max(cur["x1"], w["x1"])
                cur["bottom"] = max(cur["bottom"], w["bottom"])
            else:
                y = (cur["top"] + cur["bottom"]) / 2.0
                lines.append((y, _norm(" ".join(cur["text"])), cur["x0"], cur["x1"]))
                cur = {"top": w["top"], "bottom": w["bottom"], "text": [w["text"]], "x0": w["x0"], "x1": w["x1"]}
        y = (cur["top"] + cur["bottom"]) / 2.0
        lines.append((y, _norm(" ".join(cur["text"])), cur["x0"], cur["x1"]))
        return lines

    left_words  = [w for w in words if (w["x0"] + w["x1"]) / 2.0 <= mid_x]
    right_words = [w for w in words if (w["x0"] + w["x1"]) / 2.0 >  mid_x]

    left_lines  = _build_lines_from_words(left_words)
    right_lines = _build_lines_from_words(right_words)

    # fusion des lignes par Y croissant
    events = []
    for y, txt, x0, x1 in left_lines:
        if txt:
            events.append((y, "L", txt))
    for y, txt, x0, x1 in right_lines:
        if txt:
            events.append((y, "R", txt))
    events.sort(key=lambda e: e[0])

    BULLET_RE = re.compile(r"^[•●▪\-–]")
    PAIR_SEP_RE = re.compile(r"(.{3,120}?)\s*[:\-–]\s*(.+)$")

    tuples = []
    cur_label = None
    buffer_lines = []

    def flush_buffer():
        nonlocal buffer_lines, cur_label, tuples
        if cur_label and buffer_lines:
            val = _norm(" ".join(buffer_lines))
            if val:
                tuples.append({"page": str(page.page_number), "label": cur_label, "value": val})
        buffer_lines = []

    last_y = None
    for y, kind, txt in events:
        if last_y is not None and abs(y - last_y) <= y_tol:
            pass
        last_y = y

        if kind == "L":
            flush_buffer()
            cur_label = txt
        else:  # "R"
            if not cur_label:
                m = PAIR_SEP_RE.match(txt)
                if m:
                    tuples.append({"page": str(page.page_number), "label": _norm(m.group(1)), "value": _norm(m.group(2))})
                continue

            if BULLET_RE.match(txt) or (":" in txt[:25]):
                flush_buffer()
                val = _norm(re.sub(BULLET_RE, "", txt, count=1))
                if val:
                    tuples.append({"page": str(page.page_number), "label": cur_label, "value": val})
            else:
                buffer_lines.append(txt)

    flush_buffer()

    # Règle utile : séparer les 'Modalités de remboursement' en deux tuples si '... et ...'
    out = []
    for t in tuples:
        if re.search(r"(?i)modalit[éè]s?.*remboursement", t["label"]) and " et " in t["value"]:
            parts = [ _norm(p) for p in t["value"].split(" et ") if _norm(p) ]
            for p in parts:
                out.append({"page": t["page"], "label": t["label"], "value": p})
        else:
            out.append(t)
    return out

def _extract_footnotes(page):
    """Notes de bas de page: lignes commençant par '1 ', '2 ', ... (un tuple par note)."""
    text = page.extract_text() or ""
    notes = []
    cur_num, cur_buf = None, []
    for raw in text.splitlines():
        line = _norm(raw)
        m = re.match(r"^([0-9]+)\s+(.*)$", line)
        if m:
            # nouvelle note -> flush précédente
            if cur_num is not None and cur_buf:
                notes.append({"page": str(page.page_number), "label": f"Note {cur_num}", "value": _norm(" ".join(cur_buf))})
            cur_num = m.group(1)
            cur_buf = [m.group(2)]
        else:
            if cur_num is not None:
                cur_buf.append(line)
    if cur_num is not None and cur_buf:
        notes.append({"page": str(page.page_number), "label": f"Note {cur_num}", "value": _norm(" ".join(cur_buf))})
    return notes

def _summary_by_page(tuples):
    """Récap : nombre de tuples par page."""
    if not tuples:
        return pd.DataFrame(columns=["page","total"]).astype("string")
    counts = {}
    for t in tuples:
        p = str(t.get("page","?"))
        counts[p] = counts.get(p, 0) + 1
    data = [{"page": p, "total": str(counts[p])} for p in sorted(counts, key=lambda x: int(x) if x.isdigit() else 0)]
    return pd.DataFrame(data, columns=["page","total"]).astype("string")

def _to_df(tuples):
    if not tuples:
        return pd.DataFrame(columns=["page","label","value"]).astype("string")
    df = pd.DataFrame(tuples)[["page","label","value"]]
    for c in df.columns:
        df[c] = df[c].astype("string")
    return df

# ---------------------------------------------
# UI minimale : upload → tuples robustes
# ---------------------------------------------
uploaded = st.file_uploader("Dépose un PDF de Term Sheet", type=["pdf"])

if uploaded:
    pdf_path = OUTPUTS / uploaded.name
    pdf_path.write_bytes(uploaded.getvalue())

    tuples = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # 1) reconstruction 2 colonnes robuste (mots → lignes → appariement)
            tuples.extend(_pair_two_columns(page))
            # 2) notes de bas de page
            tuples.extend(_extract_footnotes(page))

    st.subheader("Résumé de l’extraction")
    st.dataframe(_summary_by_page(tuples), use_container_width=True)

    st.subheader("Tuples extraits (bruts, sans mapping)")
    st.caption("Chaque ligne = (page, label brut, valeur brute). Puces séparées en entrées distinctes.")
    st.dataframe(_to_df(tuples), use_container_width=True)

else:
    st.info("Dépose un PDF pour démarrer.")
