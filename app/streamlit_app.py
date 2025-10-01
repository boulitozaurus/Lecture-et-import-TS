# --- MUST BE FIRST: reader-only app (no src imports) ---
from pathlib import Path
import streamlit as st
import pandas as pd
import pdfplumber
import re, statistics
# 👉 ajoute (en haut du fichier, avec les autres imports) :
try:
    import numpy as np
except Exception:
    np = None


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

BULLET_RE = re.compile(r"^[•●▪·\-–]+[\s]*")
PAIR_SEP_RE = re.compile(r"^(.{2,80}?)\s*[:\-–]\s*(.+)$")

def _extract_words_safe(page):
    """Extraction de mots compatible avec plusieurs versions de pdfplumber."""
    try:
        return page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False) or []
    except TypeError:
        try:
            return page.extract_words(x_tolerance=1.5, y_tolerance=1.5) or []
        except Exception:
            return page.extract_words() or []

def _find_split_x(words):
    """Trouve la césure gauche/droite via histogramme des centres X (fallback médiane)."""
    centers = [ (w["x0"] + w["x1"]) / 2.0 for w in words ]
    if not centers:
        return None
    if np is not None:
        try:
            hist, edges = np.histogram(centers, bins=40)
            if hist.sum() > 0:
                n = len(hist)
                lh = int(np.argmax(hist[:n//2])) if n >= 2 else 0
                rh = int(np.argmax(hist[n//2:])) + n//2 if n >= 2 else 0
                if 0 <= lh < rh-1:
                    valley_idx = lh + 1 + int(np.argmin(hist[lh+1:rh]))
                    return float((edges[valley_idx] + edges[valley_idx+1]) / 2.0)
        except Exception:
            pass
    return float(statistics.median(centers))

def _group_lines(words):
    """Regroupe chars en lignes (basé sur Y) et renvoie des dicts {'y','x0','text'} triés."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    heights = [ (w["bottom"] - w["top"]) for w in words if "bottom" in w and "top" in w ]
    line_tol = max(2.0, statistics.median(heights)/2.0) if heights else 3.0
    lines = []
    cur = {"top": words[0]["top"], "bottom": words[0]["bottom"], "x0": words[0]["x0"], "x1": words[0]["x1"], "buf":[words[0]["text"]]}
    for w in words[1:]:
        if abs(w["top"] - cur["top"]) <= line_tol:
            cur["buf"].append(w["text"])
            cur["x0"] = min(cur["x0"], w["x0"])
            cur["x1"] = max(cur["x1"], w["x1"])
            cur["bottom"] = max(cur["bottom"], w["bottom"])
        else:
            y = (cur["top"] + cur["bottom"]) / 2.0
            lines.append({"y": y, "x0": cur["x0"], "text": re.sub(r"\s+", " ", " ".join(cur["buf"]).strip())})
            cur = {"top": w["top"], "bottom": w["bottom"], "x0": w["x0"], "x1": w["x1"], "buf":[w["text"]]}
    y = (cur["top"] + cur["bottom"]) / 2.0
    lines.append({"y": y, "x0": cur["x0"], "text": re.sub(r"\s+", " ", " ".join(cur["buf"]).strip())})
    return [ln for ln in lines if ln["text"]]

def _split_value_segments(lines):
    """
    Découpe une séquence de lignes droite en segments de valeur :
    - nouvelle valeur si la ligne commence par une puce
    - ou si 'X : Y' (sous-paire)
    Sinon, on concatène (wrapping).
    """
    segs = []
    buf = []
    def flush():
        if buf:
            val = re.sub(r"\s+", " ", " ".join(buf).strip())
            if val:
                segs.append(val)
        buf.clear()
    for ln in lines:
        t = ln["text"]
        if BULLET_RE.match(t) or PAIR_SEP_RE.match(t):
            flush()
            t = BULLET_RE.sub("", t, count=1).strip()
            if t:
                segs.append(t)
        else:
            buf.append(t)
    flush()
    return segs if segs else [""]  # au moins un segment vide si rien

def _pair_two_columns(page, margin=8.0):
    """
    Coupage 2 colonnes :
    - labels = lignes dont x0 <= split_x - margin
    - valeurs = lignes dont x0 >= split_x + margin
    Chaque label collecte toutes les valeurs jusqu'au prochain label (fenêtre [y_label, y_next_label)).
    """
    words = _extract_words_safe(page)
    if not words:
        # Fallback lecture "label: valeur" depuis texte simple
        return _extract_text_pairs(page)

    split_x = _find_split_x(words)
    if split_x is None:
        return _extract_text_pairs(page)

    lines = sorted(_group_lines(words), key=lambda d: d["y"])
    left  = [ln for ln in lines if ln["x0"] <= split_x - margin]
    right = [ln for ln in lines if ln["x0"] >= split_x + margin]
    if not left and not right:
        return _extract_text_pairs(page)

    tuples = []

    # Indexation des lignes droite par Y (pour fenêtre facile)
    right_by_y = right

    for i, lab in enumerate(left):
        y0 = lab["y"]
        y1 = left[i+1]["y"] if i+1 < len(left) else float("inf")

        # toutes les lignes droite entre y0-δ et y1-ε
        bucket = [ln for ln in right_by_y if y0 - 1.5 <= ln["y"] < y1 - 0.1]
        if not bucket:
            continue

        # Scinder en segments (puces, sous-paires)
        segments = _split_value_segments(bucket)

        # Cas spécial 'Modalités de remboursement' : séparer "… et …"
        is_modalites = bool(re.search(r"(?i)modalit[éè]s?.*remboursement", lab["text"]))
        for seg in segments:
            if is_modalites and " et " in seg:
                for part in [p.strip() for p in seg.split(" et ") if p.strip()]:
                    tuples.append({"page": str(page.page_number), "label": lab["text"], "value": part})
            else:
                tuples.append({"page": str(page.page_number), "label": lab["text"], "value": seg})

    return tuples

def _extract_text_pairs(page):
    """Fallback : parse les lignes 'label : valeur' du texte simple."""
    out = []
    text = page.extract_text() or ""
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw or "").strip()
        m = PAIR_SEP_RE.match(line)
        if m:
            out.append({"page": str(page.page_number), "label": m.group(1).strip(), "value": m.group(2).strip()})
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
