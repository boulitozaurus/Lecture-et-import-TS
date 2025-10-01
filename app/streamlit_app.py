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

import re, statistics

import re, statistics

BULLET_RE   = re.compile(r"^[•●▪·\-–]+[\s]*")
PAIR_SEP_RE = re.compile(r"^(.{2,120}?)\s*[:\-–]\s*(.+)$")

def _extract_words_safe(page):
    """Compat pdfplumber (kwargs varient selon versions)."""
    try:
        return page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False) or []
    except TypeError:
        try:
            return page.extract_words(x_tolerance=1.5, y_tolerance=1.5) or []
        except Exception:
            return page.extract_words() or []

def _group_lines(words):
    """Mots -> lignes (y proches). Retourne liste de dicts {'y','x0','x1','text'}."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    heights = [ (w["bottom"] - w["top"]) for w in words if "bottom" in w and "top" in w ]
    line_tol = max(2.0, (sorted(heights)[len(heights)//2] if heights else 6.0)/2.0)

    lines = []
    cur = {"top": words[0]["top"], "bottom": words[0]["bottom"], "x0": words[0]["x0"], "x1": words[0]["x1"], "buf":[words[0]["text"]]}
    for w in words[1:]:
        if abs(w["top"] - cur["top"]) <= line_tol:
            cur["buf"].append(w["text"])
            cur["x0"] = min(cur["x0"], w["x0"])
            cur["x1"] = max(cur["x1"], w["x1"])
            cur["bottom"] = max(cur["bottom"], w["bottom"])
        else:
            y  = (cur["top"] + cur["bottom"]) / 2.0
            tx = re.sub(r"\s+", " ", " ".join(cur["buf"]).strip())
            if tx:
                lines.append({"y": y, "x0": cur["x0"], "x1": cur["x1"], "text": tx})
            cur = {"top": w["top"], "bottom": w["bottom"], "x0": w["x0"], "x1": w["x1"], "buf":[w["text"]]}
    y  = (cur["top"] + cur["bottom"]) / 2.0
    tx = re.sub(r"\s+", " ", " ".join(cur["buf"]).strip())
    if tx:
        lines.append({"y": y, "x0": cur["x0"], "x1": cur["x1"], "text": tx})

    return sorted(lines, key=lambda d: d["y"])

def _kmeans_1d(xs, n_iter=30):
    """K-means 1D k=2 maison (stable, pas de dépendance)."""
    xs = [float(x) for x in xs]
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return (0.0, 0.0), [0]*0
    c0 = xs_sorted[len(xs_sorted)//4]
    c1 = xs_sorted[3*len(xs_sorted)//4]
    c0, c1 = float(min(c0, c1)), float(max(c0, c1))
    for _ in range(n_iter):
        g0 = [x for x in xs if abs(x - c0) <= abs(x - c1)]
        g1 = [x for x in xs if abs(x - c0) >  abs(x - c1)]
        if not g0 or not g1:
            break
        nc0 = sum(g0)/len(g0); nc1 = sum(g1)/len(g1)
        if abs(nc0-c0) < 0.1 and abs(nc1-c1) < 0.1:
            c0, c1 = nc0, nc1
            break
        c0, c1 = nc0, nc1
    labels = [0 if abs(x - c0) <= abs(x - c1) else 1 for x in xs]
    if c0 > c1:                 # s'assurer que 0 = gauche
        c0, c1 = c1, c0
        labels = [1-l for l in labels]
    return (c0, c1), labels

def _split_value_segments(lines_texts):
    """
    Valeur -> segments :
    - nouvelle entrée si puce (•, -, – …) ou si 'X : Y'
    - sinon concatène (wrapping).
    """
    segs, buf = [], []
    def flush():
        if buf:
            s = re.sub(r"\s+", " ", " ".join(buf).strip())
            if s:
                segs.append(s)
        buf.clear()

    for t in lines_texts:
        if BULLET_RE.match(t) or PAIR_SEP_RE.match(t):
            flush()
            t = BULLET_RE.sub("", t, count=1).strip()
            if t:
                segs.append(t)
        else:
            buf.append(t)
    flush()
    return segs if segs else [""]

def _extract_text_pairs(page):
    """Fallback ultra-simple: 'label : valeur' sur texte brut."""
    out = []
    text = page.extract_text() or ""
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw or "").strip()
        m = PAIR_SEP_RE.match(line)
        if m:
            out.append({"page": str(page.page_number), "label": m.group(1).strip(), "value": m.group(2).strip()})
    return out

def _pair_two_columns(page, y_pad=2.0):
    """
    1) mots -> lignes
    2) k-means 1D sur x0 des LIGNES -> 2 clusters (gauche/droite)
    3) re-classe en droite toute ligne qui commence par une puce
    4) associe chaque label gauche aux lignes droites jusqu’au label suivant
    5) découpe la valeur en segments (puces, X:Y, “et” pour Modalités)
    """
    words = _extract_words_safe(page)
    if not words:
        return _extract_text_pairs(page)

    lines = _group_lines(words)
    if not lines:
        return _extract_text_pairs(page)

    xs = [ln["x0"] for ln in lines]
    (_, _), lab = _kmeans_1d(xs)
    left_lines  = [ln for ln, L in zip(lines, lab) if L == 0]
    right_lines = [ln for ln, L in zip(lines, lab) if L == 1]

    # Re-classer en "droite" toute ligne qui commence par une puce
    moved = []
    keep_left = []
    for ln in left_lines:
        if BULLET_RE.match(ln["text"]):
            moved.append(ln)
        else:
            keep_left.append(ln)
    left_lines  = keep_left
    right_lines = sorted(right_lines + moved, key=lambda d: d["y"])

    left_lines.sort(key=lambda d: d["y"])

    tuples = []
    r_idx = 0
    for i, lab_ln in enumerate(left_lines):
        y0 = lab_ln["y"]
        y1 = left_lines[i+1]["y"] if i+1 < len(left_lines) else float("inf")

        # avancer le pointeur droite jusqu’à la fenêtre du label
        while r_idx < len(right_lines) and right_lines[r_idx]["y"] < y0 - y_pad:
            r_idx += 1

        bucket = []
        j = r_idx
        while j < len(right_lines) and right_lines[j]["y"] < y1 - y_pad:
            bucket.append(right_lines[j]["text"])
            j += 1

        if not bucket:
            continue

        segs = _split_value_segments(bucket)
        is_modalites = bool(re.search(r"(?i)modalit[éè]s?.*remboursement", lab_ln["text"]))
        for seg in segs:
            if is_modalites and " et " in seg:
                for part in [p.strip() for p in seg.split(" et ") if p.strip()]:
                    tuples.append({"page": str(page.page_number), "label": lab_ln["text"], "value": part})
            else:
                tuples.append({"page": str(page.page_number), "label": lab_ln["text"], "value": seg})

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
