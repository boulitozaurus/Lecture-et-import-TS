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

BULLET_RE = re.compile(r"^[•●▪·\-–]+[\s]*")
PAIR_SEP_RE = re.compile(r"^(.{2,80}?)\s*[:\-–]\s*(.+)$")

def _extract_words_safe(page):
    """Compat pdfplumber: essaie avec paramètres, puis fallback."""
    try:
        return page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False) or []
    except TypeError:
        try:
            return page.extract_words(x_tolerance=1.5, y_tolerance=1.5) or []
        except Exception:
            return page.extract_words() or []

def _kmeans_1d(xs, n_iter=30):
    """Petit k-means 1D (k=2) pour séparer gauche/droite sans dépendance externe."""
    xs = [float(x) for x in xs]
    xs_sorted = sorted(xs)
    c0 = xs_sorted[len(xs_sorted)//4]
    c1 = xs_sorted[3*len(xs_sorted)//4]
    c0, c1 = float(min(c0, c1)), float(max(c0, c1))
    for _ in range(n_iter):
        g0 = [x for x in xs if abs(x - c0) <= abs(x - c1)]
        g1 = [x for x in xs if abs(x - c0) >  abs(x - c1)]
        if not g0 or not g1:
            break
        new_c0 = sum(g0) / len(g0)
        new_c1 = sum(g1) / len(g1)
        if abs(new_c0 - c0) < 0.1 and abs(new_c1 - c1) < 0.1:
            c0, c1 = new_c0, new_c1
            break
        c0, c1 = new_c0, new_c1
    labels = [0 if abs(x - c0) <= abs(x - c1) else 1 for x in xs]
    if c0 > c1:  # s'assurer que 0 = gauche
        c0, c1 = c1, c0
        labels = [1 - l for l in labels]
    return (c0, c1), labels

def _build_lines_from_words(words, line_tol=None):
    """Regroupe mots -> lignes (pour UN cluster)."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    heights = [ (w["bottom"] - w["top"]) for w in words if "bottom" in w and "top" in w ]
    if line_tol is None:
        line_tol = max(2.0, (sorted(heights)[len(heights)//2] if heights else 6.0) / 2.0)
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
            lines.append((y, re.sub(r"\s+", " ", " ".join(cur["buf"]).strip()), cur["x0"], cur["x1"]))
            cur = {"top": w["top"], "bottom": w["bottom"], "x0": w["x0"], "x1": w["x1"], "buf":[w["text"]]}
    y = (cur["top"] + cur["bottom"]) / 2.0
    lines.append((y, re.sub(r"\s+", " ", " ".join(cur["buf"]).strip()), cur["x0"], cur["x1"]))
    return lines

def _split_value_segments(lines_texts):
    """
    Découpe la valeur en segments :
    - nouvelle entrée si puce (•, -, …) ou si 'X : Y'
    - sinon, concatène (wrapping).
    """
    segs, buf = [], []
    def flush():
        if buf:
            s = re.sub(r"\s+", " ", " ".join(buf).strip())
            if s:
                segs.append(s)
        buf.clear()
    for txt in lines_texts:
        if BULLET_RE.match(txt) or PAIR_SEP_RE.match(txt):
            flush()
            txt = BULLET_RE.sub("", txt, count=1).strip()
            if txt:
                segs.append(txt)
        else:
            buf.append(txt)
    flush()
    return segs if segs else [""]

def _pair_two_columns(page, y_pad=2.0):
    """
    Sépare d’abord les mots en 2 clusters X (gauche/droite) par k-means 1D,
    reconstruit les lignes dans CHAQUE cluster, puis associe
    chaque label gauche aux lignes droites suivantes jusqu’au prochain label.
    """
    words = _extract_words_safe(page)
    if not words:
        return _extract_text_pairs(page)

    centers = [ (w["x0"] + w["x1"]) / 2.0 for w in words ]
    (_, _), labels = _kmeans_1d(centers)

    left_words  = [w for w, l in zip(words, labels) if l == 0]
    right_words = [w for w, l in zip(words, labels) if l == 1]

    left_lines  = _build_lines_from_words(left_words)
    right_lines = _build_lines_from_words(right_words)

    left_lines.sort(key=lambda t: t[0])   # by y
    right_lines.sort(key=lambda t: t[0])  # by y

    tuples = []
    r_idx = 0
    for i, (y_label, label_txt, _, _) in enumerate(left_lines):
        next_y = left_lines[i+1][0] if i+1 < len(left_lines) else float("inf")

        # avancer l’index right jusqu’à la fenêtre du label
        while r_idx < len(right_lines) and right_lines[r_idx][0] < y_label - y_pad:
            r_idx += 1

        # collecter toutes les lignes droites jusqu’au prochain label
        bucket = []
        j = r_idx
        while j < len(right_lines) and right_lines[j][0] < next_y - y_pad:
            bucket.append(right_lines[j][1])
            j += 1

        if not bucket:
            continue

        segments = _split_value_segments(bucket)

        # Cas spécial: Modalités de remboursement → séparer “ … et … ”
        is_modalites = bool(re.search(r"(?i)modalit[éè]s?.*remboursement", label_txt))
        for seg in segments:
            if is_modalites and " et " in seg:
                for part in [p.strip() for p in seg.split(" et ") if p.strip()]:
                    tuples.append({"page": str(page.page_number), "label": label_txt, "value": part})
            else:
                tuples.append({"page": str(page.page_number), "label": label_txt, "value": seg})

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
