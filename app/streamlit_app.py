# --- MUST BE FIRST: reader-only app (no src imports) ---
from pathlib import Path
import streamlit as st
import pandas as pd
import pdfplumber
import re, statistics
import fitz  
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

BULLET_RE   = re.compile(r"^[•●▪·\-–]+[\s]*")
ENUM_RE     = re.compile(r"^(?:\d+[\.\)]|[a-z]\))\s+")
PAIR_SEP_RE = re.compile(r"^(.{2,120}?)\s*[:\-–]\s*(.+)$")
NOTE_RE     = re.compile(r"^note\s+\d+\b", re.I)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _fitz_get_lines(page: fitz.Page):
    """
    Retourne une liste de lignes avec bbox à partir de get_text('dict'):
    [{'y': centre_y, 'x0': x0, 'x1': x1, 'w': width, 'text': '...'}, ...]
    """
    d = page.get_text("dict")
    out = []
    for blk in d.get("blocks", []):
        for ln in blk.get("lines", []):
            x0 = min((sp["bbox"][0] for sp in ln.get("spans", []) ), default=None)
            x1 = max((sp["bbox"][2] for sp in ln.get("spans", []) ), default=None)
            y0 = min((sp["bbox"][1] for sp in ln.get("spans", []) ), default=None)
            y1 = max((sp["bbox"][3] for sp in ln.get("spans", []) ), default=None)
            if None in (x0, x1, y0, y1):
                continue
            text = _norm(" ".join(sp.get("text","") for sp in ln.get("spans", [])))
            if not text:
                continue
            out.append({
                "y": (y0 + y1) / 2.0,
                "x0": x0,
                "x1": x1,
                "w": x1 - x0,
                "text": text
            })
    return sorted(out, key=lambda r: (r["y"], r["x0"]))

def _kmeans_1d(xs, n_iter=25):
    """K-means 1D (k=2) pour séparer gauche/droite sur les centres X des LIGNES."""
    xs = [float(x) for x in xs]
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return (0.0, 0.0), []
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
            c0, c1 = nc0, nc1; break
        c0, c1 = nc0, nc1
    labels = [0 if abs(x - c0) <= abs(x - c1) else 1 for x in xs]
    if c0 > c1:  # garantir 0 = gauche
        c0, c1 = c1, c0
        labels = [1 - l for l in labels]
    return (c0, c1), labels

def _merge_left_labels_if_no_right_between(left, right, y_pad=2.0):
    """
    Certains labels sont sur 2 lignes ("Modalités de" + "remboursement").
    On fusionne des lignes LEFT contiguës s’il n’y a AUCUNE ligne RIGHT entre leurs Y.
    """
    merged = []
    i = 0
    while i < len(left):
        y0 = left[i]["y"]
        txt = left[i]["text"]
        j = i + 1
        while j < len(left):
            y1 = left[j]["y"]
            has_right_between = any((y0 - y_pad) <= r["y"] < (y1 - y_pad) for r in right)
            if has_right_between:
                break
            txt += " " + left[j]["text"]
            j += 1
        merged.append({"y": y0, "text": _norm(txt)})
        i = j
    return merged

def _split_value_segments(lines_texts):
    """
    Valeur -> segments :
    - nouveau segment si puce (•, -, – …) ou si 'X : Y'
    - sinon, concatène (wrapping).
    """
    segs, buf = [], []
    def flush():
        if buf:
            s = _norm(" ".join(buf))
            if s:
                segs.append(s)
        buf.clear()
    for t in lines_texts:
        if BULLET_RE.match(t) or PAIR_SEP_RE.match(t):
            flush()
            t2 = BULLET_RE.sub("", t, count=1).strip()
            if t2:
                segs.append(t2)
        else:
            buf.append(t)
    flush()
    return segs if segs else [""]

def _two_column_pairs_with_fitz(page: fitz.Page, y_pad=3.0):
    """
    Lignes (bbox) -> 2 colonnes (k-means) -> reclasser puces/énum/notes à droite ->
    fusion labels multilignes -> associer RIGHT entre label et label+1 -> découper en segments.
    """
    lines = _fitz_get_lines(page)
    if not lines:
        return []

    # 1) k-means sur centres X des lignes
    xmid = [ (ln["x0"] + ln["x1"]) / 2.0 for ln in lines ]
    (cL, cR), labs = _kmeans_1d(xmid)

    left0  = [ln for ln, L in zip(lines, labs) if L == 0]
    right0 = [ln for ln, L in zip(lines, labs) if L == 1]
    med_lw = statistics.median([ln["w"] for ln in left0])  if left0  else 30.0
    med_rw = statistics.median([ln["w"] for ln in right0]) if right0 else 80.0
    split  = (cL + cR) / 2.0
    margin = max(4.0, (med_lw + med_rw) / 10.0)

    # 2) reclassement agressif des lignes "valeur"
    left, right = [], []
    for ln in lines:
        xm = (ln["x0"] + ln["x1"]) / 2.0
        txt = ln["text"]
        width = ln["w"]

        force_right = False
        if BULLET_RE.match(txt) or ENUM_RE.match(txt) or NOTE_RE.match(txt):
            force_right = True
        if width > max(med_rw*0.7, med_lw*1.3):   # très large pour un label
            force_right = True
        if xm > split + margin:                    # nettement à droite
            force_right = True

        if force_right:
            right.append(ln)
        else:
            if xm < split - margin:
                left.append(ln)
            else:
                # zone ambigüe: privilégie droite si plutôt large
                (right if width >= med_lw else left).append(ln)

    left  = sorted(left, key=lambda d: d["y"])
    right = sorted(right, key=lambda d: d["y"])

    # 3) fusion labels multilignes si aucune ligne droite entre-deux
    left = _merge_left_labels_if_no_right_between(left, right, y_pad=y_pad)

    # 4) association label -> valeurs (toutes RIGHT entre y(label) et y(label+1))
    tuples = []
    ridx = 0
    for i, lab in enumerate(left):
        y0 = lab["y"]
        y1 = left[i+1]["y"] if i+1 < len(left) else float("inf")
        # avancer
        while ridx < len(right) and right[ridx]["y"] < y0 - y_pad:
            ridx += 1
        bucket = []
        j = ridx
        while j < len(right) and right[j]["y"] < y1 - y_pad:
            bucket.append(right[j]["text"])
            j += 1
        if not bucket:
            continue

        segments = _split_value_segments(bucket)

        is_modalites = bool(re.search(r"(?i)modalit[éè]s?.*remboursement", lab["text"]))
        for seg in segments:
            if is_modalites and " et " in seg:
                parts = [p.strip() for p in seg.split(" et ") if p.strip()]
                for part in parts:
                    tuples.append({"page": str(page.number + 1), "label": lab["text"], "value": part})
            else:
                tuples.append({"page": str(page.number + 1), "label": lab["text"], "value": seg})

    return tuples

def _extract_footnotes_fitz(page: fitz.Page):
    """Notes simples : lignes qui commencent par '1 ', '2 ', ... ou 'Note N'."""
    text = page.get_text("text") or ""
    out = []
    cur_num, buf = None, []
    for raw in text.splitlines():
        line = _norm(raw)
        m = re.match(r"^(\d+)\s+(.*)$", line)
        if NOTE_RE.match(line):
            # flush précédente
            if cur_num is not None and buf:
                out.append({"page": str(page.number + 1), "label": f"Note {cur_num}", "value": _norm(' '.join(buf))})
            cur_num = re.sub(r"^note\s+", "", line.split()[1], flags=re.I) if len(line.split())>1 else "?"
            buf = [" ".join(line.split()[2:])] if len(line.split())>2 else []
            continue
        if m:
            if cur_num is not None and buf:
                out.append({"page": str(page.number + 1), "label": f"Note {cur_num}", "value": _norm(' '.join(buf))})
            cur_num = m.group(1)
            buf = [m.group(2)]
        else:
            if cur_num is not None:
                buf.append(line)
    if cur_num is not None and buf:
        out.append({"page": str(page.number + 1), "label": f"Note {cur_num}", "value": _norm(' '.join(buf))})
    return out

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
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            tuples.extend(_two_column_pairs_with_fitz(page))   # 2 colonnes robustes
            tuples.extend(_extract_footnotes_fitz(page))       # notes (un tuple/note)

    # Affichage (full texte)
    def _to_df(rows):
        if not rows:
            return pd.DataFrame(columns=["page","label","value"]).astype("string")
        df = pd.DataFrame(rows)[["page","label","value"]]
        for c in df.columns:
            df[c] = df[c].astype("string")
        return df

    st.subheader("Tuples extraits (bruts, sans mapping)")
    st.caption("Chaque ligne = (page, label brut, valeur brute). Puces & sous-paires séparées.")
    st.dataframe(_to_df(tuples), use_container_width=True)
else:
    st.info("Dépose un PDF pour démarrer.")

