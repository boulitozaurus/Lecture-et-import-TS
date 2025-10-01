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
import fitz
import re, statistics

BULLET_RE   = re.compile(r"^[•●▪·\-–]+[\s]*")
ENUM_RE     = re.compile(r"^(?:\d+[\.\)]|[a-z]\))\s+")
NOTE_RE     = re.compile(r"^note\s+\d+\b", re.I)
PAIR_SEP_RE = re.compile(r"^(.{2,120}?)\s*[:\-–]\s*(.+)$")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _get_lines_with_chars(page: fitz.Page):
    """
    Retourne une liste de lignes avec:
      - bbox moyenne (x0,x1,y0,y1),
      - texte concaténé,
      - liste 'chars' = [(cx0,cx1,cy0,cy1,char), ...]
    """
    d = page.get_text("rawdict")
    lines = []
    for blk in d.get("blocks", []):
        for li in blk.get("lines", []):
            spans = li.get("spans", [])
            chars = []
            x0 = y0 = float("inf")
            x1 = y1 = float("-inf")
            buf = []
            for sp in spans:
                for ch in sp.get("chars", []):
                    (cx0, cy0, cx1, cy1) = ch["bbox"]
                    x0 = min(x0, cx0); y0 = min(y0, cy0)
                    x1 = max(x1, cx1); y1 = max(y1, cy1)
                    buf.append(ch.get("c",""))
                    chars.append((cx0, cx1, cy0, cy1, ch.get("c","")))
            text = _norm("".join(buf))
            if text:
                lines.append({
                    "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                    "y":  (y0+y1)/2.0,
                    "w":  x1-x0,
                    "text": text,
                    "chars": chars
                })
    return sorted(lines, key=lambda r: (r["y"], r["x0"]))

def _compute_split_x_from_char_density(lines, bins=96):
    """
    Projection horizontale de la densité (largeur cumulée des caractères) → on cherche une vallée.
    """
    if not lines:
        return None
    xmin = min(l["x0"] for l in lines); xmax = max(l["x1"] for l in lines)
    if xmax <= xmin:
        return None
    step = (xmax - xmin) / bins
    hist = [0.0] * bins

    for l in lines:
        for (cx0, cx1, _, _, _) in l["chars"]:
            i0 = int(max(0, min(bins-1, (cx0 - xmin) // step)))
            i1 = int(max(0, min(bins-1, (cx1 - xmin) // step)))
            for i in range(i0, i1+1):
                hist[i] += 1.0

    # deux masses → vallée entre pics
    # on coupe au minimum local le plus profond vers le milieu
    mid = bins//2
    left_peak  = max(range(0, mid), key=lambda i: hist[i]) if any(hist[:mid]) else 0
    right_peak = max(range(mid, bins), key=lambda i: hist[i]) if any(hist[mid:]) else bins-1
    if right_peak - left_peak >= 4:
        valley = min(range(left_peak+1, right_peak), key=lambda i: hist[i])
        return xmin + valley * step
    # fallback : médiane des centres
    centers = [ (l["x0"]+l["x1"])/2.0 for l in lines ]
    return statistics.median(centers)

def _char_coverage_ratio(line, split_x):
    """Retourne (ratio_gauche, ratio_droite) en sommant la largeur de char de part et d’autre du split."""
    left = right = 0.0
    for (cx0, cx1, _, _, _) in line["chars"]:
        midc = (cx0+cx1)/2.0
        w = max(0.0, cx1 - cx0)
        if midc <= split_x:
            left += w
        else:
            right += w
    tot = left + right
    if tot <= 0:
        return 0.5, 0.5
    return left/tot, right/tot

def _merge_left_labels_if_no_right_between(left, right, y_pad=2.0):
    """Fusionne les labels multilignes si aucune ligne droite ne s'intercale entre les deux."""
    merged = []
    i = 0
    while i < len(left):
        y0 = left[i]["y"]; txt = left[i]["text"]
        j = i + 1
        while j < len(left):
            y1 = left[j]["y"]
            if any((y0 - y_pad) <= r["y"] < (y1 - y_pad) for r in right):
                break
            txt += " " + left[j]["text"]
            j += 1
        merged.append({"y": y0, "text": _norm(txt)})
        i = j
    return merged

def _split_value_segments(lines_texts):
    """Découpe la valeur en segments (puces, 'X : Y'); sinon concatène."""
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

def _two_column_pairs_char_coverage(page: fitz.Page, y_pad=3.0, left_threshold=0.6):
    """
    Classe chaque ligne par *couverture de caractères* gauche/droite.
    → labels à gauche (> left_threshold), valeurs à droite.
    """
    lines = _get_lines_with_chars(page)
    if not lines:
        return []

    split_x = _compute_split_x_from_char_density(lines)
    if split_x is None:
        # fallback texte brut simple "label : valeur"
        txt = page.get_text("text") or ""
        out = []
        for raw in txt.splitlines():
            m = PAIR_SEP_RE.match(_norm(raw))
            if m:
                out.append({"page": str(page.number + 1), "label": m.group(1).strip(), "value": m.group(2).strip()})
        return out

    left_lines, right_lines = [], []
    for ln in lines:
        lg, rg = _char_coverage_ratio(ln, split_x)
        txt = ln["text"]

        # forcer à droite les lignes "valeur évidente"
        force_right = BULLET_RE.match(txt) or ENUM_RE.match(txt) or NOTE_RE.match(txt)
        if force_right:
            right_lines.append({"y": ln["y"], "text": txt})
        else:
            if lg >= left_threshold:
                left_lines.append({"y": ln["y"], "text": txt})
            else:
                right_lines.append({"y": ln["y"], "text": txt})

    left_lines  = sorted(left_lines, key=lambda d: d["y"])
    right_lines = sorted(right_lines, key=lambda d: d["y"])

    # fusion labels multilignes
    left_lines = _merge_left_labels_if_no_right_between(left_lines, right_lines, y_pad=y_pad)

    # associer label -> valeurs
    tuples = []
    ridx = 0
    for i, lab in enumerate(left_lines):
        y0 = lab["y"]
        y1 = left_lines[i+1]["y"] if i+1 < len(left_lines) else float("inf")
        # avancer
        while ridx < len(right_lines) and right_lines[ridx]["y"] < y0 - y_pad:
            ridx += 1
        bucket = []
        j = ridx
        while j < len(right_lines) and right_lines[j]["y"] < y1 - y_pad:
            bucket.append(right_lines[j]["text"])
            j += 1
        if not bucket:
            continue

        segments = _split_value_segments(bucket)
        is_modalites = bool(re.search(r"(?i)modalit[éè]s?.*remboursement", lab["text"]))
        for seg in segments:
            if is_modalites and " et " in seg:
                for part in [p.strip() for p in seg.split(" et ") if p.strip()]:
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
        tuples = []
        for page in doc:
            tuples.extend(_two_column_pairs_char_coverage(page))

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

