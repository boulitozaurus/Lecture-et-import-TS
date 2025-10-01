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
BULLET_RE   = re.compile(r"^[•●▪·\-–]+[\s]*")
ENUM_RE     = re.compile(r"^(?:\d+[\.\)]|[a-z]\))\s+")
PAIR_SEP_RE = re.compile(r"^(.{2,160}?)\s*[:\-–]\s*(.+)$")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _drawings_lines(page: fitz.Page, min_len=8.0, tol=0.8):
    """Retourne (vlines, hlines) listant les segments presque verticaux / horizontaux."""
    v, h = [], []
    for d in page.get_drawings():
        for p in d.get("items", []):
            if p[0] != "l":   # on ne garde que les segments (“line”)
                continue
            (_, x0, y0, x1, y1, _, _) = p
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            length = (dx*dx + dy*dy) ** 0.5
            if length < min_len:
                continue
            if dx <= tol and dy > dx:   # quasi vertical
                v.append((x0, y0, x1, y1, length))
            elif dy <= tol and dx > dy: # quasi horizontal
                h.append((x0, y0, x1, y1, length))
    return v, h

def _pick_col_split(vlines, page_rect):
    """Choisit la plus longue verticale “centrale” comme césure colonnes."""
    if not vlines:
        return None
    page_mid = (page_rect.x0 + page_rect.x1)/2.0
    # garder les lignes qui couvrent une grande partie de la page
    H = page_rect.y1 - page_rect.y0
    cand = []
    for (x0,y0,x1,y1,L) in vlines:
        cov = abs(y1 - y0) / H
        xm  = (x0 + x1)/2.0
        cand.append((cov, -abs(xm - page_mid), xm))  # max cov, min distance au centre
    cov, _, xm = max(cand)
    if cov < 0.35:   # si la séparation verticale ne couvre pas assez → trop risqué
        return None
    return xm

def _bands_from_hlines(hlines, page_rect, min_height=8.0):
    """Convertit les horizontales en bandes (rangées)."""
    ys = []
    for (_x0,y0,_x1,y1,_L) in hlines:
        ys.append(y0); ys.append(y1)
    ys = sorted(ys)
    # dédoublonner proche
    merged = []
    for y in ys:
        if not merged or abs(y - merged[-1]) > 2.0:
            merged.append(y)
    if len(merged) < 2:
        return []
    bands = []
    for a, b in zip(merged, merged[1:]):
        if (b - a) >= min_height:
            bands.append((a, b))
    # si aucune bande exploitable: une seule "méga-bande" page entière
    if not bands:
        bands = [(page_rect.y0, page_rect.y1)]
    return bands

def _page_lines_with_bbox(page: fitz.Page):
    """Récupère toutes les lignes (texte) avec leur bbox."""
    d = page.get_text("dict")
    out = []
    for blk in d.get("blocks", []):
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            x0 = min(sp["bbox"][0] for sp in spans)
            y0 = min(sp["bbox"][1] for sp in spans)
            x1 = max(sp["bbox"][2] for sp in spans)
            y1 = max(sp["bbox"][3] for sp in spans)
            text = _norm(" ".join(sp.get("text","") for sp in spans))
            if text:
                out.append({"x0":x0, "y0":y0, "x1":x1, "y1":y1, "y":(y0+y1)/2.0, "text":text})
    return sorted(out, key=lambda r: (r["y"], r["x0"]))

def _split_value_segments(lines_texts):
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

def _two_column_pairs_from_grid(page: fitz.Page, y_pad=2.0):
    """
    1) Cherche colonnes & rangées via traits vectoriels.
    2) Collecte label (gauche) + valeurs (droite) par rangée.
    3) Éclate puces et 'X : Y' en tuples séparés.
    """
    rect = page.rect
    vlines, hlines = _drawings_lines(page)
    split_x = _pick_col_split(vlines, rect)
    lines = _page_lines_with_bbox(page)

    # Fallback si pas de césure détectée
    if split_x is None:
        return _two_column_pairs_char_coverage(page)  # ← ta version char-coverage déjà en place

    bands = _bands_from_hlines(hlines, rect)
    if not bands:
        bands = [(rect.y0, rect.y1)]

    tuples = []
    for (y0, y1) in bands:
        # lignes qui tombent dans la bande
        band_lines = [ln for ln in lines if (y0 - y_pad) <= ln["y"] < (y1 - y_pad)]
        if not band_lines:
            continue
        left  = [ln for ln in band_lines if ln["x1"] <= split_x - 2.0]   # totalement à gauche
        right = [ln for ln in band_lines if ln["x0"] >= split_x + 2.0]   # totalement à droite

        # sécu: les lignes avec chevauchement (x0 < split < x1) → on les pousse côté droit
        overlap = [ln for ln in band_lines if (ln not in left and ln not in right)]
        right.extend(overlap)

        if not left:
            continue  # pas d’étiquette → on ignore la rangée

        # fusion des labels gauche (multilignes) dans la bande
        left = sorted(left, key=lambda r: r["y"])
        label_text = _norm(" ".join(ln["text"] for ln in left))

        if not right:
            continue  # pas de valeur

        # tri des valeurs par y puis éclatement
        right = sorted(right, key=lambda r: (r["y"], r["x0"]))
        segments = _split_value_segments([r["text"] for r in right])

        # Cas "Modalités de remboursement" : séparer par " et "
        is_modalites = bool(re.search(r"(?i)modalit[éè]s?.*remboursement", label_text))
        for seg in segments:
            if is_modalites and " et " in seg:
                for part in [p.strip() for p in seg.split(" et ") if p.strip()]:
                    tuples.append({"page": str(page.number + 1), "label": label_text, "value": part})
            else:
                tuples.append({"page": str(page.number + 1), "label": label_text, "value": seg})

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
            tuples.extend(_two_column_pairs_from_grid(page))

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

