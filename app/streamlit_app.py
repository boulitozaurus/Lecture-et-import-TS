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

# ================== ANCRAGE SUR TES LABELS ==================
# 1) Tes labels "toujours vrais" + regex tolérantes (accents / variantes / articles)
CANON_LABELS: dict[str, list[str]] = {
    "Porteur de projet": [r"\bporteur\s+de\s+projet\b"],
    "Objet de prêt obligataire": [r"\bobjet\s+d[eu]\s+pr[ée]t\s+obligataire\b"],
    "Montant total du prêt obligataire": [r"\bmontant\s+total\s+du\s+pr[ée]t\s+obligataire\b"],
    "Taux d'intérêt annuel": [r"\btaux\s+d['’]int[ée]r[êe]t\s+annuel\b"],
    "Durée (mois)": [r"\bdur[ée]e\s*\(\s*mois\s*\)\b"],
    "Modalités de remboursement": [r"\bmodalit[ée]s?\s+de\s+remboursement\b"],
    "Remboursement anticipé sans frais": [r"\bremboursement\s+anticip[ée]\s+sans\s+frais\b"],
    "Sûretés": [r"\bs[uû]ret[ée]s?\b"],
    "Caution": [r"\bcaution\b"],
    "GADP": [r"\bga[dp]p\b"],  # tolère GADP / GAPD
    "Engagements irrévocables et inconditionnels": [r"\bengagements?\s+irrevocables?\s+et\s+inconditionnels?\b"],
    "Frais de structuration": [r"\bfrais\s+de\s+structuration\b"],
    "Exit fees": [r"\bexit\s+fees?\b"],
    "Frais de gestion mensuels": [r"\bfrais\s+de\s+gestion\s+mensuel(?:le)?s?\b"],
    "Prime d'assurance": [r"\bprime\s+d['’]assurance\b"],
    "Validité de l'offre": [r"\bvalidit[ée]\s+de\s+l['’]offre\b"],
    "Reporting": [r"\breporting\b"],
    "Covenants": [r"\bcovenants?\b"],
    "Conditions préalables à la signature d'une convention cadre": [
        r"\bconditions?\s+pr[ée]alables?\s+à\s+la\s+signature\s+d['’]une\s+convention\s+cadre\b"
    ],
    "Conditions préalables à la mise en ligne": [
        r"\bconditions?\s+pr[ée]alables?\s+à\s+la\s+mise\s+en\s+ligne\b"
    ],
    "Conditions suspensives à la libération des fonds": [
        r"\bconditions?\s+suspensives?\s+à\s+la\s+lib[ée]ration\s+des?\s+fonds?\b"
    ],
    "Libération des fonds": [r"\blib[ée]ration\s+des?\s+fonds?\b"],
}

BULLET_RE   = re.compile(r"^[•●▪·\-–]+\s*")
ENUM_RE     = re.compile(r"^(?:\d+[\.\)]|[a-z]\))\s+")
PAIR_SEP_RE = re.compile(r"^(.{2,200}?)\s*[:\-–]\s*(.+)$")
NOTE_RE     = re.compile(r"^note\s+\d+\b", re.I)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# ---- Extraction de lignes + caractères (bbox) ----
def _get_lines_with_chars(page: fitz.Page):
    d = page.get_text("rawdict")
    out = []
    for blk in d.get("blocks", []):
        for li in blk.get("lines", []):
            spans = li.get("spans", [])
            if not spans:
                continue
            x0 = y0 = float("inf")
            x1 = y1 = float("-inf")
            chars = []
            buf = []
            for sp in spans:
                for ch in sp.get("chars", []):
                    cx0, cy0, cx1, cy1 = ch["bbox"]
                    x0 = min(x0, cx0); y0 = min(y0, cy0)
                    x1 = max(x1, cx1); y1 = max(y1, cy1)
                    buf.append(ch.get("c", ""))
                    chars.append((cx0, cx1, cy0, cy1))
            text = _norm("".join(buf))
            if text:
                out.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "y": (y0+y1)/2, "w": x1-x0, "text": text, "chars": chars})
    return sorted(out, key=lambda r: (r["y"], r["x0"]))

# ---- Césure par vallée de densité de caractères ----
def _compute_split_x(lines, bins=96):
    if not lines:
        return None
    xmin = min(l["x0"] for l in lines); xmax = max(l["x1"] for l in lines)
    if xmax <= xmin:
        return None
    step = (xmax - xmin) / bins
    hist = [0.0] * bins
    for l in lines:
        for (cx0, cx1, _, _) in l["chars"]:
            i0 = int(max(0, min(bins-1, (cx0 - xmin) // step)))
            i1 = int(max(0, min(bins-1, (cx1 - xmin) // step)))
            for i in range(i0, i1+1):
                hist[i] += 1.0
    mid = bins // 2
    left_peak  = max(range(0, mid), key=lambda i: hist[i]) if any(hist[:mid]) else 0
    right_peak = max(range(mid, bins), key=lambda i: hist[i]) if any(hist[mid:]) else bins-1
    if right_peak - left_peak >= 4:
        valley = min(range(left_peak+1, right_peak), key=lambda i: hist[i])
        return xmin + valley * step
    # fallback médiane des centres
    centers = [ (l["x0"]+l["x1"]) / 2 for l in lines ]
    return statistics.median(centers)

def _char_coverage_ratio(line, split_x: float):
    left = right = 0.0
    for (cx0, cx1, _, _) in line["chars"]:
        w = max(0.0, cx1 - cx0)
        if ((cx0 + cx1) / 2.0) <= split_x:
            left += w
        else:
            right += w
    tot = left + right
    if tot <= 0:
        return 0.5, 0.5
    return left / tot, right / tot

# ---- Matching d’ancre (label) avec tes regex ----
def _match_canonical_label(text: str) -> str | None:
    t = text.lower()
    for canonical, patterns in CANON_LABELS.items():
        for pat in patterns:
            if re.search(pat, t, flags=re.I):
                return canonical
    return None

def _merge_left_labels_if_no_right_between(left, right, y_pad=2.0):
    merged = []
    i = 0
    while i < len(left):
        y0 = left[i]["y"]; txt = left[i]["text"]
        j = i + 1
        while j < len(left):
            y1 = left[j]["y"]
            if any((y0 - y_pad) <= r["y"] < (y1 - y_pad) for r in right):
                break
            # tenter de fusionner plusieurs lignes pour matcher une ancre longue
            cand = _norm(txt + " " + left[j]["text"])
            if _match_canonical_label(cand):
                txt = cand
                j += 1
            else:
                break
        merged.append({"y": y0, "text": _norm(txt)})
        i = j
    return merged

def _split_value_segments(lines_texts: list[str]) -> list[str]:
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

# ---- Parse principal basé sur TES ancres ----
def _pairs_with_anchors(page: fitz.Page, y_pad=3.0, left_threshold=0.6):
    lines = _get_lines_with_chars(page)
    if not lines:
        return []

    split_x = _compute_split_x(lines)
    if split_x is None:
        return []

    # 1) Classement gauche/droite par couverture + forçages évidents
    left_candidates, right_candidates = [], []
    for ln in lines:
        lg, _rg = _char_coverage_ratio(ln, split_x)
        txt = ln["text"]
        force_right = BULLET_RE.match(txt) or ENUM_RE.match(txt) or NOTE_RE.match(txt)
        if force_right:
            right_candidates.append({"y": ln["y"], "text": txt})
        else:
            (left_candidates if lg >= left_threshold else right_candidates).append({"y": ln["y"], "text": txt})

    left_candidates  = sorted(left_candidates, key=lambda d: d["y"])
    right_candidates = sorted(right_candidates, key=lambda d: d["y"])

    # 2) Ne garder comme labels que ceux qui matchent TES ancres
    left_candidates = _merge_left_labels_if_no_right_between(left_candidates, right_candidates, y_pad=y_pad)
    anchors = []
    for ln in left_candidates:
        canon = _match_canonical_label(ln["text"])
        if canon:
            anchors.append({"y": ln["y"], "label": canon})
    anchors = sorted(anchors, key=lambda a: a["y"])

    if not anchors:
        return []  # aucune ancre reconnue sur cette page

    # 3) Associer chaque ancre aux valeurs droites jusqu’à l’ancre suivante
    tuples = []
    ridx = 0
    for i, anc in enumerate(anchors):
        y0 = anc["y"]
        y1 = anchors[i+1]["y"] if i+1 < len(anchors) else float("inf")

        while ridx < len(right_candidates) and right_candidates[ridx]["y"] < y0 - y_pad:
            ridx += 1
        bucket = []
        j = ridx
        while j < len(right_candidates) and right_candidates[j]["y"] < y1 - y_pad:
            bucket.append(right_candidates[j]["text"])
            j += 1
        if not bucket:
            continue

        segments = _split_value_segments(bucket)

        # Cas "Modalités de remboursement" → découpe par " et "
        if anc["label"] == "Modalités de remboursement":
            final = []
            for seg in segments:
                if " et " in seg:
                    final.extend([p.strip() for p in seg.split(" et ") if p.strip()])
                else:
                    final.append(seg)
            segments = final

        for seg in segments:
            tuples.append({
                "page": str(page.number + 1),
                "label": anc["label"],   # canonique et stable
                "value": seg
            })
    return tuples
# ================== FIN ANCRAGE ==================

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

# ==== Fallback "char-coverage" (MANQUANT) ====
# Classe chaque ligne par couverture de caractères à gauche vs droite d'une césure.
# Utilisé quand _two_column_pairs_from_grid ne trouve pas de césure de tableau.

NOTE_RE     = re.compile(r"^note\s+\d+\b", re.I)  # si absent, réutiliser NOTE_RE déjà défini
# BULLET_RE, ENUM_RE, PAIR_SEP_RE, _norm, _split_value_segments doivent déjà être définis plus haut.

def _get_lines_with_chars(page: fitz.Page):
    """Retourne des lignes avec bbox + caractères (pour calcul de couverture)."""
    d = page.get_text("rawdict")
    out = []
    for blk in d.get("blocks", []):
        for li in blk.get("lines", []):
            spans = li.get("spans", [])
            if not spans:
                continue
            x0 = y0 = float("inf")
            x1 = y1 = float("-inf")
            chars = []
            buf = []
            for sp in spans:
                for ch in sp.get("chars", []):
                    cx0, cy0, cx1, cy1 = ch["bbox"]
                    x0 = min(x0, cx0); y0 = min(y0, cy0)
                    x1 = max(x1, cx1); y1 = max(y1, cy1)
                    buf.append(ch.get("c",""))
                    chars.append((cx0, cx1, cy0, cy1, ch.get("c","")))
            text = _norm("".join(buf))
            if text:
                out.append({"x0":x0, "x1":x1, "y0":y0, "y1":y1, "y":(y0+y1)/2.0, "w":x1-x0, "text":text, "chars":chars})
    return sorted(out, key=lambda r: (r["y"], r["x0"]))

def _compute_split_x_from_char_density(lines, bins=96):
    """Projection horizontale de densité de caractères → coupe à la vallée entre deux pics."""
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
    mid = bins//2
    if any(hist[:mid]) and any(hist[mid:]):
        left_peak  = max(range(0, mid), key=lambda i: hist[i])
        right_peak = max(range(mid, bins), key=lambda i: hist[i])
        if right_peak - left_peak >= 4:
            valley = min(range(left_peak+1, right_peak), key=lambda i: hist[i])
            return float(xmin + valley * step)
    # fallback: médiane des centres
    centers = [ (l["x0"]+l["x1"])/2.0 for l in lines ]
    import statistics
    return float(statistics.median(centers))

def _char_coverage_ratio(line, split_x: float):
    """(ratio_gauche, ratio_droite) en sommant la largeur des caractères de part et d’autre."""
    left = right = 0.0
    for (cx0, cx1, _, _, _) in line["chars"]:
        w = max(0.0, cx1 - cx0)
        if ((cx0 + cx1) / 2.0) <= split_x:
            left += w
        else:
            right += w
    tot = left + right
    if tot <= 0:
        return 0.5, 0.5
    return left/tot, right/tot

def _merge_left_labels_if_no_right_between(left, right, y_pad=2.0):
    """Fusionne les labels multilignes s'il n'y a aucune valeur entre deux lignes gauche."""
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

def _two_column_pairs_char_coverage(page: fitz.Page, y_pad=3.0, left_threshold=0.6):
    """
    Fallback robuste quand pas de grille trouvée.
    Classe chaque ligne par couverture de caractères à gauche/droite de split_x.
    - >60% à gauche → label ; sinon → valeur
    - puces/énum/notes forcées en valeur
    - labels multilignes fusionnés s'il n'y a aucune valeur entre-deux
    """
    lines = _get_lines_with_chars(page)
    if not lines:
        return []

    split_x = _compute_split_x_from_char_density(lines)
    if split_x is None:
        # fallback minimal "label: valeur" sur le texte brut
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
        # forcer "valeur" si évident
        if BULLET_RE.match(txt) or ENUM_RE.match(txt) or NOTE_RE.match(txt):
            right_lines.append({"y": ln["y"], "text": txt})
        else:
            (left_lines if lg >= left_threshold else right_lines).append({"y": ln["y"], "text": txt})

    left_lines  = sorted(left_lines, key=lambda d: d["y"])
    right_lines = sorted(right_lines, key=lambda d: d["y"])

    # fusion labels multilignes
    left_lines = _merge_left_labels_if_no_right_between(left_lines, right_lines, y_pad=y_pad)

    # association label -> valeurs
    tuples = []
    ridx = 0
    for i, lab in enumerate(left_lines):
        y0 = lab["y"]
        y1 = left_lines[i+1]["y"] if i+1 < len(left_lines) else float("inf")
        while ridx < len(right_lines) and right_lines[ridx]["y"] < y0 - y_pad:
            ridx += 1
        bucket, j = [], ridx
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
# ==== fin fallback ====

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
            page_tuples = _pairs_with_anchors(page)
            if not page_tuples:
                page_tuples = _two_column_pairs_from_grid(page)

            tuples.extend(page_tuples)

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

