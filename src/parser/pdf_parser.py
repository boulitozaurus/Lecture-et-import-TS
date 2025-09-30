import re
from typing import Dict, List, Tuple
import pdfplumber
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

def _load_aliases() -> Dict[str, List[str]]:
    with open(CONFIG_DIR / "fields.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

ALIASES = _load_aliases()

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _try_match_key(label: str) -> Tuple[str, float]:
    label_norm = _normalize_space(label)
    best_key, best_score = None, 0.0
    for key, patterns in ALIASES.items():
        for pat in patterns or []:
            m = re.search(pat, label_norm, flags=re.I)
            if m:
                score = len(m.group(0)) / (len(label_norm) + 1)
                if score > best_score:
                    best_key, best_score = key, score
    return best_key, best_score

def _collect_pairs_from_tables(pdf: pdfplumber.PDF) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for page in pdf.pages:
        try:
            tables = page.extract_tables() or []
        except Exception:
            tables = []
        for tb in tables:
            for row in tb:
                if not row or len(row) < 2:
                    continue
                left = _normalize_space(str(row[0] or ""))
                right = _normalize_space(str(row[1] or ""))
                if left and right:
                    pairs.append((left, right))
    return pairs

def _fallback_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        line = _normalize_space(line)
        m = re.match(r"(.{3,80}?)\s*[:\-–]\s*(.+)$", line)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs

def parse_pdf(path: str) -> Dict[str, str]:
    """Retourne un dict 'clé normalisée' -> 'valeur brute' (extraction seule)."""
    out: Dict[str, str] = {}
    with pdfplumber.open(path) as pdf:
        full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])
        pairs = _collect_pairs_from_tables(pdf)
    if not pairs:
        pairs = _fallback_pairs_from_text(full_text)

    for left, right in pairs:
        key, score = _try_match_key(left)
        if key and score >= 0.15:
            if key not in out or not out[key]:
                out[key] = right

    # NE PAS appliquer de règle ici (toutes les règles dans src/rules.py)
    return out
