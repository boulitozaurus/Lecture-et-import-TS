# LookandFin – Term Sheet PDF Parser (Streamlit) – YAML Clauses

Uploader un PDF → extraction automatique des champs (prêt), correction manuelle, export CSV/JSON.
Le **catalogue de clauses suspensives** est désormais géré en **YAML** (pas d'Excel nécessaire).

## Démarrage rapide
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Clauses suspensives (YAML)
Éditez `src/clauses/clauses.yaml`. Deux formats possibles :

### 1) Liste plate
```yaml
- "Expertise indépendante (en l'état, après travaux, liquidative)"
- "Respect des covenants LTV (85% en l'état, 65% après travaux)"
- "Mise en place des sûretés (fiducie-sûreté / hypothèque)"
```

### 2) Par catégories
```yaml
Expertises:
  - "Expertise indépendante (en l'état, après travaux, liquidative)"
Covenants:
  - "Respect des covenants LTV (85% en l'état, 65% après travaux)"
Sûretés:
  - "Mise en place des sûretés (fiducie-sûreté / hypothèque)"
Libérations:
  - "Libération des fonds via notaire"
Décaissements:
  - "Décaissements travaux via compte centralisateur nanti"
  - "Tranches min. 10 000 €; acomptes plafonnés 20%"
```

L'app détecte automatiquement les deux formats et aplatit si besoin.

## Personnalisation
- Alias de libellés PDF → `src/config/fields.yaml`
- Lists de choix (Loan Type / Periodicity) → `src/config/choices.yaml`
