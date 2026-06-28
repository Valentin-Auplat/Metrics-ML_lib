# Template Diff-in-Differences (Python)

Template fonctionnel d'analyse en Difference-in-Differences (DiD), avec
génération automatique d'un rapport PDF explicatif.

## Contenu

- `did_template.py` — script principal : génération/chargement des données,
  toutes les variantes de DiD, tous les tests statistiques, tous les
  graphiques.
- `generate_pdf_report.py` — génère le rapport PDF (`outputs/rapport_diff_in_diff.pdf`)
  à partir des résultats produits par `did_template.py`.
- `data/panel_data.csv` — base de données (générée automatiquement au premier
  lancement si absente ; remplaçable par vos propres données, cf. ci-dessous).
- `outputs/` — figures (`outputs/figures/*.png`), récapitulatif chiffré
  (`outputs/resultats_did.csv`) et rapport PDF.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python did_template.py
```

Ceci exécute l'ensemble du pipeline (chargement des données → DiD 2x2 → TWFE
statique → event-study → décomposition de Goodman-Bacon → estimateur de
Callaway & Sant'Anna → tests statistiques → export des résultats) puis génère
automatiquement le rapport PDF.

## Variantes de DiD implémentées

1. **DiD classique 2x2** (un groupe traité, un groupe de contrôle, deux
   périodes).
2. **DiD multi-périodes — TWFE statique** (panel à effets fixes individu +
   temps).
3. **DiD dynamique — event-study** (leads & lags autour du traitement, test
   de pré-tendance).
4. **DiD à adoption échelonnée** :
   - décomposition de Goodman-Bacon (diagnostic du biais du TWFE) ;
   - estimateur de Callaway & Sant'Anna (ATT(g,t) robuste à l'hétérogénéité
     dynamique des effets).

## Tests statistiques implémentés

- Test conjoint de tendances parallèles (test de Wald sur les leads).
- Test de placebo (fausse date de traitement).
- Test de permutation (inférence par randomisation).
- Test de Breusch-Pagan (hétéroscédasticité).
- Bootstrap par cluster (unité) des erreurs-types.

## Utiliser vos propres données

Remplacez la fonction `load_data()` (ou directement le fichier
`data/panel_data.csv`) par votre base, avec le schéma de colonnes suivant :

| Colonne       | Type  | Description                                                |
|---------------|-------|--------------------------------------------------------------|
| `unit_id`     | int/str | identifiant unique de l'unité                              |
| `time`        | int   | période (compteur entier régulier, 1..T)                    |
| `Y`           | float | variable de résultat (outcome)                               |
| `group`       | str   | libellé de la cohorte de traitement (ou "Jamais_traite")     |
| `first_treat` | float | première période de traitement (NaN si jamais traité)        |
| `treated`     | int   | indicatrice 0/1, =1 si traité à la période `time`             |

Si votre base ne comporte qu'une seule date de traitement (pas d'adoption
échelonnée), seules les sections 1 à 3 (DiD 2x2, TWFE, event-study) sont
pertinentes.

## Limites et avertissement

La décomposition de Goodman-Bacon implémentée ici suit la logique du théorème
original (distinction comparaisons "propres" vs "à risque", pondération par
variance du traitement résiduelle des effets fixes) mais n'est pas une
réplication bit-à-bit de l'algorithme publié. Pour une réplication exacte,
utiliser le package R `bacondecomp`. Le rapport PDF généré détaille
l'ensemble des hypothèses, limites et références bibliographiques.
