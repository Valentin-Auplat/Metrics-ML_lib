#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
TEMPLATE — ANALYSE PAR CONTRÔLE SYNTHÉTIQUE (Synthetic Control Method, SCM)
==============================================================================

Implémentation de la méthode de contrôle synthétique (Abadie & Gardeazabal,
2003 ; Abadie, Diamond & Hainmueller, 2010, 2015), incluant :

  1. Construction du contrôle synthétique (optimisation des poids W et,
     en option, de la matrice de pondération des prédicteurs V).
  2. Graphiques : trajectoires traité vs synthétique, écart (gap),
     poids des unités donneuses, placebos spatiaux, placebo temporel,
     ratios RMSPE post/pré.
  3. Tests statistiques : qualité d'ajustement pré-traitement, test de
     permutation (inférence à la Fisher) basé sur les placebos spatiaux,
     test placebo temporel, et analyse de sensibilité leave-one-out.

CE SCRIPT EST UN TEMPLATE : la section CONFIG ci-dessous est l'endroit
où vous adaptez les noms de colonnes, le chemin du fichier et les
paramètres de l'analyse à VOTRE jeu de données existant.

Format de données attendu (panel long) :

    unit_col   | time_col | outcome_col | predicteur_1 | predicteur_2 | ...
    ---------------------------------------------------------------------
    "Unité A"  |  2001    |   12.3      |    ...       |     ...
    "Unité A"  |  2002    |   12.7      |    ...       |     ...
    "Unité B"  |  2001    |    9.8      |    ...       |     ...
    ...

Une unité est "traitée" à partir d'une période donnée ; les autres
unités constituent le pool de donneurs (réservoir de contrôle).

MODE « PRÊT À L'EMPLOI » : si le fichier indiqué par `data_path` n'existe
pas, le script génère automatiquement un jeu de données panel SIMULÉ
mais réaliste (une « région traitée » et 14 régions de contrôle, avec un
effet causal négatif imposé à partir de la date de traitement) et
l'enregistre à cet emplacement avant de lancer l'analyse. Cela permet de
faire tourner le script et d'observer tous les graphiques et tests
immédiatement, sans disposer encore de vos propres données. Quand vous
aurez votre fichier réel, placez-le simplement au même chemin (ou changez
`data_path`) : le script utilisera alors vos données et ignorera la
génération automatique.

Dépendances : numpy, pandas, scipy, matplotlib
==============================================================================
"""

from __future__ import annotations

import os
import json
import warnings
import dataclasses
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 0. JEU DE DONNÉES DE DÉMONSTRATION (générée seulement si aucun fichier
#    n'existe à `data_path` — voir section CONFIG et bloc d'exécution en bas
#    de fichier). À SUPPRIMER / IGNORER dès que vous fournissez vos propres
#    données réelles.
# ==============================================================================

def generate_demo_dataset(path: str, seed: int = 42) -> pd.DataFrame:
    """
    Construit un panel simulé mais réaliste pour faire fonctionner le
    template de bout en bout sans données réelles :

      - 1 région "traitée" (Région_Alpha) + 14 régions de contrôle,
      - 1996-2022 (27 périodes),
      - une variable d'intérêt ("pib_par_habitant") qui suit une tendance
        commune + un facteur latent partagé + bruit idiosyncratique,
      - deux prédicteurs auxiliaires ("investissement", "taux_emploi")
        corrélés à l'outcome, utilisés pour l'appariement pré-traitement,
      - un effet causal négatif imposé sur la Région_Alpha à partir de
        2013 (ex. choc économique régional), de magnitude croissante,
      - une hétérogénéité réaliste entre régions de contrôle (certaines
        nettement plus proches de la région traitée que d'autres, pour
        produire une composition de poids non triviale).

    Le fichier est écrit au format CSV à l'emplacement `path`.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1996, 2023))
    n_years = len(years)
    treatment_year = 2013

    treated_name = "Région_Alpha"
    donor_names = [f"Région_{c}" for c in
                   ["Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
                    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron"]]
    units = [treated_name] + donor_names

    # Facteur commun (cycle macroéconomique partagé par toutes les régions)
    common_factor = np.cumsum(rng.normal(0.25, 0.22, n_years))

    rows = []
    for u in units:
        # Chaque région a un niveau de base, une tendance et une sensibilité
        # au facteur commun différents -> hétérogénéité réaliste du pool.
        base_level = rng.normal(22.0, 4.0)
        own_trend = rng.normal(0.35, 0.12)
        loading = rng.uniform(0.5, 1.3)        # sensibilité au facteur commun
        noise = rng.normal(0, 0.4, n_years)

        investissement = (base_level * 0.4 + loading * common_factor * 0.3
                           + rng.normal(0, 0.8, n_years).cumsum() * 0.05 + 8.0)
        taux_emploi = (60 + 0.5 * own_trend * np.arange(n_years)
                        + 0.4 * loading * common_factor + rng.normal(0, 1.0, n_years))

        outcome = (base_level + own_trend * np.arange(n_years)
                   + loading * common_factor + 0.15 * investissement + noise)

        if u == treated_name:
            idx0 = years.index(treatment_year)
            # effet causal négatif, croissant en magnitude (ex. choc régional)
            effect = -np.concatenate([
                np.zeros(idx0),
                1.1 * (np.arange(n_years - idx0) + 1) ** 0.8
            ])
            outcome = outcome + effect

        for t, year in enumerate(years):
            rows.append({
                "region": u,
                "annee": year,
                "pib_par_habitant": round(float(outcome[t]), 3),
                "investissement": round(float(investissement[t]), 3),
                "taux_emploi": round(float(taux_emploi[t]), 3),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df

# ==============================================================================
# 1. CONFIG — À ADAPTER À VOTRE JEU DE DONNÉES
# ==============================================================================

@dataclass
class SCMConfig:
    # --- Chemin vers votre fichier de données existant ---------------------
    # Si ce fichier n'existe pas, un jeu de données de démonstration est
    # généré automatiquement à cet emplacement (voir generate_demo_dataset
    # et le bloc d'exécution en bas de fichier) afin que le script tourne
    # immédiatement. Remplacez simplement ce chemin par celui de VOS
    # données réelles quand vous les aurez : <-- ADAPTER
    data_path: str = "data.csv"
    sep: str = ","

    # --- Structure du panel --------------------------------------------------
    unit_col: str = "unit"               # <-- ADAPTER : colonne identifiant l'unité
    time_col: str = "year"               # <-- ADAPTER : colonne temporelle
    outcome_col: str = "outcome"         # <-- ADAPTER : variable d'intérêt (Y)

    # --- Définition du traitement --------------------------------------------
    treated_unit: str = "TreatedUnit"    # <-- ADAPTER : nom de l'unité traitée
    treatment_period: int = 2010         # <-- ADAPTER : 1ère période post-traitement
    donor_pool: Sequence[str] | None = None
    # Si None, donor_pool = toutes les unités != treated_unit

    # --- Prédicteurs utilisés pour l'appariement pré-traitement --------------
    # Inclure typiquement l'outcome retardé à quelques dates clés + covariables.
    predictors: Sequence[str] = field(default_factory=lambda: [])
    # <-- ADAPTER : ex. ["gdp_per_capita", "investment_rate", "outcome_lag1995"]
    # Si vide, seules les moyennes pré-traitement de l'outcome seront utilisées.

    # --- Fenêtre temporelle ----------------------------------------------------
    pre_periods: Sequence[int] | None = None   # période(s) pré-traitement pour le fit
    # Si None, déduit automatiquement = toutes les périodes < treatment_period

    # --- Options d'optimisation -------------------------------------------------
    nested_optimization: bool = True   # optimise aussi V (poids des prédicteurs)
    n_v_restarts: int = 6              # redémarrages multiples pour éviter optima locaux

    # --- Sorties -----------------------------------------------------------------
    output_dir: str = "outputs"
    random_seed: int = 42


# ==============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================

def load_panel(cfg: SCMConfig) -> pd.DataFrame:
    """Charge le panel et fait des vérifications de cohérence minimales."""
    df = pd.read_csv(cfg.data_path, sep=cfg.sep)

    required = {cfg.unit_col, cfg.time_col, cfg.outcome_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

    if cfg.treated_unit not in df[cfg.unit_col].unique():
        raise ValueError(
            f"L'unité traitée '{cfg.treated_unit}' n'apparaît pas dans la "
            f"colonne '{cfg.unit_col}'."
        )

    df = df.sort_values([cfg.unit_col, cfg.time_col]).reset_index(drop=True)

    # Vérification du panel équilibré (même périodes pour chaque unité)
    counts = df.groupby(cfg.unit_col)[cfg.time_col].nunique()
    if counts.nunique() > 1:
        warnings.warn(
            "Le panel n'est pas équilibré (nombre de périodes différent selon "
            "les unités). Le contrôle synthétique nécessite normalement un "
            "panel équilibré sur la fenêtre d'analyse : vérifiez vos données."
        )
    return df


def get_donor_pool(df: pd.DataFrame, cfg: SCMConfig) -> list[str]:
    if cfg.donor_pool is not None:
        return list(cfg.donor_pool)
    return sorted(u for u in df[cfg.unit_col].unique() if u != cfg.treated_unit)


def get_pre_periods(df: pd.DataFrame, cfg: SCMConfig) -> list:
    if cfg.pre_periods is not None:
        return sorted(cfg.pre_periods)
    periods = sorted(df[cfg.time_col].unique())
    return [p for p in periods if p < cfg.treatment_period]


# ==============================================================================
# 3. CONSTRUCTION DES MATRICES DE PRÉDICTEURS (X) ET DE TRAJECTOIRES (Z)
# ==============================================================================

def build_predictor_matrices(df: pd.DataFrame, cfg: SCMConfig,
                              donors: list[str], pre_periods: list):
    """
    Construit :
      X1 : vecteur (k,) des prédicteurs pré-traitement pour l'unité traitée
      X0 : matrice (k, J) des prédicteurs pré-traitement pour les J donneurs
      Z1 : vecteur (T0,) trajectoire pré-traitement de Y pour le traité
      Z0 : matrice (T0, J) trajectoires pré-traitement de Y pour les donneurs
    """
    pre = df[df[cfg.time_col].isin(pre_periods)]

    # --- Trajectoires pré-traitement de l'outcome (toujours utilisées dans Z) --
    pivot_y = pre.pivot_table(index=cfg.time_col, columns=cfg.unit_col,
                               values=cfg.outcome_col)
    pivot_y = pivot_y.reindex(pre_periods)

    Z1 = pivot_y[cfg.treated_unit].values.astype(float)
    Z0 = pivot_y[donors].values.astype(float)

    # --- Prédicteurs : moyennes pré-traitement des variables désignées --------
    predictor_cols = list(cfg.predictors)
    feature_rows = []
    if predictor_cols:
        means = pre.groupby(cfg.unit_col)[predictor_cols].mean()
        for u in [cfg.treated_unit] + donors:
            feature_rows.append(means.loc[u].values.astype(float))
        X_all = np.vstack(feature_rows)
    else:
        # Si aucun prédicteur n'est fourni, on utilise la moyenne pré-traitement
        # de l'outcome lui-même comme unique prédicteur (cas minimal).
        means_y = pre.groupby(cfg.unit_col)[cfg.outcome_col].mean()
        X_all = np.array([[means_y[u]] for u in [cfg.treated_unit] + donors])
        predictor_cols = [f"mean_{cfg.outcome_col}_pre"]

    X1 = X_all[0, :]
    X0 = X_all[1:, :].T  # (k, J)

    return X1, X0, Z1, Z0, predictor_cols


def standardize_predictors(X1: np.ndarray, X0: np.ndarray):
    """Standardise chaque prédicteur (ligne) par son écart-type empirique,
    comme recommandé par Abadie et al. pour rendre les prédicteurs comparables."""
    all_vals = np.column_stack([X1.reshape(-1, 1), X0])
    sd = all_vals.std(axis=1, ddof=1)
    sd[sd == 0] = 1.0
    return X1 / sd, (X0.T / sd).T, sd


# ==============================================================================
# 4. OPTIMISATION DES POIDS — PROBLÈME DU CONTRÔLE SYNTHÉTIQUE
# ==============================================================================

def solve_W(X0: np.ndarray, X1: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Résout : min_W (X1 - X0 W)' V (X1 - X0 W)
    sous contraintes  W_j >= 0,  somme(W) = 1.

    C'est un problème quadratique convexe sous contraintes ; on utilise SLSQP
    (alternative possible : un solveur QP dédié type `quadprog` ou `cvxpy`).
    """
    k, J = X0.shape

    def objective(W):
        diff = X1 - X0 @ W
        return float(diff @ V @ diff)

    def grad(W):
        diff = X1 - X0 @ W
        return -2.0 * (X0.T @ (V @ diff))

    w0 = np.repeat(1.0 / J, J)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                     "jac": lambda w: np.ones_like(w)}]
    bounds = Bounds(lb=np.zeros(J), ub=np.ones(J))

    res = minimize(objective, w0, jac=grad, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-12})
    W = np.clip(res.x, 0, None)
    W = W / W.sum()
    return W


def solve_V_and_W(X0: np.ndarray, X1: np.ndarray, Z0: np.ndarray, Z1: np.ndarray,
                   n_restarts: int = 6, seed: int = 42):
    """
    Optimisation imbriquée (nested optimization) :
    on choisit V (diagonale, poids des prédicteurs, somme = 1) afin de
    minimiser l'erreur quadratique moyenne de prédiction PRÉ-traitement
    de l'outcome (RMSPE pré), W(V) étant lui-même la solution optimale
    de l'étape interne pour un V donné.
    """
    k = X0.shape[0]
    rng = np.random.default_rng(seed)

    def inner_loss(v_raw):
        v = np.abs(v_raw)
        v = v / v.sum() if v.sum() > 0 else np.repeat(1.0 / k, k)
        V = np.diag(v)
        W = solve_W(X0, X1, V)
        pred = Z0 @ W
        return float(np.mean((Z1 - pred) ** 2)), W, v

    best = None
    starts = [np.repeat(1.0 / k, k)] + [rng.dirichlet(np.ones(k)) for _ in range(n_restarts - 1)]
    for v0 in starts:
        # optimisation simplexe approchée par descente sans contrainte sur des
        # paramètres positifs renormalisés (reparamétrisation simple et robuste)
        def obj(v_raw):
            loss, _, _ = inner_loss(v_raw)
            return loss

        res = minimize(obj, v0, method="Nelder-Mead",
                        options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-10})
        loss, W, v = inner_loss(res.x)
        if best is None or loss < best[0]:
            best = (loss, W, v)

    loss, W, v = best
    V = np.diag(v)
    return W, V


# ==============================================================================
# 5. MÉTRIQUES D'AJUSTEMENT (RMSPE) ET EFFET ESTIMÉ
# ==============================================================================

def rmspe(actual: np.ndarray, synthetic: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - synthetic) ** 2)))


def fit_unit_path(df: pd.DataFrame, cfg: SCMConfig, unit: str) -> pd.Series:
    sub = df[df[cfg.unit_col] == unit].sort_values(cfg.time_col)
    return sub.set_index(cfg.time_col)[cfg.outcome_col]


# ==============================================================================
# 6. PLACEBOS — INFÉRENCE « À LA FISHER »
# ==============================================================================

def run_full_scm_for_unit(df: pd.DataFrame, cfg: SCMConfig, target_unit: str,
                           all_units: list[str]):
    """
    Reproduit toute la procédure de contrôle synthétique en traitant
    `target_unit` comme « unité traitée » (utilisé pour les placebos
    spatiaux : chaque unité du pool de donneurs est tour à tour traitée
    comme si elle avait reçu le traitement).
    """
    donors = [u for u in all_units if u != target_unit]
    pre_periods = get_pre_periods(df, cfg)
    all_periods = sorted(df[cfg.time_col].unique())
    post_periods = [p for p in all_periods if p >= cfg.treatment_period]

    cfg_local = dataclasses.replace(cfg, treated_unit=target_unit, donor_pool=donors)
    X1, X0, Z1, Z0, _ = build_predictor_matrices(df, cfg_local, donors, pre_periods)
    X1s, X0s, _ = standardize_predictors(X1, X0)

    if cfg.nested_optimization:
        W, V = solve_V_and_W(X0s, X1s, Z0, Z1, n_restarts=cfg.n_v_restarts,
                              seed=cfg.random_seed)
    else:
        k = X0s.shape[0]
        V = np.eye(k) / k
        W = solve_W(X0s, X1s, V)

    full_pivot = df[df[cfg.unit_col].isin([target_unit] + donors)].pivot_table(
        index=cfg.time_col, columns=cfg.unit_col, values=cfg.outcome_col
    ).reindex(all_periods)

    actual = full_pivot[target_unit].values.astype(float)
    synthetic = full_pivot[donors].values.astype(float) @ W

    pre_mask = np.isin(all_periods, pre_periods)
    post_mask = np.isin(all_periods, post_periods)

    rmspe_pre = rmspe(actual[pre_mask], synthetic[pre_mask])
    rmspe_post = rmspe(actual[post_mask], synthetic[post_mask])

    return {
        "unit": target_unit,
        "periods": all_periods,
        "actual": actual,
        "synthetic": synthetic,
        "gap": actual - synthetic,
        "weights": dict(zip(donors, W)),
        "rmspe_pre": rmspe_pre,
        "rmspe_post": rmspe_post,
        "rmspe_ratio": rmspe_post / rmspe_pre if rmspe_pre > 0 else np.inf,
    }


def spatial_placebo_test(df: pd.DataFrame, cfg: SCMConfig, donors: list[str]):
    """
    Test de permutation à la Abadie, Diamond & Hainmueller (2010, 2015) :
    on applique la MÊME procédure de contrôle synthétique à chaque unité du
    pool de donneurs comme si elle était traitée, puis on compare le ratio
    RMSPE(post)/RMSPE(pré) de l'unité réellement traitée à la distribution de
    ces ratios pour les unités placebo. La p-value exacte est le rang de
    l'unité traitée dans cette distribution.
    """
    all_units = [cfg.treated_unit] + donors
    results = {}
    results[cfg.treated_unit] = run_full_scm_for_unit(df, cfg, cfg.treated_unit, all_units)
    for u in donors:
        try:
            results[u] = run_full_scm_for_unit(df, cfg, u, all_units)
        except Exception as e:
            warnings.warn(f"Placebo échoué pour l'unité '{u}' : {e}")
    return results


def compute_permutation_pvalue(placebo_results: dict, treated_unit: str) -> float:
    ratios = {u: r["rmspe_ratio"] for u, r in placebo_results.items()
              if np.isfinite(r["rmspe_ratio"])}
    sorted_units = sorted(ratios, key=lambda u: ratios[u], reverse=True)
    rank = sorted_units.index(treated_unit) + 1
    p_value = rank / len(sorted_units)
    return p_value, ratios


def temporal_placebo_test(df: pd.DataFrame, cfg: SCMConfig, donors: list[str],
                           fake_treatment_period):
    """
    Placebo temporel : on déplace artificiellement la date de traitement à une
    période antérieure à la vraie date (période où, en théorie, il ne devrait
    pas encore y avoir d'effet). Si le contrôle synthétique détecte un « effet »
    significatif avant le traitement réel, cela affaiblit la crédibilité de
    l'identification.

    IMPORTANT : l'échantillon est restreint aux périodes STRICTEMENT
    ANTÉRIEURES à la véritable date de traitement (`cfg.treatment_period`)
    avant d'appliquer la fausse date. Sans cette restriction, la fenêtre
    « post » du test fictif chevaucherait la véritable période de
    traitement et le test serait contaminé par le vrai effet causal.
    """
    df_pre_true = df[df[cfg.time_col] < cfg.treatment_period].copy()
    if fake_treatment_period >= cfg.treatment_period:
        raise ValueError(
            "fake_treatment_period doit être strictement antérieure à "
            "cfg.treatment_period pour un placebo temporel valide."
        )
    cfg_fake = dataclasses.replace(cfg, treatment_period=fake_treatment_period)
    all_units = [cfg.treated_unit] + donors
    return run_full_scm_for_unit(df_pre_true, cfg_fake, cfg.treated_unit, all_units)


def leave_one_out_test(df: pd.DataFrame, cfg: SCMConfig, donors: list[str],
                        main_weights: dict):
    """
    Analyse de sensibilité leave-one-out (Abadie, Diamond & Hainmueller, 2015) :
    on retire, une à la fois, chaque unité donneuse ayant un poids positif
    significatif et on recalcule le contrôle synthétique, afin de vérifier
    que le résultat n'est pas piloté par une seule unité du pool de donneurs.
    """
    influential = [u for u, w in main_weights.items() if w > 0.05]
    out = {}
    for u in influential:
        reduced_donors = [d for d in donors if d != u]
        try:
            out[u] = run_full_scm_for_unit(df, cfg, cfg.treated_unit,
                                            [cfg.treated_unit] + reduced_donors)
        except Exception as e:
            warnings.warn(f"Leave-one-out échoué en retirant '{u}' : {e}")
    return out


# ==============================================================================
# 7. GRAPHIQUES
# ==============================================================================

def plot_trends(periods, actual, synthetic, treatment_period, outpath, unit_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, actual, marker="o", lw=2, label=f"{unit_label} (observé)")
    ax.plot(periods, synthetic, marker="s", lw=2, ls="--", label="Contrôle synthétique")
    ax.axvline(treatment_period, color="grey", ls=":", lw=1.5)
    ax.text(treatment_period, ax.get_ylim()[1] * 0.97, " Traitement",
             va="top", ha="left", fontsize=9, color="grey")
    ax.set_xlabel("Période")
    ax.set_ylabel("Variable d'intérêt (outcome)")
    ax.set_title("Unité traitée vs. contrôle synthétique")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_gap(periods, gap, treatment_period, outpath):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(periods, gap, marker="o", lw=2, color="black")
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(treatment_period, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Période")
    ax.set_ylabel("Écart (observé − synthétique)")
    ax.set_title("Effet estimé du traitement au fil du temps (gap plot)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_weights(weights: dict, outpath, threshold=0.01):
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    items = [(u, w) for u, w in items if w >= threshold]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(items))))
    units = [u for u, _ in items]
    vals = [w for _, w in items]
    ax.barh(units, vals, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Poids dans le contrôle synthétique")
    ax.set_title("Composition du contrôle synthétique (unités donneuses)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_placebo_gaps(placebo_results: dict, treated_unit: str, treatment_period,
                       outpath, rmspe_pre_cutoff_ratio=2.0):
    """
    Reproduit la figure classique d'Abadie et al. : gaps de toutes les unités
    placebo en gris, gap de l'unité traitée en gras. Les placebos dont le fit
    pré-traitement est trop mauvais (RMSPE pré trop élevé par rapport à
    l'unité traitée) sont exclus pour ne pas polluer visuellement le graphique
    (pratique standard de la littérature).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    treated_rmspe_pre = placebo_results[treated_unit]["rmspe_pre"]

    for u, r in placebo_results.items():
        if u == treated_unit:
            continue
        if treated_rmspe_pre > 0 and r["rmspe_pre"] > rmspe_pre_cutoff_ratio * treated_rmspe_pre:
            continue
        ax.plot(r["periods"], r["gap"], color="grey", alpha=0.4, lw=1)

    ax.plot(placebo_results[treated_unit]["periods"],
            placebo_results[treated_unit]["gap"],
            color="crimson", lw=2.5, label=treated_unit)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(treatment_period, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Période")
    ax.set_ylabel("Écart (observé − synthétique)")
    ax.set_title("Placebos spatiaux : gaps de toutes les unités du pool")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_rmspe_ratios(ratios: dict, treated_unit: str, outpath):
    items = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
    units = [u for u, _ in items]
    vals = [v for _, v in items]
    colors = ["crimson" if u == treated_unit else "steelblue" for u in units]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(units))))
    ax.barh(units, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Ratio RMSPE post-traitement / pré-traitement")
    ax.set_title("Distribution des ratios RMSPE (test de permutation)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_leave_one_out(main_result: dict, loo_results: dict, treatment_period, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(main_result["periods"], main_result["actual"], color="black", lw=2,
            label=f"{main_result['unit']} (observé)")
    ax.plot(main_result["periods"], main_result["synthetic"], color="crimson",
            lw=2, ls="--", label="Synthétique (poids complets)")
    for u, r in loo_results.items():
        ax.plot(r["periods"], r["synthetic"], color="grey", lw=1, alpha=0.6,
                label=f"Synthétique sans {u}")
    ax.axvline(treatment_period, color="grey", ls=":", lw=1.5)
    ax.set_xlabel("Période")
    ax.set_ylabel("Variable d'intérêt (outcome)")
    ax.set_title("Sensibilité leave-one-out du contrôle synthétique")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


# ==============================================================================
# 8. PIPELINE PRINCIPAL
# ==============================================================================

def run_analysis(cfg: SCMConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = load_panel(cfg)
    donors = get_donor_pool(df, cfg)
    pre_periods = get_pre_periods(df, cfg)
    all_periods = sorted(df[cfg.time_col].unique())

    print(f"[INFO] Unité traitée       : {cfg.treated_unit}")
    print(f"[INFO] Pool de donneurs    : {len(donors)} unités")
    print(f"[INFO] Périodes pré-trait. : {pre_periods}")
    print(f"[INFO] Date de traitement  : {cfg.treatment_period}")

    # --- 1) Construction du contrôle synthétique principal -------------------
    main_result = run_full_scm_for_unit(df, cfg, cfg.treated_unit,
                                         [cfg.treated_unit] + donors)

    print(f"[INFO] RMSPE pré-traitement  : {main_result['rmspe_pre']:.4f}")
    print(f"[INFO] RMSPE post-traitement : {main_result['rmspe_post']:.4f}")
    print(f"[INFO] Ratio RMSPE post/pré  : {main_result['rmspe_ratio']:.3f}")

    plot_trends(main_result["periods"], main_result["actual"], main_result["synthetic"],
                cfg.treatment_period, os.path.join(cfg.output_dir, "fig1_trends.png"),
                cfg.treated_unit)
    plot_gap(main_result["periods"], main_result["gap"], cfg.treatment_period,
             os.path.join(cfg.output_dir, "fig2_gap.png"))
    plot_weights(main_result["weights"], os.path.join(cfg.output_dir, "fig3_weights.png"))

    # --- 2) Placebos spatiaux + test de permutation ---------------------------
    placebo_results = spatial_placebo_test(df, cfg, donors)
    p_value, ratios = compute_permutation_pvalue(placebo_results, cfg.treated_unit)
    print(f"[INFO] p-value (permutation, placebos spatiaux) : {p_value:.4f}")

    plot_placebo_gaps(placebo_results, cfg.treated_unit, cfg.treatment_period,
                       os.path.join(cfg.output_dir, "fig4_placebo_gaps.png"))
    plot_rmspe_ratios(ratios, cfg.treated_unit,
                       os.path.join(cfg.output_dir, "fig5_rmspe_ratios.png"))

    # --- 3) Placebo temporel ----------------------------------------------------
    temporal_result = None
    mid_pre = pre_periods[len(pre_periods) // 2] if len(pre_periods) >= 4 else None
    if mid_pre is not None:
        temporal_result = temporal_placebo_test(df, cfg, donors, mid_pre)
        print(f"[INFO] Placebo temporel (fausse date={mid_pre}) — "
              f"ratio RMSPE : {temporal_result['rmspe_ratio']:.3f}")

    # --- 4) Leave-one-out --------------------------------------------------------
    loo_results = leave_one_out_test(df, cfg, donors, main_result["weights"])
    if loo_results:
        plot_leave_one_out(main_result, loo_results, cfg.treatment_period,
                            os.path.join(cfg.output_dir, "fig6_leave_one_out.png"))

    # --- 5) Export des résultats numériques --------------------------------------
    summary = {
        "treated_unit": cfg.treated_unit,
        "treatment_period": cfg.treatment_period,
        "donor_pool_size": len(donors),
        "weights": {u: round(w, 4) for u, w in main_result["weights"].items() if w > 1e-3},
        "rmspe_pre": main_result["rmspe_pre"],
        "rmspe_post": main_result["rmspe_post"],
        "rmspe_ratio": main_result["rmspe_ratio"],
        "permutation_p_value": p_value,
        "rmspe_ratios_all_units": ratios,
        "temporal_placebo_ratio": temporal_result["rmspe_ratio"] if temporal_result else None,
        "leave_one_out_units_tested": list(loo_results.keys()),
        "average_treatment_effect_post": float(
            np.mean(main_result["gap"][np.isin(main_result["periods"],
                                                [p for p in all_periods if p >= cfg.treatment_period])])
        ),
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Résultats et graphiques écrits dans : {cfg.output_dir}/")
    return summary


# ==============================================================================
# 9. EXEMPLE D'UTILISATION
# ==============================================================================
#
# Tel que livré, ce bloc pointe vers un jeu de données de démonstration
# ("Région_Alpha" affectée par un choc régional simulé à partir de 2013,
# comparée à 14 régions de contrôle). S'il n'existe pas encore, il est
# généré automatiquement (cf. generate_demo_dataset). Vous pouvez donc
# exécuter ce script directement, sans rien fournir, pour voir l'ensemble
# de la démarche, des graphiques et des tests en action.
#
# QUAND VOUS AUREZ VOTRE PROPRE JEU DE DONNÉES :
#   1. Remplacez `data_path` par le chemin de votre fichier réel.
#   2. Adaptez `unit_col`, `time_col`, `outcome_col`, `treated_unit`,
#      `treatment_period` et `predictors` aux noms de colonnes et à la
#      situation de VOTRE analyse.
#   3. Supprimez (ou ignorez) la génération automatique : elle ne se
#      déclenche que si le fichier indiqué n'existe pas.

if __name__ == "__main__":
    cfg = SCMConfig(
        data_path="data_demo.csv",
        unit_col="region",                # <-- ADAPTER
        time_col="annee",                 # <-- ADAPTER
        outcome_col="pib_par_habitant",   # <-- ADAPTER
        treated_unit="Région_Alpha",       # <-- ADAPTER
        treatment_period=2013,             # <-- ADAPTER
        predictors=["investissement", "taux_emploi"],  # <-- ADAPTER
        output_dir="outputs",
    )

    if not os.path.exists(cfg.data_path):
        print(f"[INFO] Aucun fichier trouvé à '{cfg.data_path}' : génération "
              f"d'un jeu de données de démonstration pour illustrer la démarche...")
        generate_demo_dataset(cfg.data_path, seed=cfg.random_seed)
        print(f"[INFO] Jeu de données de démonstration écrit dans '{cfg.data_path}'.\n"
              f"       Remplacez ce fichier (ou changez `data_path` dans CONFIG) "
              f"par vos propres données quand vous les aurez.\n")

    run_analysis(cfg)