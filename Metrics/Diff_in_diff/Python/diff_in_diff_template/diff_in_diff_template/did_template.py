#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TEMPLATE D'ANALYSE EN DIFFERENCE-IN-DIFFERENCES (DiD)
================================================================================
Ce script est un TEMPLATE FONCTIONNEL conçu pour être appliqué directement à
une base de données panel (individus/entreprises/régions x temps).

Il implémente les principales variantes de DiD couramment utilisées dans la
littérature empirique :

    1. DiD classique 2x2 (un groupe traité, un groupe de contrôle, deux
       périodes - avant/après)
    2. DiD multi-périodes en panel à effets fixes (TWFE statique)
    3. DiD dynamique / "Event-study" (effets pré/post-traitement avec
       leads & lags)
    4. DiD avec adoption échelonnée du traitement ("staggered adoption") :
         - décomposition de Goodman-Bacon (diagnostic du TWFE biaisé)
         - estimateur de Callaway & Sant'Anna (ATT(g,t) robuste à
           l'hétérogénéité dynamique des effets)

ainsi que les tests statistiques usuels associés :
    - test conjoint de tendances parallèles (pré-tendances)
    - test de placebo (fausse date de traitement)
    - test de permutation / inférence par randomisation
    - test d'hétéroscédasticité (Breusch-Pagan)
    - bootstrap par cluster (unité) pour les erreurs-types

et génère :
    - les graphiques pertinents (tendances brutes, event-study, Bacon plot,
      comparaison TWFE vs Callaway & Sant'Anna)
    - un rapport PDF expliquant rigoureusement la méthodologie et les
      résultats (généré par generate_pdf_report.py)

----------------------------------------------------------------------------
COMMENT UTILISER CE TEMPLATE AVEC VOS PROPRES DONNEES
----------------------------------------------------------------------------
Remplacez la fonction `load_data()` par un simple chargement de votre base
existante (CSV, Parquet, SQL...), à condition que le DataFrame final possède
le schéma suivant (mêmes noms de colonnes) :

    unit_id      : identifiant unique de l'unité observée (int/str)
    time         : période (int, 1..T, doit être un compteur entier régulier)
    Y            : variable de résultat (outcome) (float)
    group        : libellé du groupe/de la cohorte de traitement (str),
                   ex. "Jamais traite", "Cohorte_2015", ...
    first_treat  : première période de traitement de l'unité (float, NaN si
                   l'unité n'est jamais traitée)
    treated      : indicatrice 0/1, =1 si l'unité est traitée à la période
                   `time` (post >= first_treat), 0 sinon

Si votre base ne contient qu'un seul groupe traité / une seule date de
traitement, seules les sections 1, 2 et 3 (DiD 2x2, TWFE, event-study) sont
pertinentes : les sections 4 (Bacon, Callaway-Sant'Anna) nécessitent une
adoption échelonnée (plusieurs dates de traitement différentes).
================================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import PanelOLS
from scipy import stats as sstats

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION GENERALE
# ==============================================================================
RNG_SEED = 5
np.random.seed(RNG_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "panel_data.csv")          # base "preexistante"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "resultats_did.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

ALPHA = 0.05                 # seuil de significativite
EVENT_WINDOW = 6             # fenetre d'event-study : [-6, +6], bornee (binnee)
N_PERMUTATIONS = 1000        # nombre de permutations pour le test de randomisation
N_BOOTSTRAP = 500            # nombre de replications bootstrap par cluster
NEVER_TREATED_LABEL = "Jamais_traite"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
PALETTE = {
    "Jamais_traite": "#6b7280",
    "Cohorte_2017": "#2563eb",
    "Cohorte_2020": "#16a34a",
    "Cohorte_2023": "#dc2626",
}


# ==============================================================================
# 1. GENERATION / CHARGEMENT DE LA BASE DE DONNEES
# ==============================================================================
def generate_synthetic_panel(
    n_per_group=150,
    n_periods=24,
    cohorts=None,
    violate_parallel_trends=False,
    sigma_unit_fe=3.0,
    sigma_noise=2.0,
):
    """
    Simule une base de donnees PANEL realiste avec adoption echelonnee du
    traitement (staggered adoption), effets de traitement HETEROGENES et
    DYNAMIQUES (qui croissent avec la duree d'exposition), ce qui est le cas
    de figure typique dans lequel le TWFE statique est biaise et dans lequel
    les estimateurs robustes (Callaway & Sant'Anna) sont preconises.

    Cette fonction simule un fichier qui, en pratique, serait deja disponible
    (export d'une base administrative, d'une enquete panel, etc.). Le CSV
    genere est sauvegarde sur disque pour materialiser une "base de donnees
    preexistante" que le reste du script consomme ensuite normalement.
    """
    if cohorts is None:
        # label -> periode de premier traitement (None = jamais traite)
        cohorts = {
            NEVER_TREATED_LABEL: None,
            "Cohorte_2017": 8,
            "Cohorte_2020": 14,
            "Cohorte_2023": 20,
        }

    # Effet de traitement dynamique : tau(g, exposition) = base_g + croissance_g * (exposition-1)
    # Les cohortes traitees plus tot ont un effet qui croit plus vite : c'est
    # exactement la configuration qui biaise le TWFE statique (cf. Goodman-Bacon).
    effect_params = {
        "Cohorte_2017": {"base": 1.5, "growth": 0.45},
        "Cohorte_2020": {"base": 1.5, "growth": 0.22},
        "Cohorte_2023": {"base": 1.5, "growth": 0.08},
    }

    rows = []
    unit_id = 0

    # Choc macro commun (effet temps), tire UNE SEULE FOIS par periode et
    # partage par toutes les unites (sinon ce n'est plus un "effet temps"
    # commun mais un bruit idiosyncratique supplementaire).
    time_fe = {t: 0.35 * t + np.random.normal(0, 0.4) for t in range(1, n_periods + 1)}

    for label, first_treat in cohorts.items():
        for _ in range(n_per_group):
            unit_id += 1
            alpha_i = np.random.normal(0, sigma_unit_fe)          # effet fixe individuel
            group_level = np.random.normal(0, 0.5)                # heterogeneite de niveau intra-groupe

            # Tendance specifique au groupe (nulle par defaut => tendances paralleles).
            # Activer violate_parallel_trends=True pour simuler une violation de
            # l'hypothese identifiante et voir le test de pre-tendances la detecter.
            group_trend_slope = 0.0
            if violate_parallel_trends and label != NEVER_TREATED_LABEL:
                group_trend_slope = 0.18

            for t in range(1, n_periods + 1):
                lambda_t = time_fe[t]
                trend_violation = group_trend_slope * t

                is_treated = int(first_treat is not None and t >= first_treat)
                tau_it = 0.0
                if is_treated:
                    exposure = t - first_treat + 1
                    p = effect_params[label]
                    tau_it = p["base"] + p["growth"] * (exposure - 1)

                eps = np.random.normal(0, sigma_noise)
                Y = alpha_i + lambda_t + group_level + trend_violation + tau_it * is_treated + eps

                rows.append((
                    unit_id, t, 2000 + t, label,
                    np.nan if first_treat is None else first_treat,
                    is_treated, Y,
                ))

    df = pd.DataFrame(
        rows,
        columns=["unit_id", "time", "year", "group", "first_treat", "treated", "Y"],
    )
    df["rel_time"] = df["time"] - df["first_treat"]  # NaN pour les jamais-traites
    return df


def load_data(force_regenerate=False):
    """
    Charge la base de donnees "preexistante". Si le fichier n'existe pas
    encore (premiere execution), il est genere une seule fois de maniere
    synthetique puis sauvegarde sur disque : a partir de la, le script tourne
    exactement comme si vous aviez fourni votre propre fichier CSV au meme
    format (voir le docstring en tete de fichier pour le schema attendu).
    """
    if force_regenerate or not os.path.exists(DATA_PATH):
        df = generate_synthetic_panel()
        df.to_csv(DATA_PATH, index=False)
        print(f"[load_data] Base synthetique generee et sauvegardee : {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"[load_data] Base de donnees chargee depuis : {DATA_PATH}")

    # Verifications minimales de schema (utiles si vous branchez vos propres donnees)
    required_cols = {"unit_id", "time", "Y", "group", "first_treat", "treated"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la base : {missing}")

    df = df.sort_values(["unit_id", "time"]).reset_index(drop=True)
    return df


# ==============================================================================
# 2. STATISTIQUES DESCRIPTIVES & GRAPHIQUE DES TENDANCES BRUTES
# ==============================================================================
def plot_raw_trends(df, savepath=os.path.join(FIG_DIR, "01_tendances_brutes.png")):
    """
    Trace l'evolution moyenne de Y par groupe/cohorte, avec une ligne
    verticale a la date de traitement de chaque cohorte. C'est le premier
    graphique a toujours produire avant toute estimation : il permet une
    inspection visuelle (informelle) de l'hypothese de tendances paralleles.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    means = df.groupby(["group", "time"])["Y"].mean().reset_index()
    first_treats = df.groupby("group")["first_treat"].first()

    for label, sub in means.groupby("group"):
        ax.plot(sub["time"], sub["Y"], marker="o", markersize=3,
                linewidth=1.8, label=label.replace("_", " "),
                color=PALETTE.get(label, None))
        ft = first_treats[label]
        if pd.notna(ft):
            ax.axvline(ft - 0.5, color=PALETTE.get(label, "grey"),
                       linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Periode")
    ax.set_ylabel("Y moyen")
    ax.set_title("Trajectoires moyennes de l'outcome par cohorte de traitement")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ==============================================================================
# 3. DiD CLASSIQUE 2x2 (un groupe traite, un groupe de controle, 2 periodes)
# ==============================================================================
def did_2x2(df, treat_group, control_group=NEVER_TREATED_LABEL,
            pre_period=None, post_period=None):
    """
    DiD canonique "2x2" : un groupe traite vs un groupe de controle, une
    periode avant vs une periode apres traitement.

    Estimateur :  DiD = (Ybar_T,post - Ybar_T,pre) - (Ybar_C,post - Ybar_C,pre)
    Equivalent a l'OLS :  Y = b0 + b1*treat + b2*post + b3*(treat*post) + eps
    ou b3 est l'estimateur DiD. Les erreurs-types sont clusterisees par unite
    pour tenir compte de l'autocorrelation serielle au sein de chaque unite.
    """
    if pre_period is None:
        pre_period = int(df.loc[df["group"] == treat_group, "first_treat"].iloc[0] - 1)
    if post_period is None:
        post_period = int(df.loc[df["group"] == treat_group, "first_treat"].iloc[0])

    sub = df[(df["group"].isin([treat_group, control_group])) &
             (df["time"].isin([pre_period, post_period]))].copy()
    sub["treat"] = (sub["group"] == treat_group).astype(int)
    sub["post"] = (sub["time"] == post_period).astype(int)

    # --- (a) Calcul "a la main" par moyennes de cellules (verification pedagogique)
    cell_means = sub.groupby(["treat", "post"])["Y"].mean()
    did_manual = ((cell_means[(1, 1)] - cell_means[(1, 0)]) -
                  (cell_means[(0, 1)] - cell_means[(0, 0)]))

    # --- (b) Estimation par regression avec erreurs-types clusterisees par unite
    model = smf.ols("Y ~ treat + post + treat:post", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["unit_id"]}
    )
    coef = model.params["treat:post"]
    se = model.bse["treat:post"]
    ci_low, ci_high = model.conf_int(alpha=ALPHA).loc["treat:post"]
    pval = model.pvalues["treat:post"]

    result = {
        "methode": "DiD 2x2",
        "comparaison": f"{treat_group} vs {control_group}",
        "pre_period": pre_period,
        "post_period": post_period,
        "estimate": coef,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pvalue": pval,
        "n_obs": int(model.nobs),
        "verif_calcul_manuel": did_manual,
        "model": model,
        "cell_means": cell_means,
    }
    return result


def plot_2x2_visual(result, savepath=os.path.join(FIG_DIR, "02_did_2x2.png")):
    """Visualisation canonique du DiD 2x2 : les 2 droites avant/apres et la
    contrefactuelle (parallele a la droite de controle) qui materialise
    l'effet DiD comme l'ecart vertical entre l'observe et le contrefactuel."""
    cm = result["cell_means"]
    pre, post = result["pre_period"], result["post_period"]

    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.plot([pre, post], [cm[(0, 0)], cm[(0, 1)]], "o-", color="#6b7280",
             label="Groupe de controle (observe)")
    ax.plot([pre, post], [cm[(1, 0)], cm[(1, 1)]], "o-", color="#2563eb",
             label="Groupe traite (observe)")
    contrefactuel = cm[(1, 0)] + (cm[(0, 1)] - cm[(0, 0)])
    ax.plot([pre, post], [cm[(1, 0)], contrefactuel], "o--", color="#2563eb",
             alpha=0.5, label="Contrefactuel du groupe traite")
    ax.annotate("", xy=(post, cm[(1, 1)]), xytext=(post, contrefactuel),
                arrowprops=dict(arrowstyle="<->", color="#dc2626"))
    ax.text(post + 0.15, (cm[(1, 1)] + contrefactuel) / 2,
            f"Effet DiD\n= {result['estimate']:.2f}", color="#dc2626", fontsize=9)
    ax.set_xticks([pre, post])
    ax.set_xticklabels(["Avant", "Apres"])
    ax.set_ylabel("Y moyen")
    ax.set_title(f"DiD 2x2 : {result['comparaison']}")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ==============================================================================
# 4. DiD MULTI-PERIODES : TWFE STATIQUE (panel a effets fixes individu + temps)
# ==============================================================================
def did_twfe_static(df):
    """
    Specification "canonique" multi-periodes / multi-groupes :

        Y_it = alpha_i + lambda_t + beta * D_it + eps_it

    avec alpha_i effets fixes individu, lambda_t effets fixes temps, D_it
    l'indicatrice de traitement effectif. beta est l'estimateur DiD "TWFE"
    (Two-Way Fixed Effects). Erreurs-types clusterisees par unite.

    ATTENTION (cf. Goodman-Bacon 2021, de Chaisemartin & D'Haultfoeuille 2020) :
    en presence d'adoption echelonnee ET d'effets de traitement heterogenes
    dans le temps, beta n'estime PAS en general un effet moyen de traitement
    interpretable (ATT) : certaines comparaisons implicites utilisent des
    unites DEJA traitees comme groupe de controle, ce qui biaise l'estimateur.
    C'est precisement ce que diagnostique la decomposition de Goodman-Bacon
    (section 6) et que corrige l'estimateur de Callaway & Sant'Anna (section 7).
    """
    panel = df.set_index(["unit_id", "time"])
    mod = PanelOLS.from_formula("Y ~ 1 + treated + EntityEffects + TimeEffects",
                                 data=panel, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    return {
        "methode": "TWFE statique (multi-periodes)",
        "estimate": res.params["treated"],
        "se": res.std_errors["treated"],
        "ci_low": res.conf_int(level=1 - ALPHA).loc["treated", "lower"],
        "ci_high": res.conf_int(level=1 - ALPHA).loc["treated", "upper"],
        "pvalue": res.pvalues["treated"],
        "n_obs": int(res.nobs),
        "model": res,
    }


# ==============================================================================
# 5. DiD DYNAMIQUE / "EVENT-STUDY" (leads & lags autour du traitement)
# ==============================================================================
def build_event_dummies(df, window=EVENT_WINDOW, ref=-1):
    """
    Construit les indicatrices de temps-relatif-au-traitement (event-time
    dummies), bornees a +/- `window` (les observations au-dela sont regroupees
    dans les categories terminales, pratique standard pour eviter d'estimer
    des coefficients sur des cellules trop petites). La periode `ref`
    (par defaut -1, la periode juste avant le traitement) est omise et sert
    de reference. Les unites jamais traitees ont toutes leurs dummies a 0 et
    servent de groupe de controle.
    """
    out = df.copy()
    rel = out["rel_time"]
    cols = []
    for k in range(-window, window + 1):
        if k == ref:
            continue
        if k < -window:
            continue
        colname = f"lead_{abs(k)}" if k < 0 else f"lag_{k}"
        if k == -window:
            cond = rel <= k
        elif k == window:
            cond = rel >= k
        else:
            cond = rel == k
        out[colname] = cond.fillna(False).astype(int)
        cols.append(colname)
    return out, cols


def did_event_study(df, window=EVENT_WINDOW, ref=-1):
    """
    Regression d'event-study en panel a effets fixes :

        Y_it = alpha_i + lambda_t + sum_{k != ref} delta_k * 1{rel_time_it = k} + eps_it

    Les coefficients delta_k (pour k<0, "leads") permettent de TESTER les
    tendances paralleles avant traitement (ils devraient etre ~0 et non
    significatifs si l'hypothese identifiante est valide) ; les coefficients
    pour k>=0 ("lags") retracent la dynamique de l'effet du traitement.
    """
    out, cols = build_event_dummies(df, window=window, ref=ref)
    panel = out.set_index(["unit_id", "time"])
    formula = "Y ~ 1 + " + " + ".join(cols) + " + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(formula, data=panel, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    rel_times = [-window + i if i < window else i - window
                 for i in range(0)]  # placeholder, recalcule juste apres
    coefs = []
    for k in range(-window, window + 1):
        colname = f"lead_{abs(k)}" if k < 0 else f"lag_{k}"
        if k == ref:
            coefs.append({"rel_time": k, "estimate": 0.0, "se": 0.0,
                          "ci_low": 0.0, "ci_high": 0.0, "pvalue": np.nan})
        else:
            est = res.params[colname]
            se = res.std_errors[colname]
            ci = res.conf_int(level=1 - ALPHA).loc[colname]
            coefs.append({"rel_time": k, "estimate": est, "se": se,
                          "ci_low": ci["lower"], "ci_high": ci["upper"],
                          "pvalue": res.pvalues[colname]})
    coefs_df = pd.DataFrame(coefs).sort_values("rel_time").reset_index(drop=True)

    return {
        "methode": "Event-study TWFE (dynamique)",
        "coefs": coefs_df,
        "model": res,
        "dummy_cols": cols,
        "window": window,
        "ref": ref,
    }


def plot_event_study(coefs_df, title, savepath, color="#2563eb",
                      overlay=None, overlay_label=None, overlay_color="#dc2626"):
    """Graphique standard d'event-study : coefficients +/- IC95%, ligne
    verticale entre la derniere periode pre-traitement et la premiere periode
    post-traitement, ligne horizontale a 0."""
    fig, ax = plt.subplots(figsize=(8.5, 5))

    def _plot(d, color, label):
        ax.errorbar(d["rel_time"], d["estimate"],
                     yerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]],
                     fmt="o-", color=color, markersize=4, capsize=3,
                     linewidth=1.5, label=label)

    _plot(coefs_df, color, "Event-study TWFE")
    if overlay is not None:
        _plot(overlay, overlay_color, overlay_label)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(-0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Temps relatif au traitement (periodes)")
    ax.set_ylabel("Effet estime sur Y")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ==============================================================================
# 6. DECOMPOSITION DE GOODMAN-BACON (diagnostic du biais du TWFE statique)
# ==============================================================================
def goodman_bacon_decomposition(df, group_col="group", time_col="time",
                                 unit_col="unit_id", outcome_col="Y",
                                 treated_col="treated"):
    """
    Decomposition "a la Goodman-Bacon" (Goodman-Bacon, 2021, Journal of
    Econometrics) du coefficient TWFE statique en moyenne ponderee de TOUTES
    les comparaisons 2x2 possibles entre paires de "groupes de calendrier de
    traitement" (chaque cohorte de traitement + le groupe jamais-traite,
    considere comme une cohorte traitee "a l'infini").

    Goodman-Bacon montre que beta_TWFE = somme ponderee des beta^2x2_kl, et
    que les comparaisons entre deux groupes EVENTUELLEMENT TOUS LES DEUX
    TRAITES ("deja-traite vs deja-traite") sont les comparaisons "a risque" :
    si les effets de traitement sont heterogenes/dynamiques dans le temps,
    elles biaisent l'estimateur global (un groupe deja traite sert de
    pseudo-controle alors qu'il a lui-meme un effet de traitement actif).

    IMPLEMENTATION : pour chaque paire de cohortes (k,l), on estime
    beta^2x2_kl par un TWFE restreint au sous-echantillon {k,l}. La ponderation
    utilisee ici est la ponderation generale de toute regression poolee comme
    moyenne ponderee de coefficients de sous-groupes (poids proportionnel a
    N_kl * Var(D_kl-residualise des effets fixes)) -- ce qui correspond a la
    logique du theoreme de Goodman-Bacon (variance de traitement intra-paire),
    sans reproduire bit-a-bit l'algorithme original. Pour une replication
    EXACTE de l'algorithme publie, utiliser le package R `bacondecomp`.
    """
    cohorts = df.groupby(group_col)["first_treat"].first()
    cohorts = cohorts.fillna(np.inf)  # jamais-traite = "traite a l'infini"
    labels = list(cohorts.index)

    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            k, l = labels[i], labels[j]
            ft_k, ft_l = cohorts[k], cohorts[l]
            sub = df[df[group_col].isin([k, l])].copy()
            n_kl = sub[unit_col].nunique()

            panel = sub.set_index([unit_col, time_col])
            try:
                mod = PanelOLS.from_formula(
                    f"{outcome_col} ~ 1 + {treated_col} + EntityEffects + TimeEffects",
                    data=panel, drop_absorbed=True)
                res = mod.fit(cov_type="clustered", cluster_entity=True)
                beta_kl = res.params[treated_col]
            except Exception:
                continue

            # Variance du traitement residualisee des effets fixes (poids)
            d = sub[[unit_col, time_col, treated_col]].copy()
            d["d_resid"] = (d[treated_col]
                             - d.groupby(unit_col)[treated_col].transform("mean")
                             - d.groupby(time_col)[treated_col].transform("mean")
                             + d[treated_col].mean())
            weight_raw = n_kl * d["d_resid"].var()

            both_eventually_treated = np.isfinite(ft_k) and np.isfinite(ft_l)
            comparison_type = ("Deux groupes traites (comparaison a risque)"
                                if both_eventually_treated
                                else "Traite vs jamais-traite (comparaison propre)")

            rows.append({
                "groupe_1": k, "groupe_2": l,
                "type_comparaison": comparison_type,
                "beta_2x2": beta_kl,
                "poids_brut": weight_raw,
                "n_obs": n_kl,
            })

    bacon_df = pd.DataFrame(rows)
    bacon_df["poids"] = bacon_df["poids_brut"] / bacon_df["poids_brut"].sum()
    bacon_df = bacon_df.drop(columns="poids_brut").sort_values("poids", ascending=False)

    beta_reconstructed = float((bacon_df["poids"] * bacon_df["beta_2x2"]).sum())
    return bacon_df, beta_reconstructed


def plot_bacon(bacon_df, beta_twfe, beta_reconstructed,
               savepath=os.path.join(FIG_DIR, "04_goodman_bacon.png")):
    """'Bacon plot' : chaque comparaison 2x2 est un point (poids, estimation).
    Les comparaisons 'a risque' (deux groupes eventuellement traites) sont
    distinguees des comparaisons propres (vs jamais-traite)."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"Traite vs jamais-traite (comparaison propre)": "#16a34a",
              "Deux groupes traites (comparaison a risque)": "#dc2626"}
    for ctype, sub in bacon_df.groupby("type_comparaison"):
        ax.scatter(sub["poids"], sub["beta_2x2"], s=70, alpha=0.85,
                   color=colors.get(ctype), label=ctype,
                   edgecolor="white", linewidth=0.6)
    for _, r in bacon_df.iterrows():
        lbl = f"{r['groupe_1'][:4]}.../{r['groupe_2'][:4]}...".replace("Jama", "Jamais")
        ax.annotate(lbl, (r["poids"], r["beta_2x2"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", alpha=0.8)

    ax.axhline(beta_twfe, color="black", linestyle="--", linewidth=1,
               label=f"TWFE statique (estime direct) = {beta_twfe:.2f}")
    ax.axhline(beta_reconstructed, color="grey", linestyle=":", linewidth=1,
               label=f"Moyenne ponderee reconstruite = {beta_reconstructed:.2f}")
    ax.set_xlabel("Poids dans la decomposition")
    ax.set_ylabel("Estimation DiD 2x2")
    ax.set_title("Decomposition de Goodman-Bacon du TWFE statique")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ==============================================================================
# 7. ESTIMATEUR DE CALLAWAY & SANT'ANNA (ATT(g,t)) -- staggered adoption
# ==============================================================================
def att_gt(df, g, t, control_group=NEVER_TREATED_LABEL, group_col="group",
           time_col="time", unit_col="unit_id", outcome_col="Y"):
    """
    Calcule l'effet moyen du traitement sur les traites pour la cohorte
    traitee en periode `g`, evalue en periode `t` :

        ATT(g,t) = E[Y_t - Y_(g-1) | G=g] - E[Y_t - Y_(g-1) | controle]

    C'est un DiD 2x2 "propre" entre la cohorte g et un groupe de comparaison
    qui n'est PAS encore traite en periode t (par defaut : jamais-traite).
    Pour t < g, ATT(g,t) constitue un test de pre-tendance pour la cohorte g
    (il doit etre proche de 0 si l'hypothese de tendances paralleles est
    valide pour cette cohorte).
    """
    cohort_label = df.loc[df["first_treat"] == g, group_col].iloc[0]
    anchor = g - 1
    sub = df[(df[group_col].isin([cohort_label, control_group])) &
             (df[time_col].isin([anchor, t]))].copy()
    sub["treat"] = (sub[group_col] == cohort_label).astype(int)
    sub["post"] = (sub[time_col] == t).astype(int)

    model = smf.ols(f"{outcome_col} ~ treat + post + treat:post", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub[unit_col]}
    )
    n_g = df.loc[df[group_col] == cohort_label, unit_col].nunique()
    return {
        "g": g, "t": t, "rel_time": t - g, "cohort": cohort_label,
        "estimate": model.params["treat:post"], "se": model.bse["treat:post"],
        "pvalue": model.pvalues["treat:post"], "n_cohort": n_g,
    }


def callaway_santanna(df, control_group=NEVER_TREATED_LABEL,
                       group_col="group", time_col="time"):
    """
    Calcule l'ensemble des ATT(g,t) pour toutes les cohortes traitees et
    toutes les periodes (y compris les periodes PRE-traitement, qui servent
    de test de pre-tendance specifique a chaque cohorte), puis agrege :

      - att_simple   : moyenne des ATT(g,t) post-traitement, ponderee par la
                        taille de chaque cohorte (estimateur ATT global)
      - att_dynamic  : ATT(e) par temps relatif au traitement e=t-g, moyenne
                        ponderee par taille de cohorte sur toutes les cohortes
                        ayant ce temps relatif e en commun (= "event-study"
                        robuste a l'heterogeneite dynamique des effets,
                        directement comparable a l'event-study TWFE naif)
    """
    cohorts_g = sorted(df.loc[df[group_col] != control_group, "first_treat"].dropna().unique())
    times = sorted(df[time_col].unique())

    all_results = []
    for g in cohorts_g:
        g = int(g)
        for t in times:
            if t == g - 1:
                continue  # periode d'ancrage elle-meme, non testee
            all_results.append(att_gt(df, g, t, control_group=control_group,
                                       group_col=group_col, time_col=time_col))
    att_df = pd.DataFrame(all_results)

    # --- Agregation "simple" (ATT global post-traitement) ---
    post = att_df[att_df["rel_time"] >= 0].copy()
    weights = post.drop_duplicates("cohort").set_index("cohort")["n_cohort"]
    post["w"] = post["cohort"].map(weights)
    att_simple = float((post["estimate"] * post["w"]).sum() / post["w"].sum())
    var_simple = float(((post["w"] / post["w"].sum()) ** 2 * post["se"] ** 2).sum())
    se_simple = float(np.sqrt(var_simple))

    # --- Agregation dynamique (event-study robuste, par temps relatif e) ---
    dyn_rows = []
    for e, sub in att_df.groupby("rel_time"):
        w = sub["n_cohort"]
        att_e = float((sub["estimate"] * w).sum() / w.sum())
        se_e = float(np.sqrt(((w / w.sum()) ** 2 * sub["se"] ** 2).sum()))
        dyn_rows.append({"rel_time": e, "estimate": att_e, "se": se_e,
                         "ci_low": att_e - sstats.norm.ppf(1 - ALPHA / 2) * se_e,
                         "ci_high": att_e + sstats.norm.ppf(1 - ALPHA / 2) * se_e,
                         "n_cohorts": sub["cohort"].nunique()})
    dyn_df = pd.DataFrame(dyn_rows).sort_values("rel_time").reset_index(drop=True)

    return {
        "att_gt": att_df,
        "att_simple": att_simple,
        "se_simple": se_simple,
        "event_study": dyn_df,
    }


# ==============================================================================
# 8. TESTS STATISTIQUES
# ==============================================================================
def test_parallel_trends(event_result):
    """
    Test conjoint de tendances paralleles : test de Wald (statistique du
    Chi2) de l'hypothese nulle jointe selon laquelle TOUS les coefficients
    "leads" (periodes pre-traitement, hors reference) de l'event-study sont
    simultanement egaux a 0.

        H0 : delta_k = 0  pour tout k < 0 (k != ref)
        H1 : il existe au moins un k < 0 tel que delta_k != 0

    Le rejet de H0 ne prouve pas la validite des tendances paralleles (on ne
    peut jamais prouver une hypothese de contrefactuel), mais leur NON-rejet
    est une condition necessaire (test de specification) generalement exigee
    avant de faire confiance a un DiD.
    """
    res = event_result["model"]
    cols = event_result["dummy_cols"]
    lead_cols = [c for c in cols if c.startswith("lead_")]
    formula = " , ".join([f"{c} = 0" for c in lead_cols])
    wald = res.wald_test(formula=formula)
    return {
        "test": "Test conjoint de tendances paralleles (Wald)",
        "h0": "Tous les coefficients 'leads' pre-traitement sont nuls",
        "statistic": float(wald.stat),
        "df": int(wald.df),
        "pvalue": float(wald.pval),
        "conclusion": ("Non-rejet de H0 : pas de signe de violation des "
                       "tendances paralleles au seuil de {:.0%}".format(ALPHA)
                       if wald.pval > ALPHA else
                       "Rejet de H0 : signe potentiel de violation des "
                       "tendances paralleles (a interpreter avec prudence, "
                       "cf. biais de contamination des effets fixes temps "
                       "en cas d'adoption echelonnee)"),
    }


def test_placebo(df, treat_group, control_group=NEVER_TREATED_LABEL,
                  fake_lag=4):
    """
    Test de placebo : on attribue une FAUSSE date de traitement, anterieure
    a la vraie date de traitement (a l'interieur de la periode pre-traitement
    uniquement), puis on re-estime un DiD 2x2 sur ces donnees ou personne
    n'est reellement traite. Sous H0 (pas d'effet anticipatoire, tendances
    paralleles), l'estimateur "placebo" doit etre proche de 0 et non
    significatif.
    """
    true_first_treat = int(df.loc[df["group"] == treat_group, "first_treat"].iloc[0])
    fake_post = true_first_treat - fake_lag
    fake_pre = fake_post - 1
    if fake_pre < df["time"].min():
        raise ValueError("fake_lag trop grand : periode placebo hors panel")

    res = did_2x2(df, treat_group=treat_group, control_group=control_group,
                   pre_period=fake_pre, post_period=fake_post)
    return {
        "test": "Test de placebo (fausse date de traitement pre-periode)",
        "h0": "Effet placebo nul (pas d'anticipation / tendances paralleles)",
        "fake_pre_period": fake_pre, "fake_post_period": fake_post,
        "estimate": res["estimate"], "se": res["se"], "pvalue": res["pvalue"],
        "conclusion": ("Non-rejet de H0 : aucun effet placebo detecte"
                       if res["pvalue"] > ALPHA else
                       "Rejet de H0 : effet placebo significatif, "
                       "possible violation des tendances paralleles ou "
                       "anticipation du traitement"),
    }


def test_permutation(df, treat_group, control_group=NEVER_TREATED_LABEL,
                      n_perm=N_PERMUTATIONS, seed=123):
    """
    Test de permutation / inference par randomisation : on calcule la
    distribution de l'estimateur DiD 2x2 SOUS H0 (effet nul) en reaffectant
    aleatoirement, `n_perm` fois, le statut "traite" parmi l'ensemble des
    unites (en conservant le nombre d'unites traitees), puis on situe
    l'estimation REELLE dans cette distribution nulle. Particulierement
    recommande quand le nombre de clusters (unites ou groupes) est faible,
    cas ou les tests asymptotiques bases sur l'erreur-type clusterisee sont
    peu fiables.
    """
    rng = np.random.default_rng(seed)
    sub = df[df["group"].isin([treat_group, control_group])].copy()
    real_result = did_2x2(df, treat_group=treat_group, control_group=control_group)
    real_estimate = real_result["estimate"]
    pre, post = real_result["pre_period"], real_result["post_period"]

    unit_ids = sub["unit_id"].unique()
    n_treated = sub.loc[sub["group"] == treat_group, "unit_id"].nunique()
    sub2p = sub[sub["time"].isin([pre, post])].copy()

    null_estimates = np.empty(n_perm)
    for b in range(n_perm):
        fake_treated_units = rng.choice(unit_ids, size=n_treated, replace=False)
        tmp = sub2p.copy()
        tmp["treat"] = tmp["unit_id"].isin(fake_treated_units).astype(int)
        tmp["post"] = (tmp["time"] == post).astype(int)
        m = smf.ols("Y ~ treat + post + treat:post", data=tmp).fit()
        null_estimates[b] = m.params["treat:post"]

    p_perm = float(np.mean(np.abs(null_estimates) >= np.abs(real_estimate)))
    return {
        "test": "Test de permutation (inference par randomisation)",
        "h0": "Effet de traitement nul",
        "real_estimate": real_estimate,
        "n_perm": n_perm,
        "pvalue_permutation": p_perm,
        "null_distribution": null_estimates,
        "conclusion": ("Non-rejet de H0" if p_perm > ALPHA else
                       "Rejet de H0 : l'estimation reelle est extreme par "
                       "rapport a la distribution nulle de permutation"),
    }


def test_heteroskedasticity(model_2x2):
    """
    Test de Breusch-Pagan d'heteroscedasticite des residus de la regression
    DiD 2x2. En cas de rejet de H0 (homoscedasticite), cela justifie a
    posteriori l'usage d'erreurs-types robustes (HC) ou clusterisees plutot
    que les erreurs-types OLS standard.
    """
    model = model_2x2["model"]
    bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    return {
        "test": "Test de Breusch-Pagan (heteroscedasticite)",
        "h0": "Homoscedasticite des residus",
        "statistic": float(bp_stat),
        "pvalue": float(bp_pval),
        "conclusion": ("Non-rejet de H0 : pas d'heteroscedasticite detectee"
                       if bp_pval > ALPHA else
                       "Rejet de H0 : heteroscedasticite detectee -> "
                       "l'usage d'erreurs-types robustes/clusterisees est "
                       "justifie (deja applique par defaut dans ce template)"),
    }


def cluster_bootstrap_se(df, treat_group, control_group=NEVER_TREATED_LABEL,
                          n_boot=N_BOOTSTRAP, seed=321):
    """
    Bootstrap par cluster (re-echantillonnage des UNITES avec remise, et non
    des observations individuelles) pour obtenir une erreur-type et un
    intervalle de confiance alternatifs a ceux, asymptotiques, de la
    regression avec erreurs-types clusterisees. Recommande en complement
    lorsque le nombre de clusters est faible (la theorie asymptotique du
    cluster-robust sandwich estimator est alors peu fiable).
    """
    rng = np.random.default_rng(seed)
    res0 = did_2x2(df, treat_group=treat_group, control_group=control_group)
    pre, post = res0["pre_period"], res0["post_period"]
    sub = df[(df["group"].isin([treat_group, control_group])) &
              (df["time"].isin([pre, post]))].copy()
    units = sub["unit_id"].unique()

    boot_estimates = np.empty(n_boot)
    for b in range(n_boot):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        frames = []
        for new_id, u in enumerate(sampled_units):
            tmp = sub[sub["unit_id"] == u].copy()
            tmp["unit_id"] = new_id  # re-identifier pour eviter les doublons d'index
            frames.append(tmp)
        boot_sample = pd.concat(frames, ignore_index=True)
        boot_sample["treat"] = (boot_sample["group"] == treat_group).astype(int)
        boot_sample["post"] = (boot_sample["time"] == post).astype(int)
        m = smf.ols("Y ~ treat + post + treat:post", data=boot_sample).fit()
        boot_estimates[b] = m.params["treat:post"]

    se_boot = float(np.std(boot_estimates, ddof=1))
    ci_low, ci_high = np.percentile(boot_estimates, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "test": "Bootstrap par cluster (unite) des erreurs-types",
        "estimate": res0["estimate"],
        "se_cluster_asymptotique": res0["se"],
        "se_bootstrap": se_boot,
        "ci_bootstrap_low": float(ci_low),
        "ci_bootstrap_high": float(ci_high),
        "n_boot": n_boot,
        "boot_distribution": boot_estimates,
    }


# ==============================================================================
# 9. GRAPHIQUES COMPLEMENTAIRES (CS vs TWFE, distributions des tests)
# ==============================================================================
def plot_cs_vs_twfe(cs_result, event_result,
                     savepath=os.path.join(FIG_DIR, "05_callaway_santanna_vs_twfe.png")):
    """Superpose l'event-study TWFE 'naif' (potentiellement biaise en cas
    d'heterogeneite dynamique des effets) et l'event-study issu de
    l'agregation des ATT(g,t) de Callaway & Sant'Anna (robuste)."""
    return plot_event_study(
        event_result["coefs"],
        title="Event-study : TWFE naif vs Callaway & Sant'Anna (robuste)",
        savepath=savepath,
        color="#9ca3af",
        overlay=cs_result["event_study"],
        overlay_label="Callaway & Sant'Anna (ATT(e) agrege)",
        overlay_color="#16a34a",
    )


def plot_distribution(values, observed, title, xlabel, savepath, color="#2563eb"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(values, bins=40, color=color, alpha=0.55, edgecolor="white")
    ax.axvline(observed, color="#dc2626", linewidth=2,
               label=f"Estimation observee = {observed:.2f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequence")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


def plot_att_gt_heatmap(att_df, savepath=os.path.join(FIG_DIR, "08_att_gt_heatmap.png")):
    """Carte des ATT(g,t) (cohorte x periode), incluant les periodes
    pre-traitement (test de pre-tendance specifique a chaque cohorte)."""
    pivot = att_df.pivot(index="cohort", columns="t", values="estimate")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(pivot.values)),
                   vmax=np.nanmax(np.abs(pivot.values)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=90)
    ax.set_xlabel("Periode (t)")
    ax.set_title("ATT(g,t) de Callaway & Sant'Anna par cohorte et periode\n"
                  "(zones avant la date de traitement = test de pre-tendance)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="ATT(g,t)")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
    return savepath


# ==============================================================================
# 10. PIPELINE PRINCIPALE
# ==============================================================================
def run_full_analysis(treat_group_2x2="Cohorte_2017"):
    """Execute l'ensemble de l'analyse DiD (toutes variantes + tous les
    tests + tous les graphiques), exporte un recapitulatif chiffre en CSV,
    et renvoie un dictionnaire de resultats utilise ensuite pour generer le
    rapport PDF."""
    print("=" * 80)
    print("ETAPE 1/9 : chargement de la base de donnees")
    print("=" * 80)
    data = load_data()

    print("\n" + "=" * 80)
    print("ETAPE 2/9 : statistiques descriptives & tendances brutes")
    print("=" * 80)
    fig_trends = plot_raw_trends(data)
    print(f"  -> figure : {fig_trends}")

    print("\n" + "=" * 80)
    print(f"ETAPE 3/9 : DiD classique 2x2 ({treat_group_2x2} vs {NEVER_TREATED_LABEL})")
    print("=" * 80)
    res_2x2 = did_2x2(data, treat_group=treat_group_2x2)
    fig_2x2 = plot_2x2_visual(res_2x2)
    print(f"  Estimation : {res_2x2['estimate']:.3f} (se={res_2x2['se']:.3f}, "
          f"IC95%=[{res_2x2['ci_low']:.3f}, {res_2x2['ci_high']:.3f}], "
          f"p={res_2x2['pvalue']:.4f})")
    print(f"  -> figure : {fig_2x2}")

    print("\n" + "=" * 80)
    print("ETAPE 4/9 : DiD multi-periodes -- TWFE statique")
    print("=" * 80)
    res_twfe = did_twfe_static(data)
    print(f"  Estimation : {res_twfe['estimate']:.3f} (se={res_twfe['se']:.3f}, "
          f"p={res_twfe['pvalue']:.4f})")

    print("\n" + "=" * 80)
    print("ETAPE 5/9 : DiD dynamique -- event-study TWFE")
    print("=" * 80)
    res_event = did_event_study(data)
    fig_event = plot_event_study(
        res_event["coefs"], title="Event-study TWFE (toutes cohortes poolees)",
        savepath=os.path.join(FIG_DIR, "03_event_study_twfe.png"))
    print(f"  -> figure : {fig_event}")

    print("\n" + "=" * 80)
    print("ETAPE 6/9 : decomposition de Goodman-Bacon")
    print("=" * 80)
    bacon_df, beta_reconstructed = goodman_bacon_decomposition(data)
    fig_bacon = plot_bacon(bacon_df, res_twfe["estimate"], beta_reconstructed)
    print(bacon_df.to_string(index=False))
    print(f"  TWFE direct = {res_twfe['estimate']:.3f} | "
          f"Reconstruction ponderee = {beta_reconstructed:.3f}")
    print(f"  -> figure : {fig_bacon}")

    print("\n" + "=" * 80)
    print("ETAPE 7/9 : estimateur de Callaway & Sant'Anna (ATT(g,t))")
    print("=" * 80)
    cs_result = callaway_santanna(data)
    print(f"  ATT global (agregation simple) = {cs_result['att_simple']:.3f} "
          f"(se={cs_result['se_simple']:.3f})")
    fig_cs_vs_twfe = plot_cs_vs_twfe(cs_result, res_event)
    fig_att_heatmap = plot_att_gt_heatmap(cs_result["att_gt"])
    print(f"  -> figures : {fig_cs_vs_twfe}, {fig_att_heatmap}")

    print("\n" + "=" * 80)
    print("ETAPE 8/9 : tests statistiques")
    print("=" * 80)
    t_parallel = test_parallel_trends(res_event)
    print(f"  [Tendances paralleles] stat={t_parallel['statistic']:.3f} "
          f"df={t_parallel['df']} p={t_parallel['pvalue']:.4f} -> {t_parallel['conclusion']}")

    t_placebo = test_placebo(data, treat_group=treat_group_2x2)
    print(f"  [Placebo] estimate={t_placebo['estimate']:.3f} "
          f"p={t_placebo['pvalue']:.4f} -> {t_placebo['conclusion']}")

    t_perm = test_permutation(data, treat_group=treat_group_2x2)
    fig_perm = plot_distribution(
        t_perm["null_distribution"], t_perm["real_estimate"],
        title="Test de permutation : distribution nulle vs estimation reelle",
        xlabel="Estimation DiD 2x2 sous permutations aleatoires",
        savepath=os.path.join(FIG_DIR, "06_permutation.png"))
    print(f"  [Permutation] p={t_perm['pvalue_permutation']:.4f} -> {t_perm['conclusion']}")
    print(f"  -> figure : {fig_perm}")

    t_hetero = test_heteroskedasticity(res_2x2)
    print(f"  [Breusch-Pagan] stat={t_hetero['statistic']:.3f} "
          f"p={t_hetero['pvalue']:.4f} -> {t_hetero['conclusion']}")

    t_boot = cluster_bootstrap_se(data, treat_group=treat_group_2x2)
    fig_boot = plot_distribution(
        t_boot["boot_distribution"], t_boot["estimate"],
        title="Bootstrap par cluster (unite) de l'estimateur DiD 2x2",
        xlabel="Estimation DiD 2x2 (replications bootstrap)",
        savepath=os.path.join(FIG_DIR, "07_bootstrap.png"), color="#16a34a")
    print(f"  [Bootstrap cluster] se_asymptotique={t_boot['se_cluster_asymptotique']:.3f} "
          f"se_bootstrap={t_boot['se_bootstrap']:.3f}")
    print(f"  -> figure : {fig_boot}")

    print("\n" + "=" * 80)
    print("ETAPE 9/9 : export du recapitulatif chiffre")
    print("=" * 80)
    summary_rows = [
        {"methode": "DiD 2x2", "comparaison": res_2x2["comparaison"],
         "estimation": res_2x2["estimate"], "se": res_2x2["se"],
         "ic_bas": res_2x2["ci_low"], "ic_haut": res_2x2["ci_high"],
         "p_value": res_2x2["pvalue"]},
        {"methode": "TWFE statique (multi-periodes, multi-cohortes)",
         "comparaison": "toutes cohortes", "estimation": res_twfe["estimate"],
         "se": res_twfe["se"], "ic_bas": res_twfe["ci_low"],
         "ic_haut": res_twfe["ci_high"], "p_value": res_twfe["pvalue"]},
        {"methode": "Callaway & Sant'Anna (ATT global agrege)",
         "comparaison": "toutes cohortes vs jamais-traite",
         "estimation": cs_result["att_simple"], "se": cs_result["se_simple"],
         "ic_bas": cs_result["att_simple"] - 1.96 * cs_result["se_simple"],
         "ic_haut": cs_result["att_simple"] + 1.96 * cs_result["se_simple"],
         "p_value": np.nan},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_PATH, index=False)
    print(f"  -> recapitulatif sauvegarde : {RESULTS_PATH}")

    return {
        "data": data,
        "res_2x2": res_2x2,
        "res_twfe": res_twfe,
        "res_event": res_event,
        "bacon_df": bacon_df,
        "beta_reconstructed": beta_reconstructed,
        "cs_result": cs_result,
        "t_parallel": t_parallel,
        "t_placebo": t_placebo,
        "t_perm": t_perm,
        "t_hetero": t_hetero,
        "t_boot": t_boot,
        "summary_df": summary_df,
        "figures": {
            "trends": fig_trends, "did_2x2": fig_2x2, "event_study": fig_event,
            "bacon": fig_bacon, "cs_vs_twfe": fig_cs_vs_twfe,
            "att_heatmap": fig_att_heatmap, "permutation": fig_perm,
            "bootstrap": fig_boot,
        },
    }


if __name__ == "__main__":
    results = run_full_analysis()
    print("\nAnalyse terminee. Generation du rapport PDF...")
    from generate_pdf_report import build_report
    build_report(results)
