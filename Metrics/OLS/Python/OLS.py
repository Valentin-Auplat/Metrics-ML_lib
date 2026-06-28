"""
Template d'analyse OLS (régression linéaire) — diagnostic complet
====================================================================

Objectif pédagogique : régresser une variable continue sur d'autres
variables continues (ici, sur le jeu de données Iris) et appliquer
la démarche standard d'un économètre :

    1. OLS "classique" (erreurs homoscédastiques, i.i.d.)
    2. Tests de diagnostic des hypothèses du modèle linéaire (Gauss-Markov)
    3. OLS avec erreurs-types robustes (HC3) si l'homoscédasticité est rejetée
    4. Comparaison des deux estimations
    5. Graphiques diagnostiques usuels

Le jeu de données Iris sert ici uniquement de support numérique
("type iris" demandé) : on régresse la longueur des pétales sur les
trois autres variables numériques, ce qui n'a aucune portée biologique
particulière mais permet d'illustrer la démarche sur un jeu de données
propre, sans valeurs manquantes ni nettoyage préalable nécessaire.

Dépendances : pandas, numpy, scipy, statsmodels, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
from sklearn.datasets import load_iris


# ====================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ====================================================================

def load_data() -> pd.DataFrame:
    """Charge Iris et renvoie un DataFrame propre, prêt pour la régression."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


# Variable dépendante et régresseurs du modèle.
# À adapter directement à votre propre jeu de données : il suffit de
# remplacer DEP_VAR / REGRESSORS et la fonction load_data() ci-dessus.
DEP_VAR = "petal_length"
REGRESSORS = ["sepal_length", "sepal_width", "petal_width"]


# ====================================================================
# 2. ESTIMATION OLS — VERSION CLASSIQUE
# ====================================================================

def fit_ols(df: pd.DataFrame, dep_var: str, regressors: list[str]):
    """Estime un OLS classique (matrice de covariance homoscédastique)."""
    formula = f"{dep_var} ~ {' + '.join(regressors)}"
    model = smf.ols(formula=formula, data=df).fit()
    return model, formula


# ====================================================================
# 3. TESTS DE DIAGNOSTIC DES HYPOTHÈSES DU MODÈLE LINÉAIRE
# ====================================================================

def run_diagnostics(model, df: pd.DataFrame, regressors: list[str]) -> dict:
    """
    Bloc de tests statistiques usuels associés aux hypothèses de Gauss-Markov :

      - Normalité des résidus     : Jarque-Bera, Shapiro-Wilk
      - Homoscédasticité          : Breusch-Pagan, White
      - Autocorrélation           : Durbin-Watson, Ljung-Box
      - Multicolinéarité          : VIF (Variance Inflation Factor)
      - Spécification du modèle   : test RESET (Ramsey)
    """
    resid = model.resid
    fitted = model.fittedvalues
    exog = model.model.exog
    results = {}

    # --- Normalité des résidus ---
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    results["jarque_bera"] = {"stat": jb_stat, "p_value": jb_p, "skew": skew, "kurtosis": kurt}

    sw_stat, sw_p = stats.shapiro(resid)
    results["shapiro_wilk"] = {"stat": sw_stat, "p_value": sw_p}

    # --- Homoscédasticité ---
    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid, exog)
    results["breusch_pagan"] = {"lm_stat": bp_stat, "lm_p_value": bp_p,
                                 "f_stat": bp_f, "f_p_value": bp_fp}

    white_stat, white_p, white_f, white_fp = het_white(resid, exog)
    results["white"] = {"lm_stat": white_stat, "lm_p_value": white_p,
                         "f_stat": white_f, "f_p_value": white_fp}

    # --- Autocorrélation ---
    results["durbin_watson"] = durbin_watson(resid)

    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    results["ljung_box"] = {"stat": lb["lb_stat"].iloc[0], "p_value": lb["lb_pvalue"].iloc[0]}

    # --- Multicolinéarité (VIF) ---
    X_vif = sm.add_constant(df[regressors])
    vif_data = pd.DataFrame({
        "variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    results["vif"] = vif_data[vif_data["variable"] != "const"].reset_index(drop=True)

    # --- Mauvaise spécification (test RESET de Ramsey, version simplifiée) ---
    # On teste si fitted^2 et fitted^3 ajoutent un pouvoir explicatif significatif.
    df_reset = df.copy()
    df_reset["_fitted2"] = fitted ** 2
    df_reset["_fitted3"] = fitted ** 3
    reset_formula = f"{DEP_VAR} ~ {' + '.join(regressors)} + _fitted2 + _fitted3"
    reset_model = smf.ols(reset_formula, data=df_reset).fit()
    reset_test = reset_model.compare_f_test(model)
    results["reset_test"] = {"f_stat": reset_test[0], "p_value": reset_test[1]}

    return results


def print_diagnostics(diag: dict, alpha: float = 0.05) -> None:
    """Affiche les résultats des tests avec une interprétation automatique."""
    print("\n" + "=" * 70)
    print("TESTS DE DIAGNOSTIC (seuil de significativité : {:.0%})".format(alpha))
    print("=" * 70)

    def verdict(p, hypothesis_rejected_means):
        flag = "REJET H0" if p < alpha else "non rejet de H0"
        return f"p = {p:.4f}  ->  {flag}  ({hypothesis_rejected_means if p < alpha else 'OK'})"

    jb = diag["jarque_bera"]
    print(f"\n[Normalité] Jarque-Bera : {verdict(jb['p_value'], 'résidus non normaux')}")
    sw = diag["shapiro_wilk"]
    print(f"[Normalité] Shapiro-Wilk : {verdict(sw['p_value'], 'résidus non normaux')}")

    bp = diag["breusch_pagan"]
    print(f"\n[Homoscédasticité] Breusch-Pagan : {verdict(bp['lm_p_value'], 'hétéroscédasticité détectée')}")
    wh = diag["white"]
    print(f"[Homoscédasticité] White         : {verdict(wh['lm_p_value'], 'hétéroscédasticité détectée')}")

    dw = diag["durbin_watson"]
    print(f"\n[Autocorrélation] Durbin-Watson = {dw:.3f} "
          f"(proche de 2 = pas d'autocorrélation ; <1.5 ou >2.5 = suspect)")
    lb = diag["ljung_box"]
    print(f"[Autocorrélation] Ljung-Box (10 lags) : {verdict(lb['p_value'], 'autocorrélation détectée')}")

    print("\n[Multicolinéarité] VIF par variable (seuil d'alerte usuel : VIF > 5 ou 10) :")
    print(diag["vif"].to_string(index=False))

    reset = diag["reset_test"]
    print(f"\n[Spécification] Test RESET (Ramsey) : "
          f"{verdict(reset['p_value'], 'mauvaise spécification du modèle (non-linéarité omise)')}")

    print("\n" + "-" * 70)
    if bp["lm_p_value"] < alpha or wh["lm_p_value"] < alpha:
        print(">>> Hétéroscédasticité détectée : passer à des erreurs-types robustes (HC3) recommandé.")
    else:
        print(">>> Pas d'hétéroscédasticité détectée : l'OLS classique reste valide,")
        print("    mais les erreurs robustes sont présentées ci-dessous à titre de comparaison.")
    print("-" * 70)


# ====================================================================
# 4. OLS AVEC ERREURS-TYPES ROBUSTES (HC3)
# ====================================================================

def fit_ols_robust(df: pd.DataFrame, dep_var: str, regressors: list[str]):
    """Même modèle, réestimé avec une matrice de covariance robuste HC3
    (Davidson-MacKinnon) : plus prudente que HC1/HC2 pour les petits échantillons."""
    formula = f"{dep_var} ~ {' + '.join(regressors)}"
    model_robust = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    return model_robust


def compare_models(model_classic, model_robust, regressors: list[str]) -> pd.DataFrame:
    """Construit un tableau comparatif coefficient / écart-type classique vs robuste."""
    comp = pd.DataFrame({
        "coefficient": model_classic.params,
        "se_classique": model_classic.bse,
        "se_robuste_HC3": model_robust.bse,
        "p_value_classique": model_classic.pvalues,
        "p_value_robuste_HC3": model_robust.pvalues,
    })
    comp["ecart_se_%"] = 100 * (comp["se_robuste_HC3"] - comp["se_classique"]) / comp["se_classique"]
    return comp.round(4)


# ====================================================================
# 5. GRAPHIQUES DIAGNOSTIQUES
# ====================================================================

def make_diagnostic_plots(model, df: pd.DataFrame, regressors: list[str], dep_var: str,
                           save_path: str = "ols_diagnostics.png") -> None:
    """
    Quatre graphiques diagnostiques classiques en régression linéaire,
    inspirés des plot.lm() de R :

      1. Résidus vs valeurs ajustées       -> détecte non-linéarité / hétéroscédasticité
      2. QQ-plot des résidus                -> détecte non-normalité
      3. Scale-Location (racine des résidus standardisés vs ajustés) -> hétéroscédasticité
      4. Résidus vs leverage (distance de Cook) -> points influents
    """
    resid = model.resid
    fitted = model.fittedvalues
    influence = OLSInfluence(model)
    standardized_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- 1. Résidus vs valeurs ajustées ---
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    # tendance lissée (LOWESS) pour repérer une courbure résiduelle
    lowess = sm.nonparametric.lowess(resid, fitted, frac=0.6)
    ax.plot(lowess[:, 0], lowess[:, 1], color="firebrick", linewidth=1.5)
    ax.set_xlabel("Valeurs ajustées")
    ax.set_ylabel("Résidus")
    ax.set_title("Résidus vs valeurs ajustées")

    # --- 2. QQ-plot des résidus ---
    ax = axes[0, 1]
    qqplot(resid, line="45", ax=ax)
    ax.set_title("QQ-plot des résidus (normalité)")

    # --- 3. Scale-Location ---
    ax = axes[1, 0]
    sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))
    ax.scatter(fitted, sqrt_abs_resid, alpha=0.6, edgecolor="k", linewidth=0.3)
    lowess2 = sm.nonparametric.lowess(sqrt_abs_resid, fitted, frac=0.6)
    ax.plot(lowess2[:, 0], lowess2[:, 1], color="firebrick", linewidth=1.5)
    ax.set_xlabel("Valeurs ajustées")
    ax.set_ylabel(r"$\sqrt{|\mathrm{résidus\ standardisés}|}$")
    ax.set_title("Scale-Location (homoscédasticité)")

    # --- 4. Résidus vs leverage + distance de Cook ---
    ax = axes[1, 1]
    ax.scatter(leverage, standardized_resid, alpha=0.6, edgecolor="k", linewidth=0.3,
               s=20 + 300 * cooks_d / cooks_d.max())  # taille proportionnelle à Cook's D
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    threshold = 4 / len(df)  # seuil usuel de levier élevé : 2(k+1)/n ou 4/n selon convention
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"seuil levier = {threshold:.3f}")
    ax.set_xlabel("Levier (hat values)")
    ax.set_ylabel("Résidus studentisés")
    ax.set_title("Résidus vs levier (taille = distance de Cook)")
    ax.legend(fontsize=8)

    fig.suptitle("Diagnostics du modèle OLS", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n[Graphique] Diagnostics sauvegardés -> {save_path}")
    plt.close(fig)


def make_coefficient_plot(comp: pd.DataFrame, save_path: str = "ols_coef_comparison.png") -> None:
    """Forest plot comparant les intervalles de confiance à 95% classiques vs robustes,
    pour visualiser directement l'impact de la correction HC3 sur la précision estimée."""
    comp_no_const = comp.drop(index="Intercept", errors="ignore")
    vars_ = comp_no_const.index
    y_pos = np.arange(len(vars_))

    fig, ax = plt.subplots(figsize=(8, 0.8 * len(vars_) + 2))

    ci_classic = 1.96 * comp_no_const["se_classique"]
    ci_robust = 1.96 * comp_no_const["se_robuste_HC3"]

    ax.errorbar(comp_no_const["coefficient"], y_pos - 0.12, xerr=ci_classic,
                fmt="o", color="steelblue", label="OLS classique (IC 95%)", capsize=4)
    ax.errorbar(comp_no_const["coefficient"], y_pos + 0.12, xerr=ci_robust,
                fmt="o", color="firebrick", label="OLS robuste HC3 (IC 95%)", capsize=4)

    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_)
    ax.set_xlabel("Coefficient estimé")
    ax.set_title("Comparaison des coefficients : erreurs-types classiques vs robustes (HC3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"[Graphique] Comparaison des coefficients -> {save_path}")
    plt.close(fig)


def make_exploratory_plots(df: pd.DataFrame, dep_var: str, regressors: list[str],
                            save_path: str = "ols_exploratory.png") -> None:
    """Nuages de points dep_var vs chaque régresseur, avant toute estimation
    -- étape exploratoire usuelle pour juger de la pertinence d'une relation linéaire."""
    n = len(regressors)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, reg in zip(axes, regressors):
        ax.scatter(df[reg], df[dep_var], alpha=0.6, edgecolor="k", linewidth=0.3)
        # droite de régression simple univariée, à titre indicatif seulement
        slope, intercept, r, p, se = stats.linregress(df[reg], df[dep_var])
        x_line = np.linspace(df[reg].min(), df[reg].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color="firebrick", linewidth=1.5,
                label=f"r = {r:.2f}")
        ax.set_xlabel(reg)
        ax.set_ylabel(dep_var)
        ax.legend(fontsize=8)
    fig.suptitle(f"Relations bivariées avec {dep_var} (exploratoire)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    fig.savefig(save_path, dpi=150)
    print(f"[Graphique] Exploration bivariée -> {save_path}")
    plt.close(fig)


# ====================================================================
# 6. PROGRAMME PRINCIPAL
# ====================================================================

def main():
    df = load_data()
    print(df.describe().round(2))

    # --- Exploration graphique préalable ---
    make_exploratory_plots(df, DEP_VAR, REGRESSORS)

    # --- OLS classique ---
    model, formula = fit_ols(df, DEP_VAR, REGRESSORS)
    print("\n" + "=" * 70)
    print(f"OLS CLASSIQUE  |  formule : {formula}")
    print("=" * 70)
    print(model.summary())

    # --- Diagnostics ---
    diag = run_diagnostics(model, df, REGRESSORS)
    print_diagnostics(diag)

    # --- OLS robuste (HC3) ---
    model_robust = fit_ols_robust(df, DEP_VAR, REGRESSORS)
    print("\n" + "=" * 70)
    print("OLS AVEC ERREURS-TYPES ROBUSTES (HC3)")
    print("=" * 70)
    print(model_robust.summary())

    # --- Comparaison classique vs robuste ---
    comp = compare_models(model, model_robust, REGRESSORS)
    print("\n" + "=" * 70)
    print("COMPARAISON OLS CLASSIQUE vs OLS ROBUSTE (HC3)")
    print("=" * 70)
    print(comp.to_string())

    # --- Graphiques diagnostiques et de comparaison ---
    make_diagnostic_plots(model, df, REGRESSORS, DEP_VAR)
    make_coefficient_plot(comp)
    
    print("\nAnalyse terminée. Fichiers générés : "
          "ols_exploratory.png, ols_diagnostics.png, ols_coef_comparison.png")


if __name__ == "__main__":
    main()
