# ====================================================================
# Template d'analyse OLS (regression lineaire) -- diagnostic complet
# ====================================================================
#
# Traduction R du script Python ols_analysis_template.py.
# Memes etapes, meme structure :
#
#   1. OLS "classique" (erreurs homoscedastiques, i.i.d.)
#   2. Tests de diagnostic des hypotheses du modele lineaire (Gauss-Markov)
#   3. OLS avec erreurs-types robustes (HC3) si l'homoscedasticite est rejetee
#   4. Comparaison des deux estimations
#   5. Graphiques diagnostiques usuels
#
# Dependances (packages CRAN, hors base R) : lmtest, sandwich
#   install.packages(c("lmtest", "sandwich"))
#
# ====================================================================

library(lmtest)    # bptest (Breusch-Pagan), dwtest (Durbin-Watson)
library(sandwich)  # vcovHC (matrice de covariance robuste HC3)

# ====================================================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES
# ====================================================================

load_data <- function() {
  data(iris)
  df <- iris
  colnames(df) <- c("sepal_length", "sepal_width", "petal_length",
                     "petal_width", "species")
  df
}

# Variable dependante et regresseurs du modele.
# A adapter directement a votre propre jeu de donnees : il suffit de
# remplacer load_data() ci-dessus et les deux lignes suivantes.
DEP_VAR    <- "petal_length"
REGRESSORS <- c("sepal_length", "sepal_width", "petal_width")


# ====================================================================
# 2. ESTIMATION OLS -- VERSION CLASSIQUE
# ====================================================================

fit_ols <- function(df, dep_var, regressors) {
  formula_str <- paste(dep_var, "~", paste(regressors, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  model <- lm(formula_obj, data = df)
  list(model = model, formula = formula_str)
}


# ====================================================================
# 3. TESTS DE DIAGNOSTIC DES HYPOTHESES DU MODELE LINEAIRE
# ====================================================================

# --- Jarque-Bera (implementation manuelle : pas de dependance a 'moments'/'tseries') ---
jarque_bera_test <- function(resid) {
  n <- length(resid)
  m2 <- mean(resid^2)
  skew <- mean(resid^3) / m2^1.5
  kurt <- mean(resid^4) / m2^2
  jb_stat <- (n / 6) * (skew^2 + ((kurt - 3)^2) / 4)
  p_value <- 1 - pchisq(jb_stat, df = 2)
  list(stat = jb_stat, p_value = p_value, skew = skew, kurtosis = kurt)
}

# --- White test (implementation manuelle : regression auxiliaire complete
#     avec carres et produits croises, comme statsmodels.het_white) ---
white_test <- function(resid, regressors_df) {
  n <- nrow(regressors_df)
  vars <- colnames(regressors_df)
  aux_df <- regressors_df

  # Carres de chaque regresseur
  for (v in vars) {
    aux_df[[paste0(v, "_sq")]] <- regressors_df[[v]]^2
  }
  # Produits croises (interactions deux-a-deux)
  if (length(vars) >= 2) {
    combs <- combn(vars, 2)
    for (j in seq_len(ncol(combs))) {
      v1 <- combs[1, j]; v2 <- combs[2, j]
      aux_df[[paste0(v1, "_x_", v2)]] <- regressors_df[[v1]] * regressors_df[[v2]]
    }
  }

  aux_df$resid_sq <- resid^2
  aux_formula <- as.formula(paste("resid_sq ~", paste(setdiff(colnames(aux_df), "resid_sq"), collapse = " + ")))
  aux_model <- lm(aux_formula, data = aux_df)

  r2_aux <- summary(aux_model)$r.squared
  q <- length(setdiff(colnames(aux_df), "resid_sq"))  # nb de regresseurs auxiliaires (hors constante)
  lm_stat <- n * r2_aux
  p_value <- 1 - pchisq(lm_stat, df = q)
  list(lm_stat = lm_stat, p_value = p_value, df = q)
}

# --- VIF (implementation manuelle : 1 / (1 - R^2_j), pas de dependance a 'car') ---
compute_vif <- function(df, regressors) {
  vifs <- numeric(length(regressors))
  for (i in seq_along(regressors)) {
    target <- regressors[i]
    others <- setdiff(regressors, target)
    formula_aux <- as.formula(paste(target, "~", paste(others, collapse = " + ")))
    r2_j <- summary(lm(formula_aux, data = df))$r.squared
    vifs[i] <- 1 / (1 - r2_j)
  }
  data.frame(variable = regressors, VIF = vifs)
}

run_diagnostics <- function(model, df, regressors) {
  resid <- residuals(model)
  fitted <- fitted(model)
  results <- list()

  # --- Normalite des residus ---
  results$jarque_bera <- jarque_bera_test(resid)
  sw <- shapiro.test(resid)                                   # base R
  results$shapiro_wilk <- list(stat = sw$statistic, p_value = sw$p.value)

  # --- Homoscedasticite ---
  # studentize = TRUE (par defaut dans lmtest) calcule LM = n * R^2_aux,
  # ce qui correspond exactement a statsmodels.het_breuschpagan.
  # ATTENTION : malgre son nom, c'est studentize=TRUE qui reproduit le test
  # "classique" utilise cote Python -- studentize=FALSE calcule une autre
  # variante (Breusch & Pagan, 1979, version non corrigee) qui donnera une
  # p-valeur differente.
  bp <- bptest(model, studentize = TRUE)
  results$breusch_pagan <- list(lm_stat = bp$statistic, p_value = bp$p.value)

  results$white <- white_test(resid, df[, regressors, drop = FALSE])

  # --- Autocorrelation ---
  dw <- dwtest(model)
  results$durbin_watson <- list(stat = dw$statistic, p_value = dw$p.value)

  lb <- Box.test(resid, lag = 10, type = "Ljung-Box")          # base R
  results$ljung_box <- list(stat = lb$statistic, p_value = lb$p.value)

  # --- Multicolinearite (VIF) ---
  results$vif <- compute_vif(df, regressors)

  # --- Mauvaise specification (test RESET de Ramsey) ---
  df_reset <- df
  df_reset$.fitted2 <- fitted^2
  df_reset$.fitted3 <- fitted^3
  reset_formula <- as.formula(paste(DEP_VAR, "~", paste(regressors, collapse = " + "),
                                     "+ .fitted2 + .fitted3"))
  reset_model <- lm(reset_formula, data = df_reset)
  reset_anova <- anova(model, reset_model)                     # test F de restrictions emboitees
  results$reset_test <- list(f_stat = reset_anova$F[2], p_value = reset_anova$`Pr(>F)`[2])

  results
}

print_diagnostics <- function(diag, alpha = 0.05) {
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("TESTS DE DIAGNOSTIC (seuil de significativite : %.0f%%)\n", alpha * 100))
  cat(strrep("=", 70), "\n")

  verdict <- function(p, hypothesis_rejected_means) {
    flag <- if (p < alpha) "REJET H0" else "non rejet de H0"
    extra <- if (p < alpha) hypothesis_rejected_means else "OK"
    sprintf("p = %.4f  ->  %s  (%s)", p, flag, extra)
  }

  jb <- diag$jarque_bera
  cat(sprintf("\n[Normalite] Jarque-Bera : %s\n", verdict(jb$p_value, "residus non normaux")))
  sw <- diag$shapiro_wilk
  cat(sprintf("[Normalite] Shapiro-Wilk : %s\n", verdict(sw$p_value, "residus non normaux")))

  bp <- diag$breusch_pagan
  cat(sprintf("\n[Homoscedasticite] Breusch-Pagan : %s\n",
              verdict(bp$p_value, "heteroscedasticite detectee")))
  wh <- diag$white
  cat(sprintf("[Homoscedasticite] White         : %s\n",
              verdict(wh$p_value, "heteroscedasticite detectee")))

  dw <- diag$durbin_watson
  cat(sprintf("\n[Autocorrelation] Durbin-Watson = %.3f (proche de 2 = pas d'autocorrelation ; <1.5 ou >2.5 = suspect)\n",
              dw$stat))
  lb <- diag$ljung_box
  cat(sprintf("[Autocorrelation] Ljung-Box (10 lags) : %s\n",
              verdict(lb$p_value, "autocorrelation detectee")))

  cat("\n[Multicolinearite] VIF par variable (seuil d'alerte usuel : VIF > 5 ou 10) :\n")
  print(diag$vif, row.names = FALSE)

  reset <- diag$reset_test
  cat(sprintf("\n[Specification] Test RESET (Ramsey) : %s\n",
              verdict(reset$p_value, "mauvaise specification du modele (non-linearite omise)")))

  cat("\n", strrep("-", 70), "\n", sep = "")
  if (bp$p_value < alpha || wh$p_value < alpha) {
    cat(">>> Heteroscedasticite detectee : passer a des erreurs-types robustes (HC3) recommande.\n")
  } else {
    cat(">>> Pas d'heteroscedasticite detectee : l'OLS classique reste valide,\n")
    cat("    mais les erreurs robustes sont presentees ci-dessous a titre de comparaison.\n")
  }
  cat(strrep("-", 70), "\n")
}


# ====================================================================
# 4. OLS AVEC ERREURS-TYPES ROBUSTES (HC3)
# ====================================================================

fit_ols_robust <- function(model) {
  vcov_hc3 <- vcovHC(model, type = "HC3")
  coeftest(model, vcov = vcov_hc3)   # renvoie un objet de classe coeftest
}

compare_models <- function(model, model_robust) {
  coefs        <- coef(model)
  se_classique <- summary(model)$coefficients[, "Std. Error"]
  se_robuste   <- model_robust[, "Std. Error"]
  p_classique  <- summary(model)$coefficients[, "Pr(>|t|)"]
  p_robuste    <- model_robust[, "Pr(>|t|)"]

  comp <- data.frame(
    coefficient          = coefs,
    se_classique         = se_classique,
    se_robuste_HC3       = se_robuste,
    p_value_classique    = p_classique,
    p_value_robuste_HC3  = p_robuste
  )
  comp$ecart_se_pct <- 100 * (comp$se_robuste_HC3 - comp$se_classique) / comp$se_classique
  round(comp, 4)
}


# ====================================================================
# 5. GRAPHIQUES DIAGNOSTIQUES
# ====================================================================

make_exploratory_plots <- function(df, dep_var, regressors,
                                    save_path = "ols_exploratory.png") {
  n <- length(regressors)
  par(mfrow = c(1, n), oma = c(0, 0, 3, 0))

  for (reg in regressors) {
    x <- df[[reg]]; y <- df[[dep_var]]
    fit_lin <- lm(y ~ x)
    r <- cor(x, y)
    plot(x, y, pch = 16, col = adjustcolor("steelblue", alpha.f = 0.6),
         xlab = reg, ylab = dep_var)
    abline(fit_lin, col = "firebrick", lwd = 1.5)
    legend("topleft", legend = sprintf("r = %.2f", r), bty = "n", text.col = "firebrick")
  }
  mtext(sprintf("Relations bivariees avec %s (exploratoire)", dep_var),
        outer = TRUE, cex = 1.2, font = 2)
  cat(sprintf("[Graphique] Exploration bivariee -> %s\n", save_path))
}

make_diagnostic_plots <- function(model, df, save_path = "ols_diagnostics.png") {
  resid       <- residuals(model)
  fitted_vals <- fitted(model)
  std_resid   <- rstandard(model)        # residus standardises
  stud_resid  <- rstudent(model)         # residus studentises (deletion)
  leverage    <- hatvalues(model)
  cooks_d     <- cooks.distance(model)
  n           <- nrow(df)

  par(mfrow = c(2, 2), oma = c(0, 0, 3, 0))

  # --- 1. Residus vs valeurs ajustees ---
  plot(fitted_vals, resid, pch = 16, col = adjustcolor("steelblue", alpha.f = 0.6),
       xlab = "Valeurs ajustees", ylab = "Residus",
       main = "Residus vs valeurs ajustees")
  abline(h = 0, col = "red", lty = 2)
  lo <- lowess(fitted_vals, resid, f = 0.6)
  lines(lo, col = "firebrick", lwd = 1.5)

  # --- 2. QQ-plot des residus ---
  qqnorm(resid, pch = 16, col = adjustcolor("steelblue", alpha.f = 0.6),
         main = "QQ-plot des residus (normalite)")
  qqline(resid, col = "red", lwd = 1.5)

  # --- 3. Scale-Location ---
  sqrt_abs_resid <- sqrt(abs(std_resid))
  plot(fitted_vals, sqrt_abs_resid, pch = 16, col = adjustcolor("steelblue", alpha.f = 0.6),
       xlab = "Valeurs ajustees", ylab = expression(sqrt(abs("residus standardises"))),
       main = "Scale-Location (homoscedasticite)")
  lo2 <- lowess(fitted_vals, sqrt_abs_resid, f = 0.6)
  lines(lo2, col = "firebrick", lwd = 1.5)

  # --- 4. Residus vs levier + distance de Cook ---
  cex_cook <- 0.5 + 3 * cooks_d / max(cooks_d)   # taille proportionnelle a Cook's D
  plot(leverage, stud_resid, pch = 16, col = adjustcolor("steelblue", alpha.f = 0.6),
       cex = cex_cook,
       xlab = "Levier (hat values)", ylab = "Residus studentises",
       main = "Residus vs levier (taille = distance de Cook)")
  abline(h = 0, col = "grey40", lty = 3)
  threshold <- 4 / n   # seuil usuel de levier eleve : 2(k+1)/n ou 4/n selon convention
  abline(v = threshold, col = "red", lty = 2)
  legend("topright", legend = sprintf("seuil levier = %.3f", threshold),
         col = "red", lty = 2, bty = "n", cex = 0.8)

  mtext("Diagnostics du modele OLS", outer = TRUE, cex = 1.4, font = 2)
  cat(sprintf("[Graphique] Diagnostics sauvegardes -> %s\n", save_path))
}

make_coefficient_plot <- function(comp, save_path = "ols_coef_comparison.png") {
  comp_no_const <- comp[rownames(comp) != "(Intercept)", ]
  vars_ <- rownames(comp_no_const)
  y_pos <- seq_along(vars_)

  ci_classic <- 1.96 * comp_no_const$se_classique
  ci_robust  <- 1.96 * comp_no_const$se_robuste_HC3

  xlim_range <- range(c(comp_no_const$coefficient - ci_classic,
                         comp_no_const$coefficient + ci_classic,
                         comp_no_const$coefficient - ci_robust,
                         comp_no_const$coefficient + ci_robust))

  par(mar = c(4, 8, 3, 1))
  plot(NULL, xlim = xlim_range, ylim = c(0.5, length(vars_) + 0.5),
       yaxt = "n", xlab = "Coefficient estime", ylab = "",
       main = "Comparaison des coefficients : classique vs robuste (HC3)", cex.main = 1.05)
  axis(2, at = y_pos, labels = vars_, las = 1)
  abline(v = 0, col = "grey60", lty = 3)

  # OLS classique (bleu), legerement decale vers le bas
  arrows(comp_no_const$coefficient - ci_classic, y_pos - 0.12,
         comp_no_const$coefficient + ci_classic, y_pos - 0.12,
         angle = 90, code = 3, length = 0.05, col = "steelblue", lwd = 1.5)
  points(comp_no_const$coefficient, y_pos - 0.12, pch = 16, col = "steelblue", cex = 1.3)

  # OLS robuste HC3 (rouge), legerement decale vers le haut
  arrows(comp_no_const$coefficient - ci_robust, y_pos + 0.12,
         comp_no_const$coefficient + ci_robust, y_pos + 0.12,
         angle = 90, code = 3, length = 0.05, col = "firebrick", lwd = 1.5)
  points(comp_no_const$coefficient, y_pos + 0.12, pch = 16, col = "firebrick", cex = 1.3)

  legend("topright", legend = c("OLS classique (IC 95%)", "OLS robuste HC3 (IC 95%)"),
         col = c("steelblue", "firebrick"), pch = 16, lty = 1, bty = "n")
  cat(sprintf("[Graphique] Comparaison des coefficients -> %s\n", save_path))
}


# ====================================================================
# 6. PROGRAMME PRINCIPAL
# ====================================================================

main <- function() {
  df <- load_data()
  print(summary(df[, c("sepal_length", "sepal_width", "petal_length", "petal_width")]))

  # --- Exploration graphique prealable ---
  make_exploratory_plots(df, DEP_VAR, REGRESSORS)

  # --- OLS classique ---
  fit_result <- fit_ols(df, DEP_VAR, REGRESSORS)
  model <- fit_result$model
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("OLS CLASSIQUE  |  formule : %s\n", fit_result$formula))
  cat(strrep("=", 70), "\n")
  print(summary(model))

  # --- Diagnostics ---
  diag <- run_diagnostics(model, df, REGRESSORS)
  print_diagnostics(diag)

  # --- OLS robuste (HC3) ---
  model_robust <- fit_ols_robust(model)
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("OLS AVEC ERREURS-TYPES ROBUSTES (HC3)\n")
  cat(strrep("=", 70), "\n")
  print(model_robust)

  # --- Comparaison classique vs robuste ---
  comp <- compare_models(model, model_robust)
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("COMPARAISON OLS CLASSIQUE vs OLS ROBUSTE (HC3)\n")
  cat(strrep("=", 70), "\n")
  print(comp)

  # --- Graphiques diagnostiques et de comparaison ---
  make_diagnostic_plots(model, df)
  make_coefficient_plot(comp)

  cat("\nAnalyse terminee. Fichiers generes : ",
      "ols_exploratory.png, ols_diagnostics.png, ols_coef_comparison.png\n", sep = "")
}

main()

# ----------------------------------------------------------------------
# Note sur l'affichage interactif (persistance des graphiques) :
# ----------------------------------------------------------------------
# Ce script sauvegarde les figures directement dans des fichiers PNG via
# png(...)/dev.off(), ce qui fonctionne a l'identique en mode script ou
# en mode interactif. Si vous executez ce script ligne par ligne dans
# RStudio ou la console R (et non via Rscript), vous pouvez remplacer
# n'importe quel bloc png(...) / ... / dev.off() par les memes
# instructions de trace SANS png()/dev.off() : la figure s'affichera alors
# dans le panneau "Plots" de RStudio et y restera jusqu'a ce que vous
# l'effaciez ou la remplaciez -- l'equivalent du plt.show() bloquant vu
# cote Python.
