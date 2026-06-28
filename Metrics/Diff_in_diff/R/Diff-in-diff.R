#!/usr/bin/env Rscript
# ==============================================================================
# TEMPLATE D'ANALYSE EN DIFFERENCE-IN-DIFFERENCES (DiD) -- VERSION R
# ==============================================================================
# Traduction fonctionnelle du template Python (did_template.py). Reproduit les
# memes variantes de DiD, les memes tests statistiques et les memes graphiques
# (sans generation de rapport PDF).
#
#   1. DiD classique 2x2
#   2. DiD multi-periodes -- TWFE statique (panel a effets fixes individu+temps)
#   3. DiD dynamique / event-study (leads & lags)
#   4. DiD a adoption echelonnee :
#        - decomposition de Goodman-Bacon (diagnostic du TWFE biaise)
#        - estimateur de Callaway & Sant'Anna (ATT(g,t))
#   5. Tests statistiques : tendances paralleles (Wald), placebo, permutation,
#      heteroscedasticite (Breusch-Pagan), bootstrap par cluster
#
# ------------------------------------------------------------------------------
# UTILISATION AVEC VOS PROPRES DONNEES
# ------------------------------------------------------------------------------
# Remplacez load_data() par la lecture de votre fichier (read.csv, etc.), a
# condition que le data.frame final contienne les colonnes :
#   unit_id (identifiant unite), time (periode entiere 1..T), Y (outcome),
#   group (libelle de cohorte / "Jamais_traite"), first_treat (NA si jamais
#   traite), treated (0/1, traitement effectif a la periode `time`)
# ==============================================================================

suppressWarnings(suppressMessages({
  library(dplyr)
  library(ggplot2)
  library(plm)
  library(lmtest)
  library(sandwich)
  library(tidyr)
}))

# ==============================================================================
# 0. CONFIGURATION GENERALE
# ==============================================================================
RNG_SEED <- 5
set.seed(RNG_SEED)

BASE_DIR <- "."
DATA_DIR <- "data"
DATA_PATH <- file.path(DATA_DIR, "panel_data.csv")
OUTPUT_DIR <- "outputs"
FIG_DIR <- file.path(OUTPUT_DIR, "figures")
RESULTS_PATH <- file.path(OUTPUT_DIR, "resultats_did.csv")

dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

ALPHA <- 0.05
EVENT_WINDOW <- 6
N_PERMUTATIONS <- 1000
N_BOOTSTRAP <- 500
NEVER_TREATED_LABEL <- "Jamais_traite"

theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12)))

PALETTE <- c(
  "Jamais_traite" = "#6b7280",
  "Cohorte_2017"  = "#2563eb",
  "Cohorte_2020"  = "#16a34a",
  "Cohorte_2023"  = "#dc2626"
)

# ==============================================================================
# 1. GENERATION / CHARGEMENT DE LA BASE DE DONNEES
# ==============================================================================
generate_synthetic_panel <- function(n_per_group = 150, n_periods = 24,
                                      cohorts = NULL,
                                      violate_parallel_trends = FALSE,
                                      sigma_unit_fe = 3.0, sigma_noise = 2.0) {
  # Simule une base PANEL avec adoption echelonnee du traitement, effets
  # heterogenes ET dynamiques (qui croissent avec la duree d'exposition) :
  # configuration typique dans laquelle le TWFE statique est biaise.
  if (is.null(cohorts)) {
    cohorts <- list(Jamais_traite = NA, Cohorte_2017 = 8,
                     Cohorte_2020 = 14, Cohorte_2023 = 20)
  }

  effect_params <- list(
    Cohorte_2017 = list(base = 1.5, growth = 0.45),
    Cohorte_2020 = list(base = 1.5, growth = 0.22),
    Cohorte_2023 = list(base = 1.5, growth = 0.08)
  )

  # Choc macro commun (effet temps), tire UNE SEULE FOIS par periode et
  # partage par toutes les unites.
  time_fe <- setNames(0.35 * (1:n_periods) + rnorm(n_periods, 0, 0.4), 1:n_periods)

  rows <- list()
  unit_id <- 0
  for (label in names(cohorts)) {
    first_treat <- cohorts[[label]]
    for (u in 1:n_per_group) {
      unit_id <- unit_id + 1
      alpha_i <- rnorm(1, 0, sigma_unit_fe)
      group_level <- rnorm(1, 0, 0.5)

      group_trend_slope <- 0
      if (violate_parallel_trends && label != NEVER_TREATED_LABEL) {
        group_trend_slope <- 0.18
      }

      for (t in 1:n_periods) {
        lambda_t <- time_fe[[as.character(t)]]
        trend_violation <- group_trend_slope * t

        is_treated <- as.integer(!is.na(first_treat) && t >= first_treat)
        tau_it <- 0
        if (is_treated == 1) {
          exposure <- t - first_treat + 1
          p <- effect_params[[label]]
          tau_it <- p$base + p$growth * (exposure - 1)
        }

        eps <- rnorm(1, 0, sigma_noise)
        Y <- alpha_i + lambda_t + group_level + trend_violation + tau_it * is_treated + eps

        rows[[length(rows) + 1]] <- data.frame(
          unit_id = unit_id, time = t, year = 2000 + t, group = label,
          first_treat = ifelse(is.na(first_treat), NA_real_, first_treat),
          treated = is_treated, Y = Y
        )
      }
    }
  }

  df <- bind_rows(rows)
  df$rel_time <- df$time - df$first_treat
  df
}

load_data <- function(force_regenerate = FALSE) {
  if (force_regenerate || !file.exists(DATA_PATH)) {
    df <- generate_synthetic_panel()
    write.csv(df, DATA_PATH, row.names = FALSE)
    cat(sprintf("[load_data] Base synthetique generee et sauvegardee : %s\n", DATA_PATH))
  } else {
    df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
    cat(sprintf("[load_data] Base de donnees chargee depuis : %s\n", DATA_PATH))
  }

  required_cols <- c("unit_id", "time", "Y", "group", "first_treat", "treated")
  missing <- setdiff(required_cols, names(df))
  if (length(missing) > 0) stop(paste("Colonnes manquantes dans la base :", paste(missing, collapse = ", ")))

  df <- df[order(df$unit_id, df$time), ]
  df
}

# ==============================================================================
# 2. STATISTIQUES DESCRIPTIVES & GRAPHIQUE DES TENDANCES BRUTES
# ==============================================================================
plot_raw_trends <- function(df, savepath = file.path(FIG_DIR, "01_tendances_brutes.png")) {
  means <- df %>% group_by(group, time) %>% summarise(Y = mean(Y), .groups = "drop")
  first_treats <- df %>% group_by(group) %>% summarise(first_treat = first(first_treat), .groups = "drop")

  p <- ggplot(means, aes(x = time, y = Y, color = group)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.3) +
    geom_vline(data = first_treats %>% filter(!is.na(first_treat)),
               aes(xintercept = first_treat - 0.5, color = group),
               linetype = "dashed", linewidth = 0.6, alpha = 0.6, show.legend = FALSE) +
    scale_color_manual(values = PALETTE, labels = function(x) gsub("_", " ", x)) +
    labs(title = "Trajectoires moyennes de l'outcome par cohorte de traitement",
         x = "Periode", y = "Y moyen", color = NULL) +
    theme(legend.position = c(0.18, 0.85))

  print(p)
  ggsave(savepath, p, width = 8.5, height = 5, dpi = 140)
  savepath
}

# ==============================================================================
# 3. DiD CLASSIQUE 2x2
# ==============================================================================
did_2x2 <- function(df, treat_group, control_group = NEVER_TREATED_LABEL,
                     pre_period = NULL, post_period = NULL) {
  if (is.null(pre_period)) {
    pre_period <- df$first_treat[df$group == treat_group][1] - 1
  }
  if (is.null(post_period)) {
    post_period <- df$first_treat[df$group == treat_group][1]
  }

  sub <- df %>%
    filter(group %in% c(treat_group, control_group), time %in% c(pre_period, post_period)) %>%
    mutate(treat = as.integer(group == treat_group),
           post = as.integer(time == post_period))

  # --- (a) Calcul "a la main" par moyennes de cellules ---
  cell_means <- sub %>% group_by(treat, post) %>% summarise(m = mean(Y), .groups = "drop")
  gm <- function(tr, po) cell_means$m[cell_means$treat == tr & cell_means$post == po]
  did_manual <- (gm(1, 1) - gm(1, 0)) - (gm(0, 1) - gm(0, 0))

  # --- (b) Estimation par regression, erreurs-types clusterisees par unite ---
  model <- lm(Y ~ treat * post, data = sub)
  vc <- vcovCL(model, cluster = ~unit_id)
  ct <- coeftest(model, vcov. = vc)
  ci <- coefci(model, vcov. = vc, level = 1 - ALPHA)

  coef_name <- "treat:post"
  list(
    methode = "DiD 2x2",
    comparaison = paste(treat_group, "vs", control_group),
    pre_period = pre_period, post_period = post_period,
    estimate = ct[coef_name, "Estimate"],
    se = ct[coef_name, "Std. Error"],
    ci_low = ci[coef_name, 1], ci_high = ci[coef_name, 2],
    pvalue = ct[coef_name, "Pr(>|t|)"],
    n_obs = nrow(sub),
    verif_calcul_manuel = did_manual,
    model = model, vcov = vc, cell_means = cell_means
  )
}

plot_2x2_visual <- function(result, savepath = file.path(FIG_DIR, "02_did_2x2.png")) {
  cm <- result$cell_means
  gm <- function(tr, po) cm$m[cm$treat == tr & cm$post == po]
  pre <- result$pre_period; post <- result$post_period
  contrefactuel <- gm(1, 0) + (gm(0, 1) - gm(0, 0))

  d <- data.frame(
    x = rep(c(pre, post), 3),
    y = c(gm(0, 0), gm(0, 1), gm(1, 0), gm(1, 1), gm(1, 0), contrefactuel),
    serie = rep(c("Groupe de controle (observe)", "Groupe traite (observe)",
                  "Contrefactuel du groupe traite"), each = 2)
  )
  d$serie <- factor(d$serie, levels = c("Groupe de controle (observe)",
                                         "Groupe traite (observe)",
                                         "Contrefactuel du groupe traite"))

  p <- ggplot(d, aes(x = x, y = y, color = serie, linetype = serie)) +
    geom_line(linewidth = 1) + geom_point(size = 2.2) +
    scale_color_manual(values = c("#6b7280", "#2563eb", "#2563eb")) +
    scale_linetype_manual(values = c("solid", "solid", "dashed")) +
    annotate("segment", x = post + 0.05, xend = post + 0.05,
             y = contrefactuel, yend = gm(1, 1),
             arrow = arrow(ends = "both", length = unit(0.1, "inches")), color = "#dc2626") +
    annotate("text", x = post + 0.15, y = (gm(1, 1) + contrefactuel) / 2,
             label = sprintf("Effet DiD\n= %.2f", result$estimate),
             color = "#dc2626", hjust = 0, size = 3.2) +
    scale_x_continuous(breaks = c(pre, post), labels = c("Avant", "Apres"),
                        expand = expansion(mult = c(0.08, 0.32))) +
    labs(title = paste("DiD 2x2 :", result$comparaison), x = NULL, y = "Y moyen", color = NULL, linetype = NULL) +
    theme(legend.position = "top", legend.text = element_text(size = 8))

  print(p)
  ggsave(savepath, p, width = 6.8, height = 5, dpi = 140)
  savepath
}

# ==============================================================================
# 4. DiD MULTI-PERIODES : TWFE STATIQUE
# ==============================================================================
did_twfe_static <- function(df) {
  pdata <- pdata.frame(df, index = c("unit_id", "time"))
  mod <- plm(Y ~ treated, data = pdata, model = "within", effect = "twoways")
  vc <- vcovHC(mod, method = "arellano", type = "HC1", cluster = "group")
  ct <- coeftest(mod, vcov. = vc)
  ci <- coefci(mod, vcov. = vc, level = 1 - ALPHA)

  list(
    methode = "TWFE statique (multi-periodes)",
    estimate = ct["treated", "Estimate"], se = ct["treated", "Std. Error"],
    ci_low = ci["treated", 1], ci_high = ci["treated", 2],
    pvalue = ct["treated", "Pr(>|t|)"], n_obs = nobs(mod),
    model = mod, vcov = vc
  )
}

# ==============================================================================
# 5. DiD DYNAMIQUE / "EVENT-STUDY"
# ==============================================================================
build_event_dummies <- function(df, window = EVENT_WINDOW, ref = -1) {
  out <- df
  cols <- c()
  for (k in (-window):window) {
    if (k == ref) next
    colname <- if (k < 0) paste0("lead_", abs(k)) else paste0("lag_", k)
    if (k == -window) {
      cond <- out$rel_time <= k
    } else if (k == window) {
      cond <- out$rel_time >= k
    } else {
      cond <- out$rel_time == k
    }
    cond[is.na(cond)] <- FALSE
    out[[colname]] <- as.integer(cond)
    cols <- c(cols, colname)
  }
  list(df = out, cols = cols)
}

did_event_study <- function(df, window = EVENT_WINDOW, ref = -1) {
  built <- build_event_dummies(df, window = window, ref = ref)
  out <- built$df; cols <- built$cols

  pdata <- pdata.frame(out, index = c("unit_id", "time"))
  formula <- as.formula(paste("Y ~", paste(cols, collapse = " + ")))
  mod <- plm(formula, data = pdata, model = "within", effect = "twoways")
  vc <- vcovHC(mod, method = "arellano", type = "HC1", cluster = "group")
  ct <- coeftest(mod, vcov. = vc)
  ci <- coefci(mod, vcov. = vc, level = 1 - ALPHA)

  coefs <- list()
  for (k in (-window):window) {
    colname <- if (k < 0) paste0("lead_", abs(k)) else paste0("lag_", k)
    if (k == ref) {
      coefs[[length(coefs) + 1]] <- data.frame(rel_time = k, estimate = 0, se = 0,
                                                 ci_low = 0, ci_high = 0, pvalue = NA)
    } else {
      coefs[[length(coefs) + 1]] <- data.frame(
        rel_time = k, estimate = ct[colname, "Estimate"], se = ct[colname, "Std. Error"],
        ci_low = ci[colname, 1], ci_high = ci[colname, 2], pvalue = ct[colname, "Pr(>|t|)"]
      )
    }
  }
  coefs_df <- bind_rows(coefs) %>% arrange(rel_time)

  list(methode = "Event-study TWFE (dynamique)", coefs = coefs_df, model = mod,
       vcov = vc, dummy_cols = cols, window = window, ref = ref)
}

plot_event_study <- function(coefs_df, title, savepath, color = "#2563eb",
                              overlay = NULL, overlay_label = NULL, overlay_color = "#dc2626") {
  coefs_df$serie <- "Event-study TWFE"
  d <- coefs_df
  if (!is.null(overlay)) {
    overlay$serie <- overlay_label
    d <- bind_rows(coefs_df, overlay)
  }
  cols <- setNames(c(color, overlay_color), c("Event-study TWFE", overlay_label))

  p <- ggplot(d, aes(x = rel_time, y = estimate, color = serie)) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.4) +
    geom_vline(xintercept = -0.5, color = "grey50", linetype = "dashed", linewidth = 0.5) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15, linewidth = 0.6) +
    geom_line(linewidth = 0.7) + geom_point(size = 1.8) +
    scale_color_manual(values = cols) +
    labs(title = title, x = "Temps relatif au traitement (periodes)",
         y = "Effet estime sur Y", color = NULL) +
    theme(legend.position = "top")

  print(p)
  ggsave(savepath, p, width = 8.5, height = 5, dpi = 140)
  savepath
}

# ==============================================================================
# 6. DECOMPOSITION DE GOODMAN-BACON
# ==============================================================================
goodman_bacon_decomposition <- function(df) {
  cohorts <- df %>% group_by(group) %>% summarise(first_treat = first(first_treat), .groups = "drop")
  cohorts$first_treat[is.na(cohorts$first_treat)] <- Inf
  labels <- cohorts$group

  rows <- list()
  for (i in 1:(length(labels) - 1)) {
    for (j in (i + 1):length(labels)) {
      k <- labels[i]; l <- labels[j]
      ft_k <- cohorts$first_treat[cohorts$group == k]
      ft_l <- cohorts$first_treat[cohorts$group == l]

      sub <- df %>% filter(group %in% c(k, l))
      n_kl <- length(unique(sub$unit_id))

      pdata <- pdata.frame(sub, index = c("unit_id", "time"))
      beta_kl <- tryCatch({
        mod <- plm(Y ~ treated, data = pdata, model = "within", effect = "twoways")
        coef(mod)[["treated"]]
      }, error = function(e) NA)
      if (is.na(beta_kl)) next

      # Variance du traitement residualisee des effets fixes (poids)
      d_resid <- sub$treated - ave(sub$treated, sub$unit_id) - ave(sub$treated, sub$time) + mean(sub$treated)
      weight_raw <- n_kl * var(d_resid)

      both_eventually_treated <- is.finite(ft_k) && is.finite(ft_l)
      comparison_type <- if (both_eventually_treated) "Deux groupes traites (comparaison a risque)" else "Traite vs jamais-traite (comparaison propre)"

      rows[[length(rows) + 1]] <- data.frame(
        groupe_1 = k, groupe_2 = l, type_comparaison = comparison_type,
        beta_2x2 = beta_kl, poids_brut = weight_raw, n_obs = n_kl
      )
    }
  }

  bacon_df <- bind_rows(rows)
  bacon_df$poids <- bacon_df$poids_brut / sum(bacon_df$poids_brut)
  bacon_df$poids_brut <- NULL
  bacon_df <- bacon_df %>% arrange(desc(poids))

  beta_reconstructed <- sum(bacon_df$poids * bacon_df$beta_2x2)
  list(bacon_df = bacon_df, beta_reconstructed = beta_reconstructed)
}

plot_bacon <- function(bacon_df, beta_twfe, beta_reconstructed,
                        savepath = file.path(FIG_DIR, "04_goodman_bacon.png")) {
  colors <- c("Traite vs jamais-traite (comparaison propre)" = "#16a34a",
              "Deux groupes traites (comparaison a risque)" = "#dc2626")
  bacon_df$label <- paste0(substr(bacon_df$groupe_1, 1, 8), "/", substr(bacon_df$groupe_2, 1, 8))

  p <- ggplot(bacon_df, aes(x = poids, y = beta_2x2, color = type_comparaison)) +
    geom_hline(yintercept = beta_twfe, linetype = "dashed", color = "black") +
    geom_hline(yintercept = beta_reconstructed, linetype = "dotted", color = "grey40") +
    geom_point(size = 3.2, alpha = 0.9) +
    ggrepel_text(bacon_df) +
    scale_color_manual(values = colors) +
    scale_x_continuous(expand = expansion(mult = c(0.18, 0.18))) +
    scale_y_continuous(expand = expansion(mult = c(0.08, 0.12))) +
    labs(title = "Decomposition de Goodman-Bacon du TWFE statique",
         x = "Poids dans la decomposition", y = "Estimation DiD 2x2", color = NULL,
         caption = sprintf("TWFE statique (estime direct) = %.2f  |  Moyenne ponderee reconstruite = %.2f",
                            beta_twfe, beta_reconstructed)) +
    theme(legend.position = "top", legend.text = element_text(size = 8))

  print(p)
  ggsave(savepath, p, width = 8, height = 5.2, dpi = 140)
  savepath
}

# Petit helper sans dependance a ggrepel (evite une dependance supplementaire) :
# etiquette texte simple legerement decalee au-dessus de chaque point.
ggrepel_text <- function(bacon_df) {
  geom_text(data = bacon_df, aes(label = label), size = 2.5, vjust = -0.8,
            show.legend = FALSE)
}

# ==============================================================================
# 7. ESTIMATEUR DE CALLAWAY & SANT'ANNA (ATT(g,t)) -- staggered adoption
# ==============================================================================
att_gt <- function(df, g, t, control_group = NEVER_TREATED_LABEL) {
  cohort_label <- df$group[df$first_treat == g & !is.na(df$first_treat)][1]
  anchor <- g - 1
  sub <- df %>%
    filter(group %in% c(cohort_label, control_group), time %in% c(anchor, t)) %>%
    mutate(treat = as.integer(group == cohort_label),
           post = as.integer(time == t))

  model <- lm(Y ~ treat * post, data = sub)
  vc <- vcovCL(model, cluster = ~unit_id)
  ct <- coeftest(model, vcov. = vc)
  n_g <- length(unique(df$unit_id[df$group == cohort_label]))

  data.frame(g = g, t = t, rel_time = t - g, cohort = cohort_label,
             estimate = ct["treat:post", "Estimate"], se = ct["treat:post", "Std. Error"],
             pvalue = ct["treat:post", "Pr(>|t|)"], n_cohort = n_g)
}

callaway_santanna <- function(df, control_group = NEVER_TREATED_LABEL) {
  cohorts_g <- sort(unique(df$first_treat[df$group != control_group & !is.na(df$first_treat)]))
  times <- sort(unique(df$time))

  all_results <- list()
  for (g in cohorts_g) {
    g <- as.integer(g)
    for (t in times) {
      if (t == g - 1) next
      all_results[[length(all_results) + 1]] <- att_gt(df, g, t, control_group = control_group)
    }
  }
  att_df <- bind_rows(all_results)

  # --- Agregation simple (ATT global post-traitement) ---
  post <- att_df %>% filter(rel_time >= 0)
  weights <- post %>% distinct(cohort, n_cohort)
  post <- post %>% left_join(weights, by = c("cohort", "n_cohort"))
  att_simple <- sum(post$estimate * post$n_cohort) / sum(post$n_cohort)
  var_simple <- sum(((post$n_cohort / sum(post$n_cohort))^2) * post$se^2)
  se_simple <- sqrt(var_simple)

  # --- Agregation dynamique (event-study robuste par temps relatif e) ---
  dyn <- att_df %>%
    group_by(rel_time) %>%
    summarise(
      estimate = sum(estimate * n_cohort) / sum(n_cohort),
      se = sqrt(sum(((n_cohort / sum(n_cohort))^2) * se^2)),
      n_cohorts = n_distinct(cohort),
      .groups = "drop"
    ) %>%
    mutate(ci_low = estimate - qnorm(1 - ALPHA / 2) * se,
           ci_high = estimate + qnorm(1 - ALPHA / 2) * se) %>%
    arrange(rel_time)

  list(att_gt = att_df, att_simple = att_simple, se_simple = se_simple, event_study = dyn)
}

# ==============================================================================
# 9. GRAPHIQUES COMPLEMENTAIRES (CS vs TWFE, heatmap, distributions)
# ==============================================================================
plot_cs_vs_twfe <- function(cs_result, event_result,
                             savepath = file.path(FIG_DIR, "05_callaway_santanna_vs_twfe.png")) {
  plot_event_study(
    event_result$coefs,
    title = "Event-study : TWFE naif vs Callaway & Sant'Anna (robuste)",
    savepath = savepath, color = "#9ca3af",
    overlay = cs_result$event_study, overlay_label = "Callaway & Sant'Anna (ATT(e) agrege)",
    overlay_color = "#16a34a"
  )
}

plot_distribution <- function(values, observed, title, xlabel, savepath, color = "#2563eb") {
  d <- data.frame(values = values)
  xmax <- max(c(values, observed)) * 1.08
  xmin <- min(c(values, observed)) * 1.08

  p <- ggplot(d, aes(x = values)) +
    geom_histogram(bins = 40, fill = color, alpha = 0.55, color = "white") +
    geom_vline(xintercept = observed, color = "#dc2626", linewidth = 1, linetype = "solid") +
    geom_vline(xintercept = 0, color = "black", linewidth = 0.5, linetype = "dashed", alpha = 0.6) +
    coord_cartesian(xlim = c(xmin, xmax)) +
    labs(title = title, x = xlabel, y = "Frequence",
         caption = sprintf("Ligne rouge : estimation observee = %.2f", observed)) +
    theme(plot.caption = element_text(color = "#dc2626", size = 9, hjust = 0.5))

  print(p)
  ggsave(savepath, p, width = 7.5, height = 4.5, dpi = 140)
  savepath
}

plot_att_gt_heatmap <- function(att_df, savepath = file.path(FIG_DIR, "08_att_gt_heatmap.png")) {
  p <- ggplot(att_df, aes(x = t, y = cohort, fill = estimate)) +
    geom_tile() +
    scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b", midpoint = 0,
                          name = "ATT(g,t)") +
    labs(title = "ATT(g,t) de Callaway & Sant'Anna par cohorte et periode\n(zones avant la date de traitement = test de pre-tendance)",
         x = "Periode (t)", y = NULL) +
    theme(axis.text.x = element_text(angle = 90, size = 6, vjust = 0.5))

  print(p)
  ggsave(savepath, p, width = 11, height = 3.4, dpi = 140)
  savepath
}

# ==============================================================================
# 8. TESTS STATISTIQUES
# ==============================================================================
wald_test_restrictions <- function(coefs, vcov_mat, names_to_test) {
  # Test de Wald manuel H0 : tous les coefficients de `names_to_test` = 0.
  b <- coefs[names_to_test]
  V <- vcov_mat[names_to_test, names_to_test]
  stat <- as.numeric(t(b) %*% solve(V) %*% b)
  df <- length(names_to_test)
  pval <- 1 - pchisq(stat, df)
  list(statistic = stat, df = df, pvalue = pval)
}

test_parallel_trends <- function(event_result) {
  lead_cols <- event_result$dummy_cols[grepl("^lead_", event_result$dummy_cols)]
  b <- coef(event_result$model)
  w <- wald_test_restrictions(b, event_result$vcov, lead_cols)

  conclusion <- if (w$pvalue > ALPHA) {
    sprintf("Non-rejet de H0 : pas de signe de violation des tendances paralleles au seuil de %.0f%%", ALPHA * 100)
  } else {
    "Rejet de H0 : signe potentiel de violation des tendances paralleles (a interpreter avec prudence, cf. biais de contamination des effets fixes temps en cas d'adoption echelonnee)"
  }
  list(test = "Test conjoint de tendances paralleles (Wald)",
       h0 = "Tous les coefficients 'leads' pre-traitement sont nuls",
       statistic = w$statistic, df = w$df, pvalue = w$pvalue, conclusion = conclusion)
}

test_placebo <- function(df, treat_group, control_group = NEVER_TREATED_LABEL, fake_lag = 4) {
  true_first_treat <- df$first_treat[df$group == treat_group][1]
  fake_post <- true_first_treat - fake_lag
  fake_pre <- fake_post - 1
  if (fake_pre < min(df$time)) stop("fake_lag trop grand : periode placebo hors panel")

  res <- did_2x2(df, treat_group = treat_group, control_group = control_group,
                  pre_period = fake_pre, post_period = fake_post)
  conclusion <- if (res$pvalue > ALPHA) {
    "Non-rejet de H0 : aucun effet placebo detecte"
  } else {
    "Rejet de H0 : effet placebo significatif, possible violation des tendances paralleles ou anticipation du traitement"
  }
  list(test = "Test de placebo (fausse date de traitement pre-periode)",
       h0 = "Effet placebo nul (pas d'anticipation / tendances paralleles)",
       fake_pre_period = fake_pre, fake_post_period = fake_post,
       estimate = res$estimate, se = res$se, pvalue = res$pvalue, conclusion = conclusion)
}

test_permutation <- function(df, treat_group, control_group = NEVER_TREATED_LABEL,
                              n_perm = N_PERMUTATIONS, seed = 123) {
  set.seed(seed)
  sub <- df %>% filter(group %in% c(treat_group, control_group))
  real_result <- did_2x2(df, treat_group = treat_group, control_group = control_group)
  real_estimate <- real_result$estimate
  pre <- real_result$pre_period; post <- real_result$post_period

  unit_ids <- unique(sub$unit_id)
  n_treated <- length(unique(sub$unit_id[sub$group == treat_group]))
  sub2p <- sub %>% filter(time %in% c(pre, post))

  null_estimates <- numeric(n_perm)
  for (b in 1:n_perm) {
    fake_treated_units <- sample(unit_ids, size = n_treated, replace = FALSE)
    tmp <- sub2p
    tmp$treat <- as.integer(tmp$unit_id %in% fake_treated_units)
    tmp$post <- as.integer(tmp$time == post)
    m <- lm(Y ~ treat * post, data = tmp)
    null_estimates[b] <- coef(m)[["treat:post"]]
  }

  p_perm <- mean(abs(null_estimates) >= abs(real_estimate))
  conclusion <- if (p_perm > ALPHA) {
    "Non-rejet de H0"
  } else {
    "Rejet de H0 : l'estimation reelle est extreme par rapport a la distribution nulle de permutation"
  }
  list(test = "Test de permutation (inference par randomisation)", h0 = "Effet de traitement nul",
       real_estimate = real_estimate, n_perm = n_perm, pvalue_permutation = p_perm,
       null_distribution = null_estimates, conclusion = conclusion)
}

test_heteroskedasticity <- function(model_2x2) {
  bp <- bptest(model_2x2$model)
  conclusion <- if (bp$p.value > ALPHA) {
    "Non-rejet de H0 : pas d'heteroscedasticite detectee"
  } else {
    "Rejet de H0 : heteroscedasticite detectee -> l'usage d'erreurs-types robustes/clusterisees est justifie (deja applique par defaut dans ce template)"
  }
  list(test = "Test de Breusch-Pagan (heteroscedasticite)", h0 = "Homoscedasticite des residus",
       statistic = as.numeric(bp$statistic), pvalue = bp$p.value, conclusion = conclusion)
}

cluster_bootstrap_se <- function(df, treat_group, control_group = NEVER_TREATED_LABEL,
                                  n_boot = N_BOOTSTRAP, seed = 321) {
  set.seed(seed)
  res0 <- did_2x2(df, treat_group = treat_group, control_group = control_group)
  pre <- res0$pre_period; post <- res0$post_period
  sub <- df %>% filter(group %in% c(treat_group, control_group), time %in% c(pre, post))
  units <- unique(sub$unit_id)

  boot_estimates <- numeric(n_boot)
  for (b in 1:n_boot) {
    sampled_units <- sample(units, size = length(units), replace = TRUE)
    frames <- vector("list", length(sampled_units))
    for (new_id in seq_along(sampled_units)) {
      tmp <- sub[sub$unit_id == sampled_units[new_id], ]
      tmp$unit_id <- new_id
      frames[[new_id]] <- tmp
    }
    boot_sample <- bind_rows(frames)
    boot_sample$treat <- as.integer(boot_sample$group == treat_group)
    boot_sample$post <- as.integer(boot_sample$time == post)
    m <- lm(Y ~ treat * post, data = boot_sample)
    boot_estimates[b] <- coef(m)[["treat:post"]]
  }

  se_boot <- sd(boot_estimates)
  ci <- quantile(boot_estimates, probs = c(ALPHA / 2, 1 - ALPHA / 2))
  list(test = "Bootstrap par cluster (unite) des erreurs-types", estimate = res0$estimate,
       se_cluster_asymptotique = res0$se, se_bootstrap = se_boot,
       ci_bootstrap_low = ci[1], ci_bootstrap_high = ci[2], n_boot = n_boot,
       boot_distribution = boot_estimates)
}

# ==============================================================================
# 10. PIPELINE PRINCIPALE
# ==============================================================================
run_full_analysis <- function(treat_group_2x2 = "Cohorte_2017") {
  cat(strrep("=", 80), "\n", "ETAPE 1/9 : chargement de la base de donnees\n", strrep("=", 80), "\n", sep = "")
  data <- load_data()

  cat("\n", strrep("=", 80), "\n", "ETAPE 2/9 : statistiques descriptives & tendances brutes\n", strrep("=", 80), "\n", sep = "")
  fig_trends <- plot_raw_trends(data)
  cat(sprintf("  -> figure : %s\n", fig_trends))

  cat("\n", strrep("=", 80), "\n", sprintf("ETAPE 3/9 : DiD classique 2x2 (%s vs %s)\n", treat_group_2x2, NEVER_TREATED_LABEL), strrep("=", 80), "\n", sep = "")
  res_2x2 <- did_2x2(data, treat_group = treat_group_2x2)
  fig_2x2 <- plot_2x2_visual(res_2x2)
  cat(sprintf("  Estimation : %.3f (se=%.3f, IC95%%=[%.3f, %.3f], p=%.4f)\n",
              res_2x2$estimate, res_2x2$se, res_2x2$ci_low, res_2x2$ci_high, res_2x2$pvalue))
  cat(sprintf("  -> figure : %s\n", fig_2x2))

  cat("\n", strrep("=", 80), "\n", "ETAPE 4/9 : DiD multi-periodes -- TWFE statique\n", strrep("=", 80), "\n", sep = "")
  res_twfe <- did_twfe_static(data)
  cat(sprintf("  Estimation : %.3f (se=%.3f, p=%.4f)\n", res_twfe$estimate, res_twfe$se, res_twfe$pvalue))

  cat("\n", strrep("=", 80), "\n", "ETAPE 5/9 : DiD dynamique -- event-study TWFE\n", strrep("=", 80), "\n", sep = "")
  res_event <- did_event_study(data)
  fig_event <- plot_event_study(res_event$coefs, title = "Event-study TWFE (toutes cohortes poolees)",
                                  savepath = file.path(FIG_DIR, "03_event_study_twfe.png"))
  cat(sprintf("  -> figure : %s\n", fig_event))

  cat("\n", strrep("=", 80), "\n", "ETAPE 6/9 : decomposition de Goodman-Bacon\n", strrep("=", 80), "\n", sep = "")
  bacon <- goodman_bacon_decomposition(data)
  fig_bacon <- plot_bacon(bacon$bacon_df, res_twfe$estimate, bacon$beta_reconstructed)
  print(bacon$bacon_df)
  cat(sprintf("  TWFE direct = %.3f | Reconstruction ponderee = %.3f\n", res_twfe$estimate, bacon$beta_reconstructed))
  cat(sprintf("  -> figure : %s\n", fig_bacon))

  cat("\n", strrep("=", 80), "\n", "ETAPE 7/9 : estimateur de Callaway & Sant'Anna (ATT(g,t))\n", strrep("=", 80), "\n", sep = "")
  cs_result <- callaway_santanna(data)
  cat(sprintf("  ATT global (agregation simple) = %.3f (se=%.3f)\n", cs_result$att_simple, cs_result$se_simple))
  fig_cs_vs_twfe <- plot_cs_vs_twfe(cs_result, res_event)
  fig_att_heatmap <- plot_att_gt_heatmap(cs_result$att_gt)
  cat(sprintf("  -> figures : %s, %s\n", fig_cs_vs_twfe, fig_att_heatmap))

  cat("\n", strrep("=", 80), "\n", "ETAPE 8/9 : tests statistiques\n", strrep("=", 80), "\n", sep = "")
  t_parallel <- test_parallel_trends(res_event)
  cat(sprintf("  [Tendances paralleles] stat=%.3f df=%d p=%.4f -> %s\n",
              t_parallel$statistic, t_parallel$df, t_parallel$pvalue, t_parallel$conclusion))

  t_placebo <- test_placebo(data, treat_group = treat_group_2x2)
  cat(sprintf("  [Placebo] estimate=%.3f p=%.4f -> %s\n", t_placebo$estimate, t_placebo$pvalue, t_placebo$conclusion))

  t_perm <- test_permutation(data, treat_group = treat_group_2x2)
  fig_perm <- plot_distribution(t_perm$null_distribution, t_perm$real_estimate,
                                  title = "Test de permutation : distribution nulle vs estimation reelle",
                                  xlabel = "Estimation DiD 2x2 sous permutations aleatoires",
                                  savepath = file.path(FIG_DIR, "06_permutation.png"))
  cat(sprintf("  [Permutation] p=%.4f -> %s\n", t_perm$pvalue_permutation, t_perm$conclusion))
  cat(sprintf("  -> figure : %s\n", fig_perm))

  t_hetero <- test_heteroskedasticity(res_2x2)
  cat(sprintf("  [Breusch-Pagan] stat=%.3f p=%.4f -> %s\n", t_hetero$statistic, t_hetero$pvalue, t_hetero$conclusion))

  t_boot <- cluster_bootstrap_se(data, treat_group = treat_group_2x2)
  fig_boot <- plot_distribution(t_boot$boot_distribution, t_boot$estimate,
                                  title = "Bootstrap par cluster (unite) de l'estimateur DiD 2x2",
                                  xlabel = "Estimation DiD 2x2 (replications bootstrap)",
                                  savepath = file.path(FIG_DIR, "07_bootstrap.png"), color = "#16a34a")
  cat(sprintf("  [Bootstrap cluster] se_asymptotique=%.3f se_bootstrap=%.3f\n",
              t_boot$se_cluster_asymptotique, t_boot$se_bootstrap))
  cat(sprintf("  -> figure : %s\n", fig_boot))

  cat("\n", strrep("=", 80), "\n", "ETAPE 9/9 : export du recapitulatif chiffre\n", strrep("=", 80), "\n", sep = "")
  summary_df <- data.frame(
    methode = c("DiD 2x2", "TWFE statique (multi-periodes, multi-cohortes)",
                "Callaway & Sant'Anna (ATT global agrege)"),
    comparaison = c(res_2x2$comparaison, "toutes cohortes", "toutes cohortes vs jamais-traite"),
    estimation = c(res_2x2$estimate, res_twfe$estimate, cs_result$att_simple),
    se = c(res_2x2$se, res_twfe$se, cs_result$se_simple),
    ic_bas = c(res_2x2$ci_low, res_twfe$ci_low, cs_result$att_simple - 1.96 * cs_result$se_simple),
    ic_haut = c(res_2x2$ci_high, res_twfe$ci_high, cs_result$att_simple + 1.96 * cs_result$se_simple),
    p_value = c(res_2x2$pvalue, res_twfe$pvalue, NA)
  )
  write.csv(summary_df, RESULTS_PATH, row.names = FALSE)
  cat(sprintf("  -> recapitulatif sauvegarde : %s\n", RESULTS_PATH))

  list(data = data, res_2x2 = res_2x2, res_twfe = res_twfe, res_event = res_event,
       bacon_df = bacon$bacon_df, beta_reconstructed = bacon$beta_reconstructed,
       cs_result = cs_result, t_parallel = t_parallel, t_placebo = t_placebo,
       t_perm = t_perm, t_hetero = t_hetero, t_boot = t_boot, summary_df = summary_df,
       figures = list(trends = fig_trends, did_2x2 = fig_2x2, event_study = fig_event,
                       bacon = fig_bacon, cs_vs_twfe = fig_cs_vs_twfe,
                       att_heatmap = fig_att_heatmap, permutation = fig_perm, bootstrap = fig_boot))
}

# ==============================================================================
# EXECUTION
# ==============================================================================
# Ce bloc s'execute TOUJOURS, que vous lanciez le script via :
#   - un terminal :          Rscript did_template.R
#   - la console R :         source("did_template.R")
#   - RStudio :               bouton "Source"
# Les graphiques sont a la fois affiches (Plots pane / fenetre graphique) ET
# sauvegardes en PNG dans outputs/figures/.
results <- run_full_analysis()
cat("\nAnalyse terminee. Resultats disponibles dans l'objet `results`.\n")
