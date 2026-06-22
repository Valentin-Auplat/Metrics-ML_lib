# ===== Bloc 1 : librairies =====
# pandas/numpy -> base R
# matplotlib -> ggplot2 (graphiques 2D)
# sklearn.preprocessing.StandardScaler -> scale() en base R
# sklearn.decomposition.PCA -> prcomp() en base R (équivalent direct et standard)
# mpl_toolkits.mplot3d -> scatterplot3d ou rgl pour les graphiques 3D statiques
# plotly.express -> plotly (package R officiel, même famille, même moteur JS)
library(ggplot2)
library(scatterplot3d)   # pour l'équivalent du plot 3D matplotlib
library(plotly)          # pour l'équivalent de plotly.express

# ===== Bloc 2 : gestion du chemin =====
# sys.path.append(...) n'a pas d'équivalent nécessaire en R : R ne fonctionne pas avec un
# système d'imports basé sur des chemins de modules comme Python. On l'omet donc.
path <- dirname(dirname(getwd()))
# NB : comme précédemment, getwd() est l'approximation la plus proche de __file__,
# qui n'existe pas nativement en R en mode interactif.

# ===== Bloc 3 : chargement et preprocessing =====
df <- read.csv(file.path(path, "data"))
# df.drop(columns=["features"]) -> on retire la colonne par son nom avec une sélection négative
X <- df[, !(names(df) %in% c("features"))]
y <- df[["target"]]

# ===== Bloc 4 : standardisation =====
# StandardScaler().fit_transform(X) -> scale() en R fait exactement la même chose :
# centre (soustrait la moyenne) et réduit (divise par l'écart-type) chaque colonne
X_scaled <- scale(X)
# scale() retourne une matrice avec des attributs "center" et "scale" attachés
# (équivalent de scaler.mean_ et scaler.scale_ en Python), au cas où on en a besoin :
scaler_center <- attr(X_scaled, "center")
scaler_scale  <- attr(X_scaled, "scale")

# ===== Bloc 5 : PCA complète =====
# PCA().fit_transform(X_scaled) -> prcomp(X_scaled) en R
# center = FALSE et scale. = FALSE car on a déjà standardisé nous-mêmes au bloc précédent
# (sinon prcomp() centrerait/réduirait une deuxième fois)
pca_full <- prcomp(X_scaled, center = FALSE, scale. = FALSE)

# X_pca_full = scores des individus sur les composantes (équivalent de pca.fit_transform())
X_pca_full <- as.data.frame(pca_full$x)
# Renommage des colonnes en PC1, PC2, PC3... (déjà fait par défaut par prcomp,
# mais on le force explicitement pour être rigoureusement fidèle au code Python)
colnames(X_pca_full) <- paste0("PC", seq_len(ncol(X_pca_full)))

# ===== Bloc 6 : variance expliquée par composante =====
# pca.explained_variance_ -> les valeurs propres = (écart-type des scores)^2
valeur_propre <- pca_full$sdev^2
# pca.explained_variance_ratio_ -> valeur propre / somme des valeurs propres
variance_expliquee <- valeur_propre / sum(valeur_propre)
# np.cumsum() -> cumsum() en R (même nom, même comportement)
variance_cumulee <- cumsum(variance_expliquee)

variance_df <- data.frame(
  Composante         = paste0("PC", seq_along(valeur_propre)),
  Valeur_propre       = valeur_propre,
  Variance_expliquee  = variance_expliquee,
  Variance_cumulee    = variance_cumulee
)

cat("=== Variance expliquée ===\n")
print(variance_df)
write.csv(variance_df,
          file.path(path, "data/PCA_v1/pca_variance_expliquée_v1.csv"),
          row.names = FALSE)

# ===== Bloc 7 : loadings (contributions des features) =====
# pca.components_.T -> en R, pca_full$rotation contient déjà les loadings
# avec les features en lignes et les composantes en colonnes (pas besoin de transposer,
# contrairement à sklearn qui les stocke dans l'autre sens)
loadings_df <- as.data.frame(pca_full$rotation)
colnames(loadings_df) <- paste0("PC", seq_len(ncol(loadings_df)))
rownames(loadings_df) <- colnames(X)

cat("\n=== Loadings (contributions des features) ===\n")
print(round(loadings_df, 3))
write.csv(loadings_df, file.path(path, "data/PCA_v1/pca_loadings_v1.csv"))

# ===== Bloc 8 : données projetées =====
projected_df <- X_pca_full
projected_df$target <- y

cat("\n=== Données projetées (5 premières lignes) ===\n")
print(head(projected_df))
write.csv(projected_df,
          file.path(path, "data/PCA_v1/pca_projected_v1.csv"),
          row.names = FALSE)

# ===== Bloc 9 : scree plot + variance cumulée (2 graphiques côte à côte) =====
# plt.subplots(1, 2, ...) -> on combine deux ggplot avec gridExtra (ou patchwork),
# gridExtra est la solution la plus simple et la plus proche de subplots de matplotlib
library(gridExtra)

scree_df <- data.frame(Composante = seq_along(variance_expliquee),
                       Variance   = variance_expliquee)

p1 <- ggplot(scree_df, aes(x = Composante, y = Variance)) +
  geom_col() +
  xlab("Composante") + ylab("Variance expliquée") +
  ggtitle("Scree plot")

cum_df <- data.frame(Composante = seq_along(variance_cumulee),
                     Cumulee    = variance_cumulee)

p2 <- ggplot(cum_df, aes(x = Composante, y = Cumulee)) +
  geom_point() + geom_line() +
  geom_hline(yintercept = 0.95, color = "red", linetype = "dashed") +  # axhline 95%
  xlab("Nombre de composantes") + ylab("Variance cumulée") +
  ggtitle("Variance cumulée")

combined_plot <- grid.arrange(p1, p2, ncol = 2)
ggsave(file.path(path, "figures/PCA_v1/pca_variance_v1.png"),
       plot = combined_plot, width = 14, height = 5)

# ===== Bloc 10 : projection 2D colorée par la cible (PC1 vs PC2) =====
# reset_index(drop=True) n'a pas vraiment d'équivalent nécessaire en R car les data.frames
# R n'ont pas d'index "actif" comme pandas qui pourrait désaligner les jointures ;
# on s'assure simplement que X_pca_full et y ont le même nombre de lignes dans le même ordre
X_pca_full2 <- X_pca_full
y_reset <- y

plot_df_2d <- data.frame(PC1 = X_pca_full2$PC1,
                         PC2 = X_pca_full2$PC2,
                         target = factor(y_reset, labels = c("Décédés", "Survivants")))

var1_pct <- round(variance_expliquee[1] * 100, 1)
var2_pct <- round(variance_expliquee[2] * 100, 1)

p_2d <- ggplot(plot_df_2d, aes(x = PC1, y = PC2, color = target)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("Décédés" = "red", "Survivants" = "green")) +
  xlab(paste0("PC1 (", var1_pct, "% var)")) +
  ylab(paste0("PC2 (", var2_pct, "% var)")) +
  ggtitle("Projection sur les 2 premières composantes principales")

ggsave(file.path(path, "figures/PCA_v1/pca_projection_2d_v1.png"), plot = p_2d,
       width = 10, height = 7)

# ===== Bloc 11 : projection 2D (PC1 vs PC3) =====
var3_pct <- round(variance_expliquee[3] * 100, 1)

plot_df_pc13 <- data.frame(PC1 = X_pca_full2$PC1,
                           PC3 = X_pca_full2$PC3,
                           target = factor(y_reset, labels = c("Décédés", "Survivants")))

p_2d_pc13 <- ggplot(plot_df_pc13, aes(x = PC1, y = PC3, color = target)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("Décédés" = "red", "Survivants" = "green")) +
  xlab(paste0("PC1 (", var1_pct, "% var)")) +
  ylab(paste0("PC3 (", var3_pct, "% var)")) +
  ggtitle("Projection sur les composantes principales 1 et 3")

ggsave(file.path(path, "figures/PCA_v1/pca_projection_2d_PC1_PC3_v1.png"),
       plot = p_2d_pc13, width = 10, height = 7)

cat("\n=== Fichiers générés ===\n")
cat("CSV: pca_variance_Master.csv, pca_loadings_Master.csv, pca_projected_Master.csv\n")
cat("Figures: pca_variance_Master.png, pca_projection_2d_Master.png, pca_projection_2d_PC1_PC3_Master.png\n")


# ===== Bloc 12 : plot 3D statique =====
# mpl_toolkits.mplot3d Axes3D -> scatterplot3d::scatterplot3d() en R
# (package conçu spécifiquement pour reproduire ce type de scatter 3D statique)

# On sépare les deux groupes comme en Python, car scatterplot3d gère mal
# les groupes par couleur dans un seul appel si on veut une vraie légende
couleurs <- ifelse(y_reset == 0, "red", "green")

png(file.path(path, "figures/PCA_v1/pca_projection_3d_v1.png"),
    width = 12, height = 9, units = "in", res = 150)

s3d <- scatterplot3d(
  x = X_pca_full2$PC1,
  y = X_pca_full2$PC3,
  z = X_pca_full2$PC2,   # PC2 sur l'axe z, comme dans le code Python
  color = couleurs,
  pch = 19,
  xlab = paste0("PC1 (", var1_pct, "%) — solitude/famille"),
  ylab = paste0("PC3 (", var3_pct, "%) — genre"),
  zlab = paste0("PC2 (", round(variance_expliquee[2]*100,1), "%) — classe"),
  main = "Projection 3D — PC1, PC2, PC3"
)
legend("topright", legend = c("Décédés", "Survivants"),
       col = c("red", "green"), pch = 19)

dev.off()

# ===== Bloc 13 : plot 3D interactif (plotly) =====
# plotly.express px.scatter_3d -> plotly::plot_ly() en R, même moteur sous-jacent
plot_df_3d <- data.frame(
  PC1 = X_pca_full2$PC1,
  PC2 = X_pca_full2$PC2,
  PC3 = X_pca_full2$PC3,
  Survived = factor(y_reset, labels = c("Décédé", "Survivant"))
)

fig_3d <- plot_ly(
  plot_df_3d,
  x = ~PC1, y = ~PC3, z = ~PC2,
  color = ~Survived,
  colors = c("Décédé" = "red", "Survivant" = "green"),
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 4, opacity = 0.6)
) %>%
  layout(
    title = "Projection PCA 3D — Titanic",
    scene = list(
      xaxis = list(title = paste0("PC1 (", var1_pct, "%) — solitude/famille")),
      yaxis = list(title = paste0("PC3 (", var3_pct, "%) — genre")),
      zaxis = list(title = paste0("PC2 (", round(variance_expliquee[2]*100,1), "%) — classe"))
    )
  )

# fig.write_html(...) -> htmlwidgets::saveWidget() en R
htmlwidgets::saveWidget(fig_3d,
                        file.path(path, "figures/PCA_v1/pca_projection_3d_v1.html"))

# fig.show() -> print(fig_3d) ouvre le widget dans le viewer RStudio
# print(fig_3d)