# ===== Bloc 1 : chargement des librairies =====
# numpy/pandas -> base R (pas besoin de librairie spécifique pour l'algèbre matricielle)
# matplotlib/seaborn -> ggplot2 (graphiques) + reshape2 (pour le heatmap)
# scipy -> pas utilisé directement dans ce script, donc pas d'équivalent nécessaire
library(ggplot2)
library(reshape2)   # pour transformer la matrice de corrélation en format long (utile pour ggplot)

# ===== Bloc 2 : construction du chemin et lecture du fichier =====
# dirname(dirname(...)) en Python remonte de deux niveaux de dossier
# En R, on utilise dirname() de la même façon
path <- dirname(dirname(getwd()))  
# NB : en Python, __file__ donne le chemin du script. En R il n'y a pas d'équivalent direct
# quand on exécute en interactif ; getwd() est l'approximation la plus proche.
df <- read.csv(file.path(path, "file_name"))

# ===== Bloc 3 : séparation features / cible =====
# En Python : A = df[:, col:col]  (sélection de colonnes), y = cible
# En R, un data.frame ne se subset pas avec [ , ] comme une matrice numpy,
# mais la syntaxe est analogue : df[ , col:col] sélectionne des colonnes par indice
A <- as.matrix(df[, col:col])      # on convertit en matrice pour faire de l'algèbre linéaire
n <- nrow(A)
p <- ncol(A)
y <- as.matrix(df[, col])          # la variable cible (colonne unique)

# ===== Bloc 4 : centrage-réduction des données =====
# np.mean(axis=0) / np.std(axis=0) -> colMeans() / apply(..., sd) en R
mA <- colMeans(A)
sA <- apply(A, 2, sd)
# On centre-réduit chaque colonne : (A - mA) / sA
# sweep() permet d'appliquer une opération colonne par colonne comme le faisait
# la diffusion ("broadcasting") automatique de numpy
A <- sweep(A, 2, mA, "-")
A <- sweep(A, 2, sA, "/")

# ===== Bloc 5 : matrice de corrélation (calculée à la main) =====
# t(A) %*% A est l'équivalent de A.transpose().dot(A) en numpy
# %*% est l'opérateur de produit matriciel en R (équivalent de .dot() ou @ en Python)
C <- t(A) %*% A

# ===== Bloc 6 : heatmap de la matrice de corrélation =====
# seaborn.heatmap n'a pas d'équivalent direct en base R/ggplot2,
# on reconstruit donc le même résultat "à la main" avec ggplot2 :
C_df <- melt(C)   # transforme la matrice en format long (x, y, value) pour ggplot
ggplot(C_df, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +  # équivalent de annot=True, fmt='.2f'
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +  # cmap='coolwarm'
  ggtitle("title") +
  theme_minimal()

ggsave(file.path(path, "file/name.png"))  # équivalent de plt.savefig()

# ===== Bloc 7 : bar chart de la corrélation features vs cible =====
# u = A.transpose().dot(y) / n
u <- (t(A) %*% y) / n
barplot(as.vector(u), names.arg = 1:p,
        xlab = "Feature index", ylab = "Correlation with target")
grid()   # équivalent de plt.grid()

# ===== Bloc 8 : (commenté) stats par catégorie =====
# Équivalent R de la partie Python commentée, au cas où vous voudriez l'utiliser :
# df$AgeGroup <- cut(df$Age, breaks = c(0, 12, 18, 35, 60, 100),
#                     labels = c("Child", "Teen", "Young Adult", "Adult", "Senior"))
#
# for (j in c("Pclass", "Sex", "Embarked", "AgeGroup")) {
#   cat("\n--- Survival rate by", j, "---\n")
#   print(tapply(df$Survived, df[[j]], mean) * 100)
# }

# ===== Bloc 9 : valeurs propres et vecteurs propres de C =====
# np.linalg.eigh -> eigen() en R (eigen() gère nativement les matrices symétriques
# et retourne déjà les valeurs propres triées par ordre décroissant, donc pas besoin
# de refaire le tri manuellement comme en Python)
eig <- eigen(C, symmetric = TRUE)
eigenvalues  <- eig$values
eigenvectors <- eig$vectors

# Affichage (équivalent de print() en Python)
cat("Eigenvalues:\n")
print(eigenvalues)
cat("\nEigenvectors (columns):\n")
print(eigenvectors)

# ===== Bloc 10 : graphique des valeurs propres décroissantes =====
plot(1:length(eigenvalues), eigenvalues, type = "b", pch = 19,
     xlab = "Index", ylab = "Eigenvalue",
     main = "Eigenvalues of the covariance matrix (descending order)")
grid()

# ===== Bloc 11 : projection de A sur les 2 premiers axes de l'ACP =====
# A @ eigenvectors[:, :2]  ->  %*% en R, et [, 1:2] pour les 2 premières colonnes
A_projected <- A %*% eigenvectors[, 1:2]
plot(A_projected[, 1], A_projected[, 2], pch = ".",
     xlab = "PC1", ylab = "PC2")
grid()