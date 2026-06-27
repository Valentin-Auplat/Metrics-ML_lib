# ============================================================
# 1. CHARGEMENT DU PACKAGE
# ============================================================
# MASS contient lda() pour faire du Linear Discriminant Analysis
library(MASS)

set.seed(42)

# ============================================================
# 2. GÉNÉRATION DE DEUX CLASSES "ELLIPTIQUES" ET ALLONGÉES
# ============================================================
# On construit une matrice de covariance qui donne une forte
# variance dans la direction (1,1) et une faible variance dans
# la direction perpendiculaire (1,-1). Ça crée des nuages de
# points en forme d'ellipses allongées en diagonale.

# Matrice de covariance partagée par les deux classes
Sigma <- matrix(c(8, 7,
                  7, 8), nrow = 2)
# Valeurs propres de Sigma : très différentes -> ellipse allongée
# le long du vecteur propre (1,1)/sqrt(2)

n <- 100

# Classe 1 : centrée en (0, 0)
class1 <- mvrnorm(n, mu = c(0, 0), Sigma = Sigma)

# Classe 2 : centrée en (3, -3)
# Le décalage entre les deux moyennes se fait dans la direction
# (1,-1), c'est-à-dire PERPENDICULAIRE à l'axe d'allongement (1,1).
# C'est exactement le piège pour PCA.
class2 <- mvrnorm(n, mu = c(3, -3), Sigma = Sigma)

X <- rbind(class1, class2)
y <- factor(c(rep("A", n), rep("B", n)))

# ============================================================
# 3. VISUALISATION DES DONNÉES BRUTES
# ============================================================
plot(X[, 1], X[, 2],
     col = y, pch = 19,
     xlab = "x1", ylab = "x2",
     main = "Deux classes : ellipses allongées, décalées perpendiculairement")
legend("topleft", legend = levels(y), col = 1:2, pch = 19)
grid()

# ============================================================
# 4. PCA — maximise la variance totale, IGNORE les labels
# ============================================================
pca <- prcomp(X, center = TRUE, scale. = FALSE)

# Premier axe principal (direction de variance maximale)
pc1_direction <- pca$rotation[, 1]  # vecteur propre n°1
cat("Direction du 1er axe PCA :", pc1_direction, "\n")
# -> proche de (1,1)/sqrt(2), car c'est la direction d'allongement
#    des ellipses, indépendamment du fait que ce soit cette
#    direction qui sépare le mieux (ou pas) les classes.

# ============================================================
# 5. LDA — maximise la séparation entre classes (S_B / S_W)
# ============================================================
lda_model <- lda(X, grouping = y)

# Coefficients de l'axe discriminant (équivalent du vecteur w
# qu'on a résolu mathématiquement via S_W^{-1} S_B w = lambda w)
lda_direction <- lda_model$scaling[, 1]
lda_direction <- lda_direction / sqrt(sum(lda_direction^2))  # normalisation
cat("Direction du 1er axe LDA :", lda_direction, "\n")
# -> proche de (1,-1)/sqrt(2), la direction qui sépare le mieux
#    les deux nuages, même si la variance y est plus faible.

# ============================================================
# 6. VISUALISATION COMPARATIVE DES DEUX AXES SUR LES DONNÉES
# ============================================================
plot(X[, 1], X[, 2],
     col = y, pch = 19,
     xlab = "x1", ylab = "x2",
     main = "Axe PCA (bleu) vs Axe LDA (vert)")
legend("topleft", legend = levels(y), col = 1:2, pch = 19)

center <- colMeans(X)
scale_len <- 6  # juste pour que les flèches soient visibles

# Axe PCA en bleu : suit l'allongement des ellipses
arrows(center[1] - scale_len * pc1_direction[1],
       center[2] - scale_len * pc1_direction[2],
       center[1] + scale_len * pc1_direction[1],
       center[2] + scale_len * pc1_direction[2],
       col = "blue", lwd = 3, length = 0.1)

# Axe LDA en vert : suit la séparation entre les classes
arrows(center[1] - scale_len * lda_direction[1],
       center[2] - scale_len * lda_direction[2],
       center[1] + scale_len * lda_direction[1],
       center[2] + scale_len * lda_direction[2],
       col = "darkgreen", lwd = 3, length = 0.1)

legend("bottomright",
       legend = c("Axe PCA", "Axe LDA"),
       col = c("blue", "darkgreen"), lwd = 3)
grid()

# ============================================================
# 7. PROJECTION 1D SUR CHAQUE AXE — pour voir QUI sépare vraiment
# ============================================================
proj_pca <- X %*% pc1_direction   # projection sur l'axe PCA
proj_lda <- X %*% lda_direction   # projection sur l'axe LDA

par(mfrow = c(2, 1))  # deux graphiques empilés

# Projection sur l'axe PCA : les classes sont MÉLANGÉES
stripchart(proj_pca ~ y, method = "jitter", pch = 19, col = 1:2,
           main = "Projection sur l'axe PCA (classes mélangées)",
           xlab = "valeur projetée")

# Projection sur l'axe LDA : les classes sont SÉPARÉES
stripchart(proj_lda ~ y, method = "jitter", pch = 19, col = 1:2,
           main = "Projection sur l'axe LDA (classes séparées)",
           xlab = "valeur projetée")

par(mfrow = c(1, 1))  # on remet l'affichage par défaut
