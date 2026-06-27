# ============================================================
# 1. IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

np.random.seed(42)

# ============================================================
# 2. GÉNÉRATION DE DEUX CLASSES "ELLIPTIQUES" ET ALLONGÉES
# ============================================================
# On construit une matrice de covariance qui donne une forte
# variance dans la direction (1,1) et une faible variance dans
# la direction perpendiculaire (1,-1). Ça crée des nuages de
# points en forme d'ellipses allongées en diagonale.

# Matrice de covariance partagée par les deux classes
Sigma = np.array([[8, 7],
                   [7, 8]])
# Valeurs propres de Sigma : très différentes -> ellipse allongée
# le long du vecteur propre (1,1)/sqrt(2)

n = 100

# Classe 1 : centrée en (0, 0)
class1 = np.random.multivariate_normal(mean=[0, 0], cov=Sigma, size=n)

# Classe 2 : centrée en (3, -3)
# Le décalage entre les deux moyennes se fait dans la direction
# (1,-1), c'est-à-dire PERPENDICULAIRE à l'axe d'allongement (1,1).
# C'est exactement le piège pour PCA.
class2 = np.random.multivariate_normal(mean=[3, -3], cov=Sigma, size=n)

X = np.vstack([class1, class2])
y = np.array(["A"] * n + ["B"] * n)

# ============================================================
# 3. VISUALISATION DES DONNÉES BRUTES
# ============================================================
colors = np.where(y == "A", "tab:blue", "tab:orange")

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=colors, s=40)
plt.title("Deux classes : ellipses allongées, décalées perpendiculairement")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(handles=[
    plt.Line2D([], [], marker="o", color="w", markerfacecolor="tab:blue", label="A", markersize=8),
    plt.Line2D([], [], marker="o", color="w", markerfacecolor="tab:orange", label="B", markersize=8)
])
plt.grid(True)
plt.show()

# ============================================================
# 4. PCA — maximise la variance totale, IGNORE les labels
# ============================================================
pca = PCA(n_components=2)
pca.fit(X)

# Premier axe principal (direction de variance maximale)
pc1_direction = pca.components_[0]  # vecteur propre n°1
print("Direction du 1er axe PCA :", pc1_direction)
# -> proche de (1,1)/sqrt(2), car c'est la direction d'allongement
#    des ellipses, indépendamment du fait que ce soit cette
#    direction qui sépare le mieux (ou pas) les classes.

# ============================================================
# 5. LDA — maximise la séparation entre classes (S_B / S_W)
# ============================================================
lda_model = LDA(n_components=1)
lda_model.fit(X, y)

# Coefficients de l'axe discriminant (équivalent du vecteur w
# qu'on a résolu mathématiquement via S_W^{-1} S_B w = lambda w)
lda_direction = lda_model.scalings_[:, 0]
lda_direction = lda_direction / np.linalg.norm(lda_direction)  # normalisation
print("Direction du 1er axe LDA :", lda_direction)
# -> proche de (1,-1)/sqrt(2), la direction qui sépare le mieux
#    les deux nuages, même si la variance y est plus faible.

# ============================================================
# 6. VISUALISATION COMPARATIVE DES DEUX AXES SUR LES DONNÉES
# ============================================================
center = X.mean(axis=0)
scale_len = 6  # juste pour que les flèches soient visibles

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=colors, s=40)
plt.title("Axe PCA (bleu) vs Axe LDA (vert)")
plt.xlabel("x1")
plt.ylabel("x2")

# Axe PCA en bleu : suit l'allongement des ellipses
plt.arrow(center[0] - scale_len * pc1_direction[0],
          center[1] - scale_len * pc1_direction[1],
          2 * scale_len * pc1_direction[0],
          2 * scale_len * pc1_direction[1],
          color="blue", linewidth=3, head_width=0.4, length_includes_head=True)

# Axe LDA en vert : suit la séparation entre les classes
plt.arrow(center[0] - scale_len * lda_direction[0],
          center[1] - scale_len * lda_direction[1],
          2 * scale_len * lda_direction[0],
          2 * scale_len * lda_direction[1],
          color="darkgreen", linewidth=3, head_width=0.4, length_includes_head=True)

plt.legend(handles=[
    plt.Line2D([], [], color="blue", lw=3, label="Axe PCA"),
    plt.Line2D([], [], color="darkgreen", lw=3, label="Axe LDA")
], loc="lower right")
plt.grid(True)
plt.show()

# ============================================================
# 7. PROJECTION 1D SUR CHAQUE AXE — pour voir QUI sépare vraiment
# ============================================================
proj_pca = X @ pc1_direction   # projection sur l'axe PCA
proj_lda = X @ lda_direction   # projection sur l'axe LDA

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

# Projection sur l'axe PCA : les classes sont MÉLANGÉES
jitter_A = np.random.normal(0, 0.05, size=n)
jitter_B = np.random.normal(0, 0.05, size=n)

axes[0].scatter(proj_pca[:n], jitter_A, color="tab:blue", label="A")
axes[0].scatter(proj_pca[n:], jitter_B, color="tab:orange", label="B")
axes[0].set_title("Projection sur l'axe PCA (classes mélangées)")
axes[0].set_xlabel("valeur projetée")
axes[0].set_yticks([])
axes[0].legend()

# Projection sur l'axe LDA : les classes sont SÉPARÉES
axes[1].scatter(proj_lda[:n], jitter_A, color="tab:blue", label="A")
axes[1].scatter(proj_lda[n:], jitter_B, color="tab:orange", label="B")
axes[1].set_title("Projection sur l'axe LDA (classes séparées)")
axes[1].set_xlabel("valeur projetée")
axes[1].set_yticks([])
axes[1].legend()

plt.tight_layout()
plt.show()