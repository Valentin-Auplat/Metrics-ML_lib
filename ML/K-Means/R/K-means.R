# ============================================================
# 1. CHARGEMENT DES PACKAGES
# ============================================================
# stats::kmeans est inclus dans R de base, donc pas besoin de
# packages externes pour le clustering lui-même.
# On utilise juste les graphiques de base pour rester simple.

# ============================================================
# 2. GÉNÉRATION DU DATASET SYNTHÉTIQUE (équivalent de make_blobs)
# ============================================================
# R n'a pas de make_blobs() natif comme sklearn. On reproduit le
# même principe "à la main" : on tire des points aléatoires autour
# de K centres fixés, avec un certain écart-type (cluster_std).

set.seed(42)  # équivalent de random_state=42 : reproductibilité

n_samples   <- 150
centers_k   <- 3
cluster_std <- 2.5

# On choisit 3 centres "vrais" arbitraires dans le plan (x1, x2)
true_centers <- matrix(c( 0,  0,
                          10, 10,
                          0, 10),
                       nrow = centers_k, byrow = TRUE)

# On répartit les 150 points de façon égale entre les 3 clusters
points_per_cluster <- rep(n_samples %/% centers_k, centers_k)

# Pour chaque centre, on génère un nuage de points gaussien autour
# de lui (rnorm = bruit gaussien, comme le ferait make_blobs)
X <- do.call(rbind, lapply(1:centers_k, function(k) {
  cbind(
    rnorm(points_per_cluster[k], mean = true_centers[k, 1], sd = cluster_std),
    rnorm(points_per_cluster[k], mean = true_centers[k, 2], sd = cluster_std)
  )
}))
colnames(X) <- c("x1", "x2")

# y_true : on garde la trace du cluster d'origine de chaque point
# (utile seulement si on veut comparer à la vérité, pas obligatoire
# pour le clustering lui-même)
y_true <- rep(1:centers_k, points_per_cluster)

# ============================================================
# 3. VISUALISATION DES DONNÉES BRUTES (équivalent du premier plt.scatter)
# ============================================================
plot(X[, "x1"], X[, "x2"],
     main = "Synthetic 2D dataset for K-means",
     xlab = "x1", ylab = "x2",
     pch = 19, cex = 1,    # pch=19 -> points pleins, cex -> taille
     col = "black")
grid()  # équivalent de plt.grid(which="both")

# ============================================================
# 4. K-MEANS "TOUT FAIT" (remplace toute ta mécanique manuelle :
#    compute_distances, assign_clusters, update_centroids, et la
#    boucle kmeans())
# ============================================================
# stats::kmeans() fait exactement ce que faisait KMeans de sklearn :
# - centers = 3        -> nombre de clusters K
# - iter.max = 200      -> équivalent de max_iter
# - nstart = 1          -> équivalent de n_init=1 (une seule
#                          initialisation aléatoire des centroïdes ;
#                          en pratique on met souvent nstart=10 ou
#                          plus pour éviter un mauvais minimum local,
#                          comme le n_init>1 de sklearn)
# - algorithm = "Lloyd" -> c'est exactement l'algorithme que tu as
#                          recodé à la main (assignation puis mise
#                          à jour des centroïdes par la moyenne)

km <- kmeans(X, centers = centers_k, iter.max = 200,
             nstart = 1, algorithm = "Lloyd")

# ============================================================
# 5. RÉCUPÉRATION DES RÉSULTATS
# ============================================================
labels    <- km$cluster        # équivalent de labels (km.fit_predict(X))
centroids <- km$centers        # équivalent de km.cluster_centers_
distortion <- km$tot.withinss / nrow(X)  
# km$tot.withinss = somme des distances au carré intra-cluster
# (équivalent de km.inertia_ dans sklearn).
# On divise par n pour retrouver exactement ta formule
# compute_distortion (qui divisait par X.shape[0]).

n_iter <- km$iter  # équivalent de km.n_iter_ : nb d'itérations réelles

cat("Distortion (inertia) :", distortion, "\n")
cat("Nombre d'itérations  :", n_iter, "\n")

# ============================================================
# 6. VISUALISATION FINALE (équivalent de plot_clusters)
# ============================================================
# On colore chaque point selon son cluster, et on superpose les
# centroïdes en grosses croix rouges, comme dans ta fonction
# plot_clusters(X, labels, centroids, iteration).

plot(X[, "x1"], X[, "x2"],
     col = labels,          # couleur = numéro de cluster (1, 2, 3...)
     pch = 19, cex = 1.2,
     main = paste("K-means iteration", n_iter),
     xlab = "x1", ylab = "x2")

points(centroids[, "x1"], centroids[, "x2"],
       pch = 4,             # pch=4 -> symbole "X"
       cex = 3,             # grosse taille, comme s=300 dans matplotlib
       lwd = 3,             # épaisseur du trait
       col = "red")

legend("topright", legend = "Centroids", pch = 4, col = "red", pt.cex = 1.5)
grid()

