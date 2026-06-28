# install.packages("glmnet")
library(glmnet)

## ============================================================
## 1. Données
## ============================================================
data(iris)
X <- as.matrix(iris[, 1:4])
y <- iris$Species
feature_names <- colnames(X)
target_names  <- levels(y)

print(head(iris))
print(dim(iris))

## ============================================================
## 2. Split train/test 50/50 (équivalent random_state=42)
## ============================================================
set.seed(42)
n <- nrow(X)
train_idx <- sample(seq_len(n), size = floor(0.5 * n))

X_train <- X[train_idx, ];  X_test <- X[-train_idx, ]
y_train <- y[train_idx];    y_test <- y[-train_idx]

## ============================================================
## 3. Standardisation (moyenne/sd calculées sur train uniquement)
## ============================================================
train_means <- colMeans(X_train)
train_sds   <- apply(X_train, 2, sd)

scale_with <- function(mat, means, sds) sweep(sweep(mat, 2, means, "-"), 2, sds, "/")

X_train_s <- scale_with(X_train, train_means, train_sds)
X_test_s  <- scale_with(X_test,  train_means, train_sds)

## ============================================================
## 4. Calcul du chemin complet de régularisation
##    (au lieu d'un lambda isolé -> warm start correct, coefficients fiables)
## ============================================================
full_path <- glmnet(X_train_s, y_train, family = "multinomial", alpha = 0)

# Scores sur tout le chemin
pred_class_path <- predict(full_path, X_test_s, type = "class")          # matrice (n_test x n_lambda)
scores_path <- apply(pred_class_path, 2, function(col) mean(col == y_test))

# Sélection du meilleur lambda (premier max en cas d'égalité)
best_idx    <- which.max(scores_path)
best_lambda <- full_path$lambda[best_idx]
cat(sprintf("Meilleur lambda : %.5f (score test = %.4f)\n",
            best_lambda, scores_path[best_idx]))

# Pour information : C approximatif correspondant (1/lambda, cf. avertissement précédent)
cat(sprintf("C équivalent approx. : %.4f\n", 1 / best_lambda))

## ============================================================
## 5. Prédiction finale au meilleur lambda
## ============================================================
y_best_pred <- predict(full_path, X_test_s, type = "class", s = best_lambda)
cat(sprintf("Score test interne : %.4f\n", mean(y_best_pred == y_test)))

## ============================================================
## 6. Scatter brut : longueur des pétales vs classe réelle
## ============================================================
y_numeric <- as.numeric(y) - 1
plot(X[, 3], y_numeric, xlab = "Longueur des pétales (cm)", ylab = "Classe",
     main = "Longueur des pétales vs classe", pch = 16,
     col = as.numeric(y))

## ============================================================
## 7. Plot multinomial : probabilités prédites par classe
## ============================================================
x_range <- seq(min(X[, 3]), max(X[, 3]), length.out = 300)

X_synth <- matrix(rep(colMeans(X), 300), nrow = 300, byrow = TRUE)
X_synth[, 3] <- x_range
colnames(X_synth) <- feature_names

X_synth_s <- scale_with(X_synth, train_means, train_sds)

# Prédiction au bon lambda via s= (et non plus un fit isolé)
probas <- predict(full_path, X_synth_s, type = "response", s = best_lambda)[, , 1]  # 300 x 3

plot(x_range, probas[, 1], type = "l", col = 1, lwd = 2,
     xlab = "Longueur des pétales (cm)", ylab = "Probabilité prédite",
     main = "Probabilités par classe vs longueur des pétales",
     ylim = c(0, 1))
lines(x_range, probas[, 2], col = 2, lwd = 2)
lines(x_range, probas[, 3], col = 3, lwd = 2)

# Points observés, alignés sur leur vraie classe (0, 1 ou 2) avec un jitter vertical
for (i in seq_along(target_names)) {
  mask   <- (y_numeric == (i - 1))
  jitter <- runif(sum(mask), -0.02, 0.02)
  points(X[mask, 3], rep((i - 1) / 2, sum(mask)) + jitter,  # repère visuel par classe, pas une vraie probabilité
         col = i, pch = 16, cex = 0.6)
}
legend("topleft", legend = paste0("P(", target_names, ")"), col = 1:3, lwd = 2)

## ============================================================
## 8. Plot binomial : setosa vs reste, courbe sigmoïde (glm, non régularisé)
## ============================================================
y_binary <- as.integer(y == "setosa")

logit_demo <- glm(y_binary ~ petal_length,
                  data = data.frame(y_binary = y_binary, petal_length = X[, 3]),
                  family = binomial)

x_range2   <- data.frame(petal_length = seq(min(X[, 3]), max(X[, 3]), length.out = 300))
proba_demo <- predict(logit_demo, newdata = x_range2, type = "response")

plot(X[, 3], y_binary, xlab = "Longueur des pétales (cm)", ylab = "P(setosa)",
     main = "Régression logistique binaire - courbe sigmoïde", pch = 16,
     col = rgb(0, 0, 0, 0.4))
lines(x_range2$petal_length, proba_demo, col = "red", lwd = 2)