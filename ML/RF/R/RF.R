# ===== Chargement des librairies =====
library(rpart)          # arbre de décision simple (équivalent tree.DecisionTreeClassifier)
library(rpart.plot)     # visualisation de l'arbre (équivalent tree.plot_tree)
library(mlr3)           # framework ML principal (équivalent sklearn)
library(mlr3learners)   # contient le learner randomForest ("classif.ranger")
library(mlr3tuning)     # équivalent de RandomizedSearchCV
library(paradox)        # définition des espaces de paramètres (équivalent param_dist)
library(caret)          # pour confusionMatrix uniquement (affichage des résultats)
library(here)           # gestion des chemins (utile seulement si on lit un fichier externe)

# ===== Récupération des données =====
# source(here("..", "dossier_fichier", "nom_fichier.R"))  # si besoin d'importer des fonctions
#
# data_path <- here("data_file")
# df <- read.csv(file.path(data_path, "data_file_name.csv"))
#
# # Nettoyage des données si besoin (catégories, NA, etc.) ici
#
# # La cible doit être un facteur pour la classification
# df$Category <- as.factor(df$Category)

data(iris)  # disponible nativement, aucun package requis

X <- as.matrix(iris[, 1:4])      # les 4 variables numériques
y <- iris$Species                 # facteur à 3 niveaux

head(iris)
dim(iris)

# ===== Définition des jeux d'entraînement et de test =====
set.seed(42)  # équivalent random_state=42, pour la reproductibilité

# p = proportion d'entraînement souhaitée (ici 70%), pas la proportion de test
train_index <- caret::createDataPartition(y, p = 0.7, list = FALSE)

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# CORRECTION : rpart() et as_task_classif() ont besoin d'un data.frame
# contenant à la fois les features ET la colonne cible — pas une matrice
# de features seule. On reconstruit donc des data.frames complets.
train_df <- data.frame(X_train, Species = y_train)
test_df  <- data.frame(X_test,  Species = y_test)

# ===== Fitting d'un arbre unique =====
# rpart = équivalent direct de DecisionTreeClassifier
clf <- rpart(
  Species ~ .,                                    # CORRECTION : "Species", pas "Category"
  data = train_df,                                # CORRECTION : data.frame complet, pas X_train
  method = "class",
  control = rpart.control(maxdepth = 3)            # CORRECTION : valeur numérique, "to_refine" n'existait pas
)

# ===== Visualisation de l'arbre (équivalent de sklearn.tree.plot_tree) =====
rpart.plot(clf,
           type = 4,            # affiche les étiquettes à chaque noeud (pas seulement les feuilles)
           extra = 104,         # affiche la classe majoritaire + la proportion de chaque classe (%)
           box.palette = "auto",  # une couleur par classe, comme les noeuds colorés de plot_tree
           main = "Arbre de décision - Iris")

# Prédiction sur l'arbre unique
y_pred <- predict(clf, newdata = test_df, type = "class")  # CORRECTION : test_df, pas X_test

# Affichage des résultats (classification_report + confusion_matrix)
print(confusionMatrix(y_pred, y_test))    # CORRECTION : y_test directement, pas y_test$Category
# (y_test est un vecteur factoriel, pas un data.frame)

# ===== Passage à la Random Forest avec mlr3 =====

# Création de la tâche de classification (équivalent de X_train, y_train regroupés)
# CORRECTION : on passe les data.frames complets (features + cible),
# et on corrige la faute de frappe "Specices" -> "Species"
task_train <- as_task_classif(train_df, target = "Species")
task_test  <- as_task_classif(test_df,  target = "Species")

# Définition du learner Random Forest (ranger est l'implémentation rapide de RF en R)
learner_rf <- lrn("classif.ranger",
                  predict_type = "prob",
                  num.trees   = 100,   # valeur initiale, sera tunée ensuite
                  max.depth   = 3      # équivalent du premier fit avec max_depth=3
)

# Premier entraînement "brut", comme dans le code Python
learner_rf$train(task_train)

# ===== Espace des hyperparamètres à tester =====
# Équivalent direct de param_dist avec randint()
search_space <- ps(
  num.trees        = p_int(lower = 100, upper = 500),  # n_estimators
  max.depth        = p_int(lower = 2,   upper = 15),   # max_depth
  min.node.size    = p_int(lower = 1,   upper = 5)     # ≈ min_samples_leaf
  # min_samples_split n'a pas d'équivalent direct dans ranger,
  # min.node.size est le paramètre le plus proche disponible
)

# ===== Configuration du Random Search =====
# Resampling = validation croisée à 5 folds (équivalent cv=5)
resampling <- rsmp("cv", folds = 5)

# Métrique = balanced accuracy (classif.fbeta est binaire uniquement dans mlr3,
# contrairement à f1_score(average="macro") qui est nativement multiclasse en sklearn ;
# classif.bacc est l'équivalent le plus proche compatible multiclasse)
measure <- msr("classif.bacc")

# Instance de tuning : regroupe tâche, learner, resampling, mesure, espace
# CORRECTION : ti() est la syntaxe recommandée (remplace TuningInstanceSingleCrit$new(),
# désormais dépréciée). Attention : "measures" est au pluriel avec ti().
instance <- ti(
  task         = task_train,
  learner      = learner_rf,
  resampling   = resampling,
  measures     = measure,
  search_space = search_space,
  terminator   = trm("evals", n_evals = 20)  # équivalent n_iter=20
)

# Random search comme algorithme de tuning (équivalent RandomizedSearchCV)
tuner <- tnr("random_search")

set.seed(42)  # équivalent random_state=42
tuner$optimize(instance)

# ===== Récupération des meilleurs hyperparamètres =====
best_params <- instance$result_learner_param_vals
cat("Best hyperparameters:\n")
print(best_params)

# ===== Nouveau fit avec les meilleurs hyperparamètres =====
best_clf <- lrn("classif.ranger", predict_type = "prob")
best_clf$param_set$values <- best_params
best_clf$train(task_train)

# ===== Prédiction et affichage des résultats finaux =====
pred <- best_clf$predict(task_test)

# Conversion en facteurs pour confusionMatrix
y_best_pred <- pred$response
y_test_vec  <- y_test    # CORRECTION : test_set n'existait pas, on réutilise y_test

print(confusionMatrix(y_best_pred, y_test_vec))