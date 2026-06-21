# ===== Chargement des librairies =====
library(rpart)          # arbre de décision simple (équivalent tree.DecisionTreeClassifier)
library(mlr3)           # framework ML principal (équivalent sklearn)
library(mlr3learners)   # contient le learner randomForest ("classif.ranger")
library(mlr3tuning)     # équivalent de RandomizedSearchCV
library(paradox)        # définition des espaces de paramètres (équivalent param_dist)
library(caret)          # pour confusionMatrix uniquement (affichage des résultats)
library(here)           # gestion des chemins

# ===== Récupération des données =====
# source(here("..", "dossier_fichier", "nom_fichier.R"))  # si besoin d'importer des fonctions

data_path <- here("data_file")
df <- read.csv(file.path(data_path, "data_file_name.csv"))

# Nettoyage des données si besoin (catégories, NA, etc.) ici

# La cible doit être un facteur pour la classification
df$Category <- as.factor(df$Category)

# ===== Définition des jeux d'entraînement et de test =====
set.seed(seed)  # équivalent random_state

# On retire les colonnes inutiles, comme X = df.drop(...)
df_model <- df[, !(names(df) %in% c("other_cols_if_needed"))]

train_index <- caret::createDataPartition(df_model$Category, p = 1 - to_refine, list = FALSE)
train_set <- df_model[train_index, ]
test_set  <- df_model[-train_index, ]

# ===== Fitting d'un arbre unique =====
# rpart = équivalent direct de DecisionTreeClassifier
clf <- rpart(
  Category ~ .,
  data = train_set,
  method = "class",
  control = rpart.control(maxdepth = to_refine)
)

# Prédiction sur l'arbre unique
y_pred <- predict(clf, newdata = test_set, type = "class")

# Affichage des résultats (classification_report + confusion_matrix)
print(confusionMatrix(y_pred, test_set$Category))


# ===== Passage à la Random Forest avec mlr3 =====

# Création de la tâche de classification (équivalent de X_train, y_train regroupés)
task_train <- as_task_classif(train_set, target = "Category")
task_test  <- as_task_classif(test_set, target = "Category")

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

# Métrique = f1_macro (mlr3 utilise classif.fbeta en macro-average)
measure <- msr("classif.fbeta", average = "macro")

# Instance de tuning : regroupe tâche, learner, resampling, mesure, espace
instance <- TuningInstanceSingleCrit$new(
  task         = task_train,
  learner      = learner_rf,
  resampling   = resampling,
  measure      = measure,
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
y_test_vec  <- test_set$Category

print(confusionMatrix(y_best_pred, y_test_vec))