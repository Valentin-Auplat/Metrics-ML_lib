import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

import os
import sys

#In case of retreiving a file to import functions (to modify depending on the architecture using os.path.dirname & os.path.join)
#sys.path.append(os.path.dirname(__file__), '..', 'file_location')
#from file_name import function

#Retreiving data
#paths definition
path = os.path.dirname(__file__)
data_path = os.path.join(path, "data_file")
#data reading
df = pd.read_csv(os.path.join(data_path, 'data_file_name'))

#Might want to clean data (work on categories, filling the blanks, regular data engineering)

#Defining train & test sets
X, y = df.drop(columns=["Category", "other_cols_if_needed"]), df["Category"]
X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size = to_refine, random_state = seed
)

#Fitting one tree
clf = tree.DecisionTreeClassifier(max_depth=to_refine)
clf = clf.fit(X_train, y_train)

#Predict over the single tree
y_pred = clf.predict(X_test)
#Tests to diplay
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


#On passe à la RF et la random search des meilleurs paramètres.
#On a besoin de fit le modèle une première fois.
clf = RandomForestClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)

#Range des paramètres à estimer :
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(2, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
#Recherche des hyperparamètres :
rand_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter = 20,
    cv=5,
    scoring='f1_macro',
    n_jobs=1,
    random_state=42
)
#Nouveau fit avec les meilleurs hyperparamètres :
rand_search.fit(X_train, y_train)
best_clf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)
#On prédit et affiche les résultats de tests :
y_best_pred = best_clf.predict(X_test)
print(classification_report(y_test, y_best_pred))
print(confusion_matrix(y_test, y_best_pred))