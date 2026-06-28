import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_prep'))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, set_config

from sklearn.datasets import load_iris

# path = os.path.dirname(os.path.dirname(__file__))


# df = pd.read_csv(os.path.join(path, 'data/train.csv'))
# df = cleaner_logreg(df)
# print(df.head())
# df.to_csv(os.path.join(path, "data/logreg_v5/test_nouvelles_variables.csv"), encoding='utf-8', index=False)

iris = load_iris()
X = iris.data          # (150, 4) -> features numériques
y = iris.target        # (150,)   -> labels 0, 1, 2
feature_names = iris.feature_names
target_names = iris.target_names

# Si tu veux un DataFrame complet pour explorer rapidement
df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print(df.head())
print(df.shape)

# X, y = df.drop(columns=["Survived", 'LastName', 'LastNameGroupSize', 'Sex_Alone']), df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

#On standardise les données #Ajouté à l'étape 4
set_config(transform_output="pandas")
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Entraînement du modèle
logit = LogisticRegression(max_iter=10000)

C = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
scores = []
for choice in C:
    logit.set_params(C=choice)
    logit.fit(X_train, y_train)
    scores.append(logit.score(X_test, y_test))
print(scores)
print(f"Score test interne: {logit.score(X_test, y_test):.4f}")

#Test du modèle
# teste = pd.read_csv(os.path.join(path, "data/test.csv"))
# teste = cleaner_logreg(teste)
# teste = teste.drop(columns=['LastName', 'LastNameGroupSize', 'Sex_Alone'])
# teste['Deck_T'] = 0
# teste = teste[X_train.columns]#Pour avoir le même ordre dans les colonnes
# #On standardise les données #Ajouté à l'étape 4
# teste[['Fare', 'Age', 'SibSp', 'Parch', 'Family_size', 'TicketGroupSize']] = scaler.transform(teste[['Fare', 'Age', 'SibSp', 'Parch', 'Family_size', 'TicketGroupSize']])

logit.set_params(C=C[scores.index(max(scores))])
logit.fit(X_train, y_train)
y_best_pred = logit.predict(X_test)

# teste.to_csv(os.path.join(path, "data/logreg_v5/submission_logreg.csv"), encoding='utf-8', index=False)
plt.scatter(X[:, 2], y)


#Plot multinomial
import numpy as np

# On fait varier la longueur des pétales sur toute sa plage,
# en figeant les 3 autres features à leur moyenne (sur les données brutes)
x_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 300)

X_synth = np.tile(X.mean(axis=0), (len(x_range), 1))  # autres features = moyenne
X_synth[:, 2] = x_range                                 # on ne fait varier que la colonne 2

X_synth_df = pd.DataFrame(X_synth, columns=feature_names)
X_synth_scaled = scaler.transform(X_synth_df)           # même standardisation que l'entraînement

probas = logit.predict_proba(X_synth_scaled)            # shape (300, 3)

# Tracé : une courbe en S par classe
plt.figure(figsize=(8, 5))
for i, name in enumerate(target_names):
    plt.plot(x_range, probas[:, i], label=f"P({name})")



#Plot binomial
# Points observés : on les place à 0 ou 1 sur la classe correspondante,
# avec un peu de bruit vertical pour la lisibilité (jitter)
for i in range(3):
    mask = (y == i)
    jitter = np.random.uniform(-0.02, 0.02, mask.sum())
    plt.scatter(X[mask, 2], np.ones(mask.sum()) + jitter - 1 + i*0,
                alpha=0.3, s=15, color=f"C{i}")

plt.xlabel("Longueur des pétales (cm)")
plt.ylabel("Probabilité prédite")
plt.title("Probabilités par classe vs longueur des pétales")
plt.legend()
plt.show()


y_binary = (y == 0).astype(int)  # setosa = 1, les autres = 0
logit_demo = LogisticRegression().fit(X[:, [2]], y_binary)  # 1 seule feature, non standardisée pour rester lisible

x_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 300).reshape(-1, 1)
proba_demo = logit_demo.predict_proba(x_range)[:, 1]

plt.scatter(X[:, 2], y_binary, alpha=0.5)
plt.plot(x_range, proba_demo, color="red")
plt.xlabel("Longueur des pétales (cm)")
plt.ylabel("P(setosa)")
plt.title("Régression logistique binaire - courbe sigmoïde")
plt.show()