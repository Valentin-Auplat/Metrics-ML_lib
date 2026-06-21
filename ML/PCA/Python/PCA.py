import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import os

path = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(path, "file_name"))

#On sépare les features de la cible
A = df[:, col:col]
y = df.reshape(-1, 1)[:, col]

#On centre et réduit les données pour qu'elles soient comparables
mA = A.mean(axis=0)
sA = A.std(axis=0)
A = (A-mA)/sA
#Au lieu de passer par la fonction corr, on fait nous-même la matrice de covariance (ou de correl dans ce cas parce qu'on a réduit)
C = A.transpose().dot(A)

#On peut produire la matrice de correlation en graph :
plt.figure(figsize=(8,6))
sns.heatmap(C, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('title')
plt.savefig(f'{path}/file/name')

#Au cas où on voudrait faire des stats par catégorie en découpant de nouvelles catégories :
# df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
#                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

# for j in ['Pclass', 'Sex', 'Embarked', 'AgeGroup']:
#     print(f"\n--- Survival rate by {j} ---")
#     print(df.groupby(j)["Survived"].mean() * 100)

