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
[n, p] = A.shape
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

#On peut construire un bar chart de la correlation entre les features et une feature ou la variable à prédire :
u = A.transpose().dot(y)/n
plt.figure()
plt.grid()
plt.bar(np.arange(1, p+1), u.flatten())
plt.tight_layout()


#Au cas où on voudrait faire des stats par catégorie en découpant de nouvelles catégories :
# df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
#                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

# for j in ['Pclass', 'Sex', 'Embarked', 'AgeGroup']:
#     print(f"\n--- Survival rate by {j} ---")
#     print(df.groupby(j)["Survived"].mean() * 100)

#Afficher les valeurs propres en ordre décroissant et les vectuers propres associés de la matrice de covariance qui constituent les axes de la PCA
#Calcul des VP (on utilise eigh parce que C est symétrique)
eigenvalues, eigenvectors = np.linalg.eigh(C)
#sorting
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
#affichage
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)
#Plot des VP en ordre décroissant
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of the covariance matrix (descending order)")
plt.grid(True)
plt.show()

#On définit manuellement les axes de la PCA par projection, et les projections de A sur ces axes
A_projected = A @ eigenvectors[:, :2]
plt.figure()
plt.grid
plt.plot(A_projected[:, 0], A_projected[:, 1], '.')

