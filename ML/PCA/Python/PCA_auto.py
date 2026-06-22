import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'file'))

path = os.path.dirname(os.path.dirname(__file__))

# === Chargement et preprocessing ===
df = pd.read_csv(os.path.join(path, "data"))
X = df.drop(columns=["features"])
y = df["target"]

# === Standardisation (obligatoire avant PCA) ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA complète (toutes les composantes pour analyse) ===
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Forcer les noms de colonnes en PC1, PC2, PC3...
# (utile si set_config(transform_output="pandas") est actif)
if isinstance(X_pca_full, pd.DataFrame):
    X_pca_full.columns = [f'PC{i+1}' for i in range(X_pca_full.shape[1])]
else:
    # Si X_pca_full est un ndarray, on le convertit en DataFrame pour la suite
    X_pca_full = pd.DataFrame(
        X_pca_full,
        columns=[f'PC{i+1}' for i in range(X_pca_full.shape[1])],
        index=X.index
    )

# === Analyse 1 : variance expliquée par composante ===
variance_df = pd.DataFrame({
    'Composante': [f'PC{i+1}' for i in range(len(pca_full.explained_variance_))],
    'Valeur_propre': pca_full.explained_variance_,
    'Variance_expliquee': pca_full.explained_variance_ratio_,
    'Variance_cumulee': np.cumsum(pca_full.explained_variance_ratio_)
})

print("=== Variance expliquée ===")
print(variance_df)
variance_df.to_csv(os.path.join(path, "data/PCA_v1/pca_variance_expliquée_v1.csv"), index=False)

# === Analyse 2 : loadings (contributions des features originales) ===
loadings_df = pd.DataFrame(
    pca_full.components_.T,  # transpose pour avoir features en lignes
    columns=[f'PC{i+1}' for i in range(pca_full.n_components_)],
    index=X.columns
)

print("\n=== Loadings (contributions des features) ===")
print(loadings_df.round(3))
loadings_df.to_csv(os.path.join(path, "data/PCA_v1/pca_loadings_v1.csv"))

# === Analyse 3 : données projetées dans l'espace PCA ===
projected_df = X_pca_full.copy()
projected_df['target'] = y.values

print("\n=== Données projetées (5 premières lignes) ===")
print(projected_df.head())
projected_df.to_csv(os.path.join(path, "data/PCA_v1/pca_projected_v1.csv"), index=False)

# === Visualisation 1 : scree plot et variance cumulée ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_)
axes[0].set_xlabel('Composante')
axes[0].set_ylabel('Variance expliquée')
axes[0].set_title('Scree plot')

axes[1].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             np.cumsum(pca_full.explained_variance_ratio_), 'o-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95%')
axes[1].set_xlabel('Nombre de composantes')
axes[1].set_ylabel('Variance cumulée')
axes[1].set_title('Variance cumulée')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(path, "figures/PCA_v1/pca_variance_v1.png"))
plt.close()

# === Visualisation 2 : projection 2D colorée par survie ===
# Aligner les index pour éviter les bugs si y et X_pca_full ont des index différents
X_pca_full = X_pca_full.reset_index(drop=True)
y_reset = y.reset_index(drop=True)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca_full.loc[y_reset == 0, 'PC1'],
            X_pca_full.loc[y_reset == 0, 'PC2'],
            alpha=0.5, label='Décédés', c='red')
plt.scatter(X_pca_full.loc[y_reset == 1, 'PC1'],
            X_pca_full.loc[y_reset == 1, 'PC2'],
            alpha=0.5, label='Survivants', c='green')
plt.xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)')
plt.title('Projection sur les 2 premières composantes principales')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "figures/PCA_v1/pca_projection_2d_v1.png"))
plt.close()

plt.figure(figsize=(10, 7))
plt.scatter(X_pca_full.loc[y_reset == 0, 'PC1'],
            X_pca_full.loc[y_reset == 0, 'PC3'],
            alpha=0.5, label='Décédés', c='red')
plt.scatter(X_pca_full.loc[y_reset == 1, 'PC1'],
            X_pca_full.loc[y_reset == 1, 'PC3'],
            alpha=0.5, label='Survivants', c='green')
plt.xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}% var)')
plt.title('Projection sur les composantes principales 1 et 3')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path, "figures/PCA_v1/pca_projection_2d_PC1_PC3_v1.png"))
plt.close()

print("\n=== Fichiers générés ===")
print("CSV: pca_variance_Master.csv, pca_loadings_Master.csv, pca_projected_Master.csv")
print("Figures: pca_variance_Master.png, pca_projection_2d_Master.png, pca_projection_2d_PC1_PC3_Master.png")




#Les plots 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Reset des index pour éviter les bugs d'alignement
X_pca_reset = X_pca_full.reset_index(drop=True)
y_reset = y.reset_index(drop=True)

# Tracer les deux classes
ax.scatter(
    X_pca_reset.loc[y_reset == 0, 'PC1'],
    X_pca_reset.loc[y_reset == 0, 'PC3'],
    X_pca_reset.loc[y_reset == 0, 'PC2'],  # PC2 sur l'axe z
    alpha=0.5, label='Décédés', c='red', s=20
)
ax.scatter(
    X_pca_reset.loc[y_reset == 1, 'PC1'],
    X_pca_reset.loc[y_reset == 1, 'PC3'],
    X_pca_reset.loc[y_reset == 1, 'PC2'],
    alpha=0.5, label='Survivants', c='green', s=20
)

ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%) — solitude/famille')
ax.set_ylabel(f'PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}%) — genre')
ax.set_zlabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%) — classe')
ax.set_title('Projection 3D — PC1, PC2, PC3')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(path, "figures/PCA_v1/pca_projection_3d_v1.png"), dpi=150)
plt.close()


import plotly.express as px

# Préparer un DataFrame combiné pour plotly
plot_df = X_pca_reset[['PC1', 'PC2', 'PC3']].copy()
plot_df['Survived'] = y_reset.map({0: 'Décédé', 1: 'Survivant'})

fig = px.scatter_3d(
    plot_df, 
    x='PC1', y='PC3', z='PC2',
    color='Survived',
    color_discrete_map={'Décédé': 'red', 'Survivant': 'green'},
    opacity=0.6,
    title='Projection PCA 3D — Titanic',
    labels={
        'PC1': f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%) — solitude/famille',
        'PC3': f'PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}%) — genre',
        'PC2': f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%) — classe'
    }
)
fig.update_traces(marker=dict(size=4))
fig.write_html(os.path.join(path, "figures/PCA_v1/pca_projection_3d_v1.html"))
# fig.show()  # ouvre dans le navigateur

