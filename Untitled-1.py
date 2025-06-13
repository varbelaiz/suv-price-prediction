# %%
import pandas as pd
import numpy as np
import re


from unicodedata import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

# %%
path = "data/train/cleaned_train.csv"

df = pd.read_csv(path)

# %% [markdown]
# # Clusterización de versiones
# 

# %% [markdown]
# ## Experimentos para un modelo especificado

# %%
brand = "Chevrolet"
model = "Tracker"

version_df = df[(df["Marca"] == brand) & (df["Modelo"] == model)].reset_index(drop=True)

# %% [markdown]
# ### Limpieza y preprocesamiento de versiones
# 
# Este código se encarga de limpiar y normalizar los nombres de las versiones de vehículos para facilitar su posterior análisis y agrupación:
# 
# **Función** `clean_version()`
# - **Normalización de caracteres**: Convierte texto a minúsculas y elimina acentos/caracteres especiales usando Unicode NFKD
# - **Filtrado de caracteres**: Solo mantiene letras, números y espacios, eliminando símbolos y puntuación
# - **Limpieza de espacios**: Normaliza espacios múltiples a espacios únicos y elimina espacios al inicio/final
# - **Manejo de valores nulos**: Convierte valores `NaN` a strings vacíos
# 

# %%
def clean_version(s: str) -> str:
    s = "" if pd.isna(s) else s
    s = normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

version_df["version_clean"] = version_df["Versión"].apply(clean_version)
version_df = version_df[version_df["version_clean"] != ""].reset_index(drop=True)

print("Número de versiones únicas:", version_df["version_clean"].nunique())

print("\nPrimeras 5 versiones únicas:")
print(version_df["version_clean"].value_counts()[:5])

# %% [markdown]
# ### HAC y TF-IDF

# %%
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
X = vectorizer.fit_transform(version_df["version_clean"])

non_zero_mask = np.array(X.sum(axis=1)).flatten() > 0
X_filtered = X[non_zero_mask].toarray()
taos_df_filtered = version_df[non_zero_mask].copy()

model = AgglomerativeClustering(
    metric="cosine",
    linkage="average",
    distance_threshold=0.8,
    n_clusters=None,
)

# Fit on filtered data
clusters_filtered = model.fit_predict(X_filtered)

# Assign clusters back to original dataframe
version_df["cluster"] = -1  # Default for zero vectors
version_df.loc[non_zero_mask, "cluster"] = clusters_filtered

# %%
canon = version_df.groupby("cluster")["Versión"].agg(lambda x: x.value_counts().idxmax())
version_df["version_canon"] = version_df["cluster"].map(canon)

summary = (
    version_df.groupby(["cluster", "version_canon"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

summary

# %% [markdown]
# ### Affinity propagation

# %%
S = cosine_similarity(X)

pref = np.median(S) - 0.05
ap = AffinityPropagation(affinity="precomputed",
                         damping=0.95,
                         preference=pref,
                         max_iter=500,
                         convergence_iter=16)

version_df["cluster"] = ap.fit_predict(S)


canon = version_df.groupby("cluster")["Versión"].agg(lambda x: x.value_counts().idxmax())
version_df["version_canon"] = version_df["cluster"].map(canon)

summary = (
    version_df.groupby(["cluster", "version_canon"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
summary


# %% [markdown]
# ### DBSCAN
# 
# - Leer `CountVectorizer` en [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) (podemos jugar con los argumentos)

# %%
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 5))
X = vectorizer.fit_transform(version_df["version_clean"]).toarray()

print(X.shape)

# %%
n_versions = version_df["version_clean"].nunique()

db = DBSCAN(eps=0.3, min_samples=max(2, n_versions // 4), metric="cosine")
version_df["cluster"] = db.fit_predict(X)

# canónico solo para clusters válidos (≠ –1)
canon = (
    version_df[version_df["cluster"] != -1]
      .groupby("cluster")["Versión"]
      .agg(lambda s: s.value_counts().idxmax())
)
version_df["version_canon"] = version_df["cluster"].map(canon)

# todo el ruido recibe una misma etiqueta
version_df.loc[version_df["cluster"] == -1, "version_canon"] = "Unassigned"

summary = (
    version_df.groupby(["cluster", "version_canon"])
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False)
)
print(summary)
print("\nClusters densos:", canon.size,
      "  Etiquetas finales:", version_df["version_canon"].nunique())


# %% [markdown]
# Asignar cada muestra al centoride al que pertenece

# %%
version_df["Versión"] = version_df["version_canon"]
version_df.drop(columns=["cluster", "version_canon"], inplace=True)

print("Versiones únicas tras el clustering:",
      version_df["Versión"].nunique())
print("Etiquetas:", version_df["Versión"].unique())


# %% [markdown]
# # Compartir features entre cada versión

# %%
def fill_by_mode(df, col, keys=("Marca", "Modelo", "Versión")):
    mode_vals = (
        df.groupby(list(keys))[col]
          .transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    )
    df[col] = df[col].fillna(mode_vals)

cols_to_fill = ["cv", "Motor", "Tracción", "Turbo"]      # agrega o quita las que quieras

for c in cols_to_fill:
    if c in version_df.columns:
        fill_by_mode(version_df, c)

print("Faltantes después de imputar:")
print(version_df[cols_to_fill].isna().sum())


