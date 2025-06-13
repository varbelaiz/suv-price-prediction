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

from difflib import SequenceMatcher


# %%
path = "data/train/cleaned_train.csv"

df = pd.read_csv(path)

# %% [markdown]
# # Clusterización de versiones
# 

# %% [markdown]
# ## Experimentos para un modelo especificado

# %%
brand = "Ford"
model = "Ecosport"

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
# ### Usar los datos de `mercadolibre_versions.csv`

# %%
ml_versions = pd.read_csv('data/mercadolibre_versions.csv')
ford_ml = ml_versions[(ml_versions['Brand'] == brand) &
                      (ml_versions['Model'] == model ) &
                      (ml_versions['Version'] != 'N/A')]

real_versions = ford_ml['Version'].tolist()

clustered_counts = version_df['Versión'].value_counts()
print(clustered_counts.head())

threshold = 0.3
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))

final_mapping = {}
mapping_results = []

for clustered in clustered_counts.index:
    if clustered == 'Unassigned':
        best_match = 'Se'
        best_score = 0.0
    else:
        best_match = None
        best_score = 0.0
        clean_clustered = clean_version(clustered)
        for rv in real_versions:
            clean_real = clean_version(rv)
            if clean_clustered and clean_real:
                X = vectorizer.fit_transform([clean_clustered, clean_real])
                score = cosine_similarity(X[0:1], X[1:2])[0][0]
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_match = rv
        if best_score < threshold:
            best_match = 'Se'
    final_mapping[clustered] = best_match
    mapping_results.append({
        'clustered_version': clustered,
        'count': clustered_counts[clustered],
        'mapped_to': best_match,
        'similarity_score': best_score
    })

mapping_df = pd.DataFrame(mapping_results)
print(mapping_df.to_string(index=False))

# %%
# Apply the mapping to the dataframe
print("Before mapping:")
print("Original version counts:", version_df["Versión"].value_counts())
print(f"Total unique versions: {version_df['Versión'].nunique()}")

# Apply the mapping
version_df["Versión"] = version_df["Versión"].map(final_mapping)

print("\nAfter mapping:")
print("Mapped version counts:", version_df["Versión"].value_counts())
print(f"Total unique versions: {version_df['Versión'].nunique()}")

# Verify that all versions are now real MercadoLibre versions
mapped_versions = set(version_df["Versión"].unique())
real_versions_set = set(real_versions)

print(f"\nValidation:")
print(f"All mapped versions are real versions: {mapped_versions.issubset(real_versions_set)}")
print(f"Mapped versions: {sorted(mapped_versions)}")
print(f"Real versions not used: {sorted(real_versions_set - mapped_versions)}")

# Show final summary
final_summary = version_df["Versión"].value_counts().reset_index()
final_summary.columns = ["Version", "Count"]
final_summary["Percentage"] = (final_summary["Count"] / final_summary["Count"].sum() * 100).round(2)

print(f"\nFinal Version Distribution:")
print("="*50)
print(final_summary.to_string(index=False))

# %% [markdown]
# # Asignar cada muestra al cluster que pertenece

# %%
version_df['Versión'] = version_df['Versión'].map(final_mapping)

summary = version_df['Versión'].value_counts().reset_index()
summary.columns = ['Version', 'Count']
summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)

print(summary.to_string(index=False))


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


