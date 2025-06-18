import pandas as pd
import numpy as np
import re
from pathlib import Path
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = Path("data/train/cleaned_train.csv")
OUTPUT_FILE = Path("data/train/transformed_train.csv")

COLUMNS_ORDER = [
    "idx", "Marca", "Modelo", "Versión", "Título", "Motor", "Turbo", "cv", 
    "Tracción", "Color", "Tipo de combustible", "Puertas", "Transmisión", 
    "Con cámara de retroceso", "Kilómetros", "Precio", "Moneda", 
    "Descripción", "Tipo de vendedor"
]


def clean_version(text: str) -> str:
    if pd.isna(text):
        return ""
    text = normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonicalize_with_dbscan(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["_clean"] = group["Versión"].apply(clean_version)
    group = group[group["_clean"] != ""].reset_index(drop=True)
    if group.empty:
        group["version_canon"] = "Unassigned"
        return group
    vect = CountVectorizer(analyzer="char", ngram_range=(1, 3))
    X = vect.fit_transform(group["_clean"]).toarray()
    eps = 0.3
    min_samples = max(2, group["_clean"].nunique() // 4)
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X)
    group["_cluster"] = clusters
    canon_map = (
        group[group["_cluster"] != -1]
        .groupby("_cluster")["Versión"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    group["version_canon"] = group["_cluster"].map(canon_map).fillna("Unassigned")
    return group.drop(columns=["_clean", "_cluster"])


def _canonicalize_with_csv(group: pd.DataFrame, versions_path: str, thr: float) -> pd.DataFrame:

    
    ml_versions = pd.read_csv(versions_path)

    brand = group["Marca"].iloc[0]
    model = group["Modelo"].iloc[0]
    real = ml_versions[(ml_versions["Brand"] == brand) & (ml_versions["Model"] == model) & (ml_versions["Version"] != "N/A")]
    real_versions = real["Version"].tolist()

    if not real_versions:
        print(f"No real versions found for {brand} {model}")
        return _canonicalize_with_dbscan(group)
    
    vect = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=True)
    mapping = {}
    for original in group["Versión"].unique():
        best_score = 0.0
        best_match = original
        cleaned_original = clean_version(original)
        for rv in real_versions:
            cleaned_rv = clean_version(rv)
            if cleaned_original and cleaned_rv:
                X = vect.fit_transform([cleaned_original, cleaned_rv])
                score = cosine_similarity(X[0:1], X[1:2])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = rv
        mapping[original] = best_match if best_score >= thr else original
    group = group.copy()
    group["version_canon"] = group["Versión"].map(mapping)
    return group


def cluster_versions(
    group: pd.DataFrame,
    method: str = "dbscan",
    ml_versions_path: str = "data/mercadolibre_versions.csv",
    threshold: float = 0.3,
) -> pd.DataFrame:
    if method == "csv":
        return _canonicalize_with_csv(group, ml_versions_path, threshold)
    return _canonicalize_with_dbscan(group)


def fill_by_mode(df: pd.DataFrame, col: str):
    modes = df.groupby(["Marca", "Modelo", "version_canon"])[col].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    df[col] = df[col].fillna(modes)


def main():
    df = pd.read_csv(INPUT_FILE)

    # Analizar y unificar versiones
    before_unique = df["Versión"].nunique()
    print("Records", len(df), "unique versions", before_unique)

    grouped = df.groupby(["Marca", "Modelo"])
    processed = [cluster_versions(g, method='csv') for _, g in grouped]

    full_df = pd.concat(processed, ignore_index=True)
    full_df["Versión"] = full_df["version_canon"]
    after_unique = full_df["Versión"].nunique()
    print("After", len(full_df), "unique", after_unique)
    
    # Imputar valores faltantes
    for col in ["cv", "Motor", "Tracción", "Turbo"]:
        if col in full_df.columns:
            fill_by_mode(full_df, col)
    full_df.drop(columns=["version_canon"], inplace=True)
    
    # Borrar "Tipo de carrocería" (redundante)
    if "Tipo de carrocería" in full_df.columns:
        full_df.drop(columns=["Tipo de carrocería"], inplace=True)
        print("Deleted 'Tipo de carrocería' column")
    
    
    # Only include columns that actually exist in the dataframe
    existing_columns = [col for col in COLUMNS_ORDER if col in full_df.columns]
    
    remaining_columns = [col for col in full_df.columns if col not in existing_columns]

    final_column_order = existing_columns + remaining_columns
    
    full_df = full_df[final_column_order]
    
    full_df.to_csv(OUTPUT_FILE, index=False)
    print("Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
