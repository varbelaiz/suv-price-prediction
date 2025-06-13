import pandas as pd
import numpy as np
import re
from pathlib import Path
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN

INPUT_FILE = Path("data/train/cleaned_train.csv")
OUTPUT_FILE = Path("data/train/transformed_train.csv")

def clean_version(s):
    s = "" if pd.isna(s) else s
    s = normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def cluster_versions(group):
    group = group.copy()
    group["version_clean"] = group["Versión"].apply(clean_version)
    group = group[group["version_clean"] != ""].reset_index(drop=True)
    
    if len(group) == 0:
        group["version_canon"] = ["Unassigned"] * len(group)
        return group
    
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 5))
    try:
        X = vectorizer.fit_transform(group["version_clean"]).toarray()
    except ValueError:
        group["version_canon"] = ["Unassigned"] * len(group)
        return group

    n_versions = group["version_clean"].nunique()
    db = DBSCAN(eps=0.3, min_samples=max(2, n_versions // 4), metric="cosine")
    group["cluster"] = db.fit_predict(X)

    canon = (
        group[group["cluster"] != -1]
            .groupby("cluster")["Versión"]
            .agg(lambda s: s.value_counts().idxmax())
    )
    group["version_canon"] = group["cluster"].map(canon)
    group["version_canon"] = group["version_canon"].fillna("Unassigned").astype(str)
    return group


def fill_by_mode(df, col):
    mode_vals = (
        df.groupby(["Marca", "Modelo", "version_canon"])[col]
          .transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    )
    df[col] = df[col].fillna(mode_vals)


def main():
    df = pd.read_csv(INPUT_FILE)
    
    # Print initial statistics
    initial_unique_versions = df['Versión'].nunique()
    initial_total_records = len(df)
    print(f"=== BEFORE PROCESSING ===")
    print(f"Total records: {initial_total_records}")
    print(f"Unique versions: {initial_unique_versions}")
    print()
    
    grouped = df.groupby(["Marca", "Modelo"])
    clustered_dfs = [cluster_versions(group) for _, group in grouped]
    full_df = pd.concat(clustered_dfs, ignore_index=True)

    full_df["Versión"] = full_df["version_canon"]
    
    # Print statistics after clustering
    final_unique_versions = full_df['Versión'].nunique()
    unassigned_count = (full_df['Versión'] == 'Unassigned').sum()
    unassigned_percentage = (unassigned_count / len(full_df)) * 100
    
    print(f"=== AFTER PROCESSING ===")
    print(f"Total records: {len(full_df)}")
    print(f"Unique versions after clustering: {final_unique_versions}")
    print(f"Unassigned versions: {unassigned_count} ({unassigned_percentage:.2f}%)")
    print(f"Version reduction: {initial_unique_versions} → {final_unique_versions} ({initial_unique_versions - final_unique_versions} fewer unique versions)")
    print()

    for col in ["cv", "Motor", "Tracción", "Turbo"]:
        if col in full_df.columns:
            fill_by_mode(full_df, col)

    full_df.drop(columns=["version_clean", "cluster", "version_canon"], inplace=True, errors="ignore")

    full_df.to_csv(OUTPUT_FILE, index=False)
    print("Clustering finalizado. Archivo guardado en:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
