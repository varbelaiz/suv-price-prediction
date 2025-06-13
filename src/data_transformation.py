import pandas as pd
import re
from unicodedata import normalize
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN


def clean_version(text: str) -> str:
    if pd.isna(text):
        return ""
    text = normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def cluster_group(group: pd.DataFrame, eps: float = 0.3) -> pd.DataFrame:
    if group["version_clean"].nunique() <= 1:
        return group

    # al menos una versión debe tener 3+ caracteres para n-grams 3-5
    if group["version_clean"].str.len().max() < 3:
        return group

    vec = CountVectorizer(analyzer="char", ngram_range=(1, 5))
    try:
        X = vec.fit_transform(group["version_clean"]).toarray()
    except ValueError:          # empty vocabulary
        return group

    n_versions = group["version_clean"].nunique()
    min_samples = max(2, n_versions // 4)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    group["cluster"] = db.fit_predict(X)

    canon = (
        group.groupby("cluster")["Versión"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    group["version_canon"] = group["cluster"].map(canon)

    noise = group["cluster"] == -1
    group.loc[noise, "version_canon"] = group.loc[noise, "Versión"]

    group["Versión"] = group["version_canon"]
    return group.drop(columns=["cluster", "version_canon"])



def main() -> None:
    src_path = Path("data/train/cleaned_train.csv")
    df = pd.read_csv(src_path)
    df["version_clean"] = df["Versión"].apply(clean_version)
    df = df[df["version_clean"] != ""].reset_index(drop=True)
    df = (
    df.groupby(["Marca", "Modelo"], group_keys=False)
      .apply(cluster_group, include_groups=True)
    )
    df.drop(columns=["version_clean"], inplace=True)
    df.to_csv("data/train/clean_transformed.csv", index=False)


if __name__ == "__main__":
    main()
