import re
import numpy as np
import pandas as pd
from pathlib import Path
from unicodedata import normalize
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(txt: str) -> str:
    if pd.isna(txt):
        return ""
    txt = normalize("NFKD", txt.lower()).encode("ascii", "ignore").decode("ascii")
    txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def turbo_process(col: pd.Series) -> pd.DataFrame:
    patt = re.compile(r"\b(turbo|tsi|biturbo|bi turbo|tci|t\/?\d)\b", re.I)
    turbo_flag = col.str.contains(patt, na=False).astype(int)
    motor_clean = col.fillna("").str.replace(patt, "", regex=True).str.strip()
    return pd.DataFrame({"Motor": motor_clean, "Turbo": turbo_flag})


def cluster_versions(grp: pd.DataFrame, eps: float = 0.3) -> pd.DataFrame:
    uniq = grp["version_clean"].nunique()
    if uniq <= 1 or grp["version_clean"].str.len().max() < 3:
        grp["Versión"] = grp["Versión"].fillna("Unassigned")
        return grp
    vec = CountVectorizer(analyzer="char", ngram_range=(1, 5))
    X = vec.fit_transform(grp["version_clean"]).toarray()
    db = DBSCAN(eps=eps, min_samples=max(2, uniq // 4), metric="cosine")
    grp["cluster"] = db.fit_predict(X)
    canon = (
        grp[grp["cluster"] != -1]
        .groupby("cluster")["Versión"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    grp["version_canon"] = grp["cluster"].map(canon)
    grp.loc[grp["cluster"] == -1, "version_canon"] = "Unassigned"
    grp["Versión"] = grp["version_canon"]
    return grp.drop(columns=["cluster", "version_canon"])


def fill_mode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        mode_vals = (
            df.groupby(["Marca", "Modelo", "Versión"])[c]
            .transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
        df[c] = df[c].fillna(mode_vals)
    return df


def main() -> None:
    src = Path("data/train/cleaned_train.csv")
    dst = Path("data/train/transformed_train.csv")
    df = pd.read_csv(src)
    df[["Motor", "Turbo"]] = turbo_process(df["Motor"])
    df["version_clean"] = df["Versión"].apply(clean_text)
    df = df[df["version_clean"] != ""].reset_index(drop=True)
    df = (
        df.groupby(["Marca", "Modelo"], group_keys=False, include_groups=True)
        .apply(cluster_versions)
    )
    df.drop(columns=["version_clean"], inplace=True, errors="ignore")
    df = fill_mode(df, ["cv", "Motor", "Tracción", "Turbo"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


if __name__ == "__main__":
    main()
