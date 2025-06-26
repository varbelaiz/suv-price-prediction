import pandas as pd
import numpy as np
import re

from pathlib import Path
from unicodedata import normalize
from typing import List, Protocol, Dict, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
TRAIN_INPUT_FILE = Path("data/train/cleaned_train.csv")
TRAIN_OUTPUT_FILE = Path("data/train/transformed_train.csv")

TEST_INPUT_FILE = Path("data/test/cleaned_test.csv")
TEST_OUTPUT_FILE = Path("data/test/transformed_test.csv")

VERSIONS_FILE = "data/mercadolibre_versions.csv"  # optional catalogue
SIMILARITY_THRESHOLD = 0.4

COLUMNS_ORDER = [
    "idx",
    "Marca",
    "Modelo",
    "Versión",
    "Título",
    "Motor",
    "Turbo",
    "cv",
    "Tracción",
    "Color",
    "Tipo de combustible",
    "Puertas",
    "Transmisión",
    "Con cámara de retroceso",
    "Kilómetros",
    "Precio",
    "Moneda",
    "Descripción",
    "Tipo de vendedor",
]


class Transformer(Protocol):
    
    def fit(self, df: pd.DataFrame) -> "Transformer": ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)



class VersionClustering:

    def __init__(self, versions_path: str | None, thr: float, verbose: bool = True):
        self.versions_path = versions_path                    # catálogo opcional
        self.thr = thr                                        # umbral de similitud
        self.verbose = verbose
        self._mapping: Dict[Tuple[str, str, str], str] = {}   # lookup exacto
        self._canon_by_bm: Dict[Tuple[str, str], List[str]] = {}  # lista de canónicas
        self._vect = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=True)

    def fit(self, df: pd.DataFrame):

        before = df["Versión"].nunique()
        
        # First pass: collect all versions to fit a global vectorizer
        all_versions = [self._clean(v) for v in df["Versión"].dropna().unique()]
        all_versions = [v for v in all_versions if v]
        
        if all_versions:
            self._vect.fit(all_versions)
            if self.verbose:
                print(f"VersionClustering: fitted vectorizer on {len(all_versions)} unique versions")
        
        processed: List[pd.DataFrame] = []

        for (brand, model), g in df.groupby(["Marca", "Modelo"]):
            clustered = self._cluster_group(g, brand, model)
            
            processed.append(clustered)
            
            # guardamos las versiones canónicas para el fallback
            self._canon_by_bm[(brand, model)] = clustered["version_canon"].unique().tolist()

        all_clustered = pd.concat(processed, ignore_index=True)

        # construimos el diccionario de lookup exacto
        for _, row in all_clustered.iterrows():
            key = (row["Marca"], row["Modelo"], row["Versión"])
            self._mapping[key] = row["version_canon"]

        after = all_clustered["version_canon"].nunique()

        if self.verbose:
            print(f"VersionClustering: unique versions reduced from {before} to {after}")

            
        return self

    def transform(self, df: pd.DataFrame):

        if not self._mapping:
            raise RuntimeError("VersionClustering must be fitted before transform().")

        def map_or_fallback(row):
            key = (row["Marca"], row["Modelo"], row["Versión"])
            # 1️⃣  Lookup exacto
            if key in self._mapping:
                return self._mapping[key]

            # 2️⃣  Fallback: versión nueva → buscamos la más parecida entre canónicas
            canon_list = self._canon_by_bm.get((row["Marca"], row["Modelo"]), [])
            if not canon_list:
                return row["Versión"]  # sin referencia para comparar

            cleaned_orig = self._clean(row["Versión"])
            best_score, best_match = 0.0, row["Versión"]

            for canon in canon_list:
                X = self._vect.fit_transform([cleaned_orig, self._clean(canon)]) if cleaned_orig else None
                score = cosine_similarity(X[0:1], X[1:2])[0][0] if X is not None else 0
                if score > best_score:
                    best_score, best_match = score, canon
            return best_match if best_score >= self.thr else row["Versión"]

        out = df.copy()
        out["Versión"] = out.apply(map_or_fallback, axis=1)
        return out

    # -------------------------- INTERNALS --------------------------------
    def _cluster_group(self, group: pd.DataFrame, brand: str, model: str) -> pd.DataFrame:
        """Two-stage clustering: catalogue matching followed by DBSCAN for unmatched versions."""
        catalogue = self._get_catalogue(brand, model)
        
        if not catalogue:
            # No catalogue available, use DBSCAN only
            return self._dbscan_cluster(group)
        
        # Stage 1: Apply catalogue-based clustering
        stage1_result = self._string_similarity_cluster(group, catalogue)
        
        # Stage 2: Apply DBSCAN to versions that weren't matched to catalogue
        # (i.e., versions where version_canon == original Versión)
        unmatched_mask = stage1_result["version_canon"] == stage1_result["Versión"]
        
        if not unmatched_mask.any():
            # All versions were successfully matched to catalogue
            return stage1_result
        
        # Extract unmatched versions for DBSCAN clustering
        unmatched_group = stage1_result[unmatched_mask].copy()
        matched_group = stage1_result[~unmatched_mask].copy()
        
        if len(unmatched_group) <= 1:
            # Too few unmatched versions to cluster
            return stage1_result
        
        # Apply DBSCAN to unmatched versions
        unmatched_clustered = self._dbscan_cluster(unmatched_group)
        
        # Combine matched and newly clustered results
        final_result = pd.concat([matched_group, unmatched_clustered], ignore_index=True)
        
        return final_result[["Marca", "Modelo", "Versión", "version_canon"]]

    def _get_catalogue(self, brand: str, model: str) -> List[str]:
        if self.versions_path and Path(self.versions_path).exists():
            cat = pd.read_csv(self.versions_path)
            return cat[(cat["Brand"] == brand) & (cat["Model"] == model)]["Version"].dropna().tolist()
        return []

    def _dbscan_cluster(self, group):

        grp = group.copy()
        grp["_clean"] = grp["Versión"].apply(self._clean)
        grp = grp[grp["_clean"] != ""].reset_index(drop=True)
        
        if grp.empty:
            grp["version_canon"] = "Unassigned"
            return grp[["Marca", "Modelo", "Versión", "version_canon"]]

        # Use the pre-fitted global vectorizer for richer feature space
        X = self._vect.transform(grp["_clean"])
        clusters = DBSCAN(eps=0.3, min_samples=max(2, grp.shape[0] // 4), metric="cosine").fit_predict(X)
        grp["_cluster"] = clusters

        canon = grp[grp["_cluster"] != -1].groupby("_cluster")["Versión"].agg(lambda s: s.value_counts().idxmax())
        grp["version_canon"] = grp["_cluster"].map(canon).fillna("Unassigned")
        return grp[["Marca", "Modelo", "Versión", "version_canon"]]

    def _string_similarity_cluster(self, group, catalogue):

        mapping: Dict[str, str] = {}
        
        for original in group["Versión"].dropna().unique():
            cleaned_orig = self._clean(original)
            if not cleaned_orig:
                mapping[original] = original
                continue
                
            best_score, best_match = 0.0, original
            
            # Transform original version using the pre-fitted global vectorizer
            orig_vec = self._vect.transform([cleaned_orig])
            
            for real in catalogue:
                cleaned_real = self._clean(real)
                if not cleaned_real:
                    continue
                    
                # Transform catalogue version using the pre-fitted global vectorizer
                real_vec = self._vect.transform([cleaned_real])
                score = cosine_similarity(orig_vec, real_vec)[0][0]
                
                if score > best_score:
                    best_score, best_match = score, real
                    
            mapping[original] = best_match if best_score >= self.thr else original

        grp = group.copy()
        grp["version_canon"] = grp["Versión"].map(mapping)
        return grp[["Marca", "Modelo", "Versión", "version_canon"]]


    @staticmethod
    def _clean(s):
        """Normaliza texto: lower, ASCII, solo [a‑z0‑9 ]."""
        if pd.isna(s):
            return ""
        if not isinstance(s, str):
            s = str(s)
        txt = normalize("NFKD", s.lower()).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9 ]+", " ", txt).strip()



class DescriptionEmbeddings:
    """
    Transform text descriptions into low-dimensional numerical features using TF-IDF + PCA.
    
    This is a standard approach for text feature engineering that:
    1. Converts text to TF-IDF vectors (captures word importance)
    2. Applies PCA to reduce dimensionality 
    3. Adds the reduced features as new columns to the dataframe
    """
    
    def __init__(self, 
                 text_column: str = "Descripción",
                 n_components: int = 20,
                 max_features: int = 1000,
                 min_df: int = 2,
                 max_df: float = 0.8,
                 verbose: bool = True):
        """
        Args:
            text_column: Column containing text descriptions
            n_components: Number of PCA components (final feature count)
            max_features: Maximum number of TF-IDF features before PCA
            min_df: Ignore terms that appear in fewer than min_df documents
            max_df: Ignore terms that appear in more than max_df fraction of documents
            verbose: Whether to print progress information
        """
        self.text_column = text_column
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.verbose = verbose
        
        # Initialize components
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=None,  # Spanish stop words could be added here
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        
        # Will store feature names for the output columns
        self.feature_names = [f"desc_pca_{i+1}" for i in range(n_components)]
        
    def fit(self, df: pd.DataFrame):
        """Fit TF-IDF vectorizer and PCA on the training data."""
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in dataframe")
        
        # Prepare text data
        texts = df[self.text_column].fillna("").astype(str)
        
        if self.verbose:
            print(f"DescriptionEmbeddings: Processing {len(texts)} descriptions")
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        if self.verbose:
            print(f"DescriptionEmbeddings: TF-IDF created {tfidf_matrix.shape[1]} features")
        
        # Fit scaler and PCA
        tfidf_scaled = self.scaler.fit_transform(tfidf_matrix.toarray())
        self.pca.fit(tfidf_scaled)
        
        if self.verbose:
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"DescriptionEmbeddings: PCA with {self.n_components} components explains {explained_variance:.3f} of variance")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform descriptions to PCA features and add them to the dataframe."""
        if not hasattr(self.tfidf, 'vocabulary_'):
            raise RuntimeError("DescriptionEmbeddings must be fitted before transform()")
        
        # Prepare text data
        texts = df[self.text_column].fillna("").astype(str)
        
        # Transform through the pipeline: TF-IDF -> Scale -> PCA
        tfidf_matrix = self.tfidf.transform(texts)
        tfidf_scaled = self.scaler.transform(tfidf_matrix.toarray())
        pca_features = self.pca.transform(tfidf_scaled)
        
        # Create dataframe with PCA features
        pca_df = pd.DataFrame(pca_features, columns=self.feature_names, index=df.index)
        
        # Concatenate with original dataframe
        result = pd.concat([df, pca_df], axis=1)
        
        return result


class FillNaNs:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self._modes: pd.Series | None = None

    def fit(self, df: pd.DataFrame):
        base = ["Marca", "Modelo"]
        ver = "Versión"
        grp = df.groupby(base + [ver])
        self._modes = grp[self.columns].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        return self

    def transform(self, df: pd.DataFrame):

        if self._modes is None:
            raise RuntimeError("FillNaNs must be fitted before transform().")

        base = ["Marca", "Modelo"]
        ver = "Versión"

        def impute(row, col):
            key = tuple(row[c] for c in base + [ver])
            return self._modes.loc[key, col] if key in self._modes.index else row[col]

        out = df.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = out.apply(lambda r: r[col] if pd.notna(r[col]) else impute(r, col), axis=1)
        return out



class DropColumns:
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        return self  

    def transform(self, df: pd.DataFrame):
        return df.drop(columns=[c for c in self.columns if c in df.columns])


class ReorderColumns:
    def __init__(self, order: List[str]):
        self.order = order

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame):
        existing = [c for c in self.order if c in df.columns]
        remaining = [c for c in df.columns if c not in existing]
        return df[existing + remaining]



class Pipeline:
    def __init__(self, steps: List[Transformer]):
        self.steps = steps

    def fit(self, df: pd.DataFrame):
        tmp = df
        for step in self.steps:
            tmp = step.fit(tmp).transform(tmp) if hasattr(step, "fit") else step.transform(tmp)
        return self

    def transform(self, df: pd.DataFrame):
        tmp = df
        for step in self.steps:
            tmp = step.transform(tmp)
        return tmp

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)




def build_pipeline(verbose: bool = True) -> Pipeline:
    return Pipeline([
        VersionClustering(VERSIONS_FILE, SIMILARITY_THRESHOLD, verbose),
        DescriptionEmbeddings(verbose=verbose),
        FillNaNs(["cv", "Motor", "Tracción", "Turbo"]),
        DropColumns(["Tipo de carrocería"]),
        ReorderColumns(COLUMNS_ORDER),
    ])




def transform_datasets(apply_to_test: bool = False, verbose: bool = True):
    pipe = build_pipeline(verbose)

    # ---- Train ----
    train_df = pd.read_csv(TRAIN_INPUT_FILE)

    # Apply transformations and sort rows by Marca and Modelo before saving
    train_out = (
        pipe.fit_transform(train_df)
            .sort_values(by=["Marca", "Modelo"], ascending=[True, True])
            .reset_index(drop=True)
    )
    train_out.to_csv(TRAIN_OUTPUT_FILE, index=False)

    # ---- Test ----
    if apply_to_test:
        test_df = pd.read_csv(TEST_INPUT_FILE)
        test_out = (
            pipe.transform(test_df)
                .sort_values(by=["Marca", "Modelo"], ascending=[True, True])
                .reset_index(drop=True)
        )
        test_out.to_csv(TEST_OUTPUT_FILE, index=False)
        if verbose:
            print("Test set transformed without leakage.")


if __name__ == "__main__":
    transform_datasets(apply_to_test=True, verbose=True)
