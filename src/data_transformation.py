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
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
TRAIN_INPUT_FILE = Path("data/train/cleaned_train.csv")
TRAIN_OUTPUT_FILE = Path("data/train/transformed_train.csv")

VAL_OUTPUT_FILE = Path("data/train/transformed_val.csv")

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
    "Color",
    "Puertas",
    "Kilómetros",
    "Precio",
    "Descripción",
    "Tipo de vendedor",
    # Note: One-hot encoded columns (Tracción, Tipo de combustible, Transmisión, 
    # Con cámara de retroceso, Moneda) will be added dynamically after the base columns
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
                print(f"\t VERSION CLUSTERING: fitted vectorizer on {len(all_versions)} unique versions")
        
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
            print(f"\t VERSION CLUSTERING: unique versions reduced from {before} to {after}")

            
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
                cleaned_canon = self._clean(canon)
                if not cleaned_canon:
                    continue
                    
                # Transform both versions using the pre-fitted global vectorizer
                X = self._vect.transform([cleaned_orig, cleaned_canon]) if cleaned_orig else None
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
                 n_components: int = 30,
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
            print(f"\t DESCRIPTION EMBEDDINGS: Processing {len(texts)} descriptions")
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        if self.verbose:
            print(f"\t DESCRIPTION EMBEDDINGS: TF-IDF created {tfidf_matrix.shape[1]} features")
        
        # Fit scaler and PCA
        tfidf_scaled = self.scaler.fit_transform(tfidf_matrix.toarray())
        self.pca.fit(tfidf_scaled)
        
        if self.verbose:
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"\t DESCRIPTION EMBEDDINGS: PCA with {self.n_components} components explains {explained_variance:.3f} of variance")
        
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





class CurrencyConverter:
    """
    Convert prices to a single currency.
    Multiplies USD prices by the specified exchange rate.
    """
    
    def __init__(self, 
                 price_column: str = "Precio",
                 currency_column: str = "Moneda", 
                 usd_to_target_rate: float = 1100.0,
                 verbose: bool = True):
        """
        Args:
            price_column: Column containing prices
            currency_column: Column containing currency information
            usd_to_target_rate: Exchange rate from USD to target currency
            verbose: Whether to print progress information
        """
        self.price_column = price_column
        self.currency_column = currency_column
        self.usd_to_target_rate = usd_to_target_rate
        self.verbose = verbose
        
    def fit(self, df: pd.DataFrame):
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert USD prices to target currency."""

        result = df.copy()
        
        # Find US$ prices
        usd_mask = (result[self.currency_column].astype(str) == 'US$')
        usd_count = usd_mask.sum()
        
        if usd_count > 0:
            # Convert US$ prices
            original_prices = result.loc[usd_mask, self.price_column].copy()
            result.loc[usd_mask, self.price_column] = original_prices * self.usd_to_target_rate
            
            if self.verbose:
                print(f"\t CURRENCY CONVERTER: Converted {usd_count} US$ prices")
        else:
            if self.verbose:
                print(f"\t CURRENCY CONVERTER: No US$ prices found to convert")
        
        return result


class OneHotEncoder:
    """
    One-Hot Encoder for categorical variables.
    Creates binary columns for each category and drops the original columns.
    Handles unseen categories by creating an 'Unknown' category.
    """
    
    def __init__(self, categorical_columns: List[str], verbose: bool = True):
        """
        Args:
            categorical_columns: List of categorical columns to encode
            verbose: Whether to print progress information
        """
        self.categorical_columns = categorical_columns
        self.verbose = verbose
        
        # Will store the categories for each column
        self._categories: Dict[str, List[str]] = {}
        self._feature_names: List[str] = []
        
    def fit(self, df: pd.DataFrame):
        """Fit one-hot encoder on training data."""
        self._categories = {}
        self._feature_names = []
        
        for col in self.categorical_columns:
            if col not in df.columns:
                if self.verbose:
                    print(f"\t ONE-HOT ENCODER: Column '{col}' not found, skipping")
                continue
            
            # Get unique categories, excluding NaN values
            categories = df[col].dropna().unique().tolist()
            categories = sorted([str(cat) for cat in categories])  # Convert to string and sort
            
            self._categories[col] = categories
            
            # Create feature names for this column
            for category in categories:
                feature_name = f"{col}_{category}"
                self._feature_names.append(feature_name)
            
            if self.verbose:
                print(f"\t ONE-HOT ENCODER: Found {len(categories)} categories for '{col}'")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using one-hot encoding."""
        if not self._categories:
            raise RuntimeError("OneHotEncoder must be fitted before transform()")
        
        result = df.copy()
        
        for col in self.categorical_columns:
            if col not in result.columns:
                continue
                
            if col not in self._categories:
                continue
            
            categories = self._categories[col]
            
            # Create binary columns for each category
            for category in categories:
                feature_name = f"{col}_{category}"
                # Create binary column: 1 if matches category, 0 otherwise
                result[feature_name] = (result[col].astype(str) == str(category)).astype(int)
            
            # Handle unseen categories by checking if any row has all zeros
            category_cols = [f"{col}_{cat}" for cat in categories]
            row_sums = result[category_cols].sum(axis=1)
            unseen_mask = (row_sums == 0) & result[col].notna()
            
            if unseen_mask.any():
                unseen_count = unseen_mask.sum()
                if self.verbose:
                    print(f"\t ONE-HOT ENCODER: {unseen_count} unseen categories in '{col}' handled")
                
                # Create an 'Unknown' category column if there are unseen values
                unknown_col = f"{col}_Unknown"
                if unknown_col not in result.columns:
                    result[unknown_col] = 0
                result.loc[unseen_mask, unknown_col] = 1
        
        # Drop the original categorical columns
        columns_to_drop = [col for col in self.categorical_columns if col in result.columns]
        result = result.drop(columns=columns_to_drop)
        
        return result


class OrderColumns:
    """Reorder columns and sort dataframe, then reset index."""
    
    def __init__(self, 
                 column_order: List[str] = None, 
                 sort_columns: List[str] = ["Marca", "Modelo"]):
        self.column_order = column_order or COLUMNS_ORDER
        self.sort_columns = sort_columns

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame):
        result = df.copy()
        
        # First: Reorder columns
        existing_ordered_cols = [c for c in self.column_order if c in result.columns]
        remaining_cols = [c for c in result.columns if c not in existing_ordered_cols]
        result = result[existing_ordered_cols + remaining_cols]
        
        # Second: Sort and reset index
        existing_sort_cols = [c for c in self.sort_columns if c in result.columns]
        if existing_sort_cols:
            result = result.sort_values(by=existing_sort_cols, ascending=True).reset_index(drop=True)
        else:
            result = result.reset_index(drop=True)
            
        return result


class TargetEncoder:
    """
    Target encoder for categorical variables.
    Replaces categorical string values with their target-encoded numeric values
    (mean of the target variable for each category).
    Uses smoothing to handle categories with few samples and unseen categories.
    """
    
    def __init__(self, 
                 categorical_columns: List[str], 
                 target_column: str = "Precio",
                 smoothing: float = 10.0,
                 verbose: bool = True):
        """
        Args:
            categorical_columns: List of categorical columns to encode
            target_column: Target variable column name
            smoothing: Smoothing parameter for regularization (higher = more smoothing)
            verbose: Whether to print progress information
        """
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.smoothing = smoothing
        self.verbose = verbose
        
        # Will store the encoding mappings for each column
        self._encodings: Dict[str, Dict] = {}
        self._global_mean: float = 0.0
        
    def fit(self, df: pd.DataFrame):
        """Fit target encoders on training data."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")
        
        # Calculate global mean for smoothing and unseen categories
        self._global_mean = df[self.target_column].mean()
        
        if self.verbose:
            print(f"\t TARGET ENCODER: Global mean target value: {self._global_mean:.2f}")
        
        # Fit encoder for each categorical column
        for col in self.categorical_columns:
            if col not in df.columns:
                if self.verbose:
                    print(f"\t TARGET ENCODER: Column '{col}' not found, skipping")
                continue
                
            # Debug: Show unique values count
            unique_count = df[col].nunique()

                
            # Calculate mean target and count for each category
            stats = df.groupby(col)[self.target_column].agg(['mean', 'count']).reset_index()
            
            # Apply smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_mean = ((stats['count'] * stats['mean'] + 
                            self.smoothing * self._global_mean) / 
                           (stats['count'] + self.smoothing))
            
            # Create encoding dictionary
            encoding_dict = dict(zip(stats[col], smoothed_mean))
            self._encodings[col] = encoding_dict
            
            if self.verbose:
                print(f"\t TARGET ENCODER: Encoded {len(encoding_dict)} categories for '{col}'")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted target encodings."""
        if not self._encodings:
            raise RuntimeError("TargetEncoder must be fitted before transform()")
        
        result = df.copy()
        
        for col in self.categorical_columns:
            if col not in result.columns:
                continue
                
            if col not in self._encodings:
                continue
                
            # Map categories to their target-encoded values
            # Use global mean for unseen categories
            encoding_dict = self._encodings[col]
            original_values = result[col].copy()
            
            # Debug: Show transform info
            transform_unique_count = result[col].nunique()
            if self.verbose:
                print(f"\t TARGET ENCODER: Transforming {transform_unique_count} unique values for '{col}'")
            
            result[col] = result[col].map(encoding_dict).fillna(self._global_mean)
            
            if self.verbose:
                unseen_count = original_values.map(encoding_dict).isna().sum()
                if unseen_count > 0:
                    print(f"\t TARGET ENCODER: {unseen_count} unseen categories in '{col}' filled with global mean")
                    
        
        return result


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
        tmp = df
        for step in self.steps:
            if hasattr(step, "fit"):
                step.fit(tmp)
                tmp = step.transform(tmp)
            else:
                tmp = step.transform(tmp)
        return tmp




def build_pipeline(verbose: bool = True) -> Pipeline:
    return Pipeline([
        VersionClustering(VERSIONS_FILE, SIMILARITY_THRESHOLD, verbose),
        DescriptionEmbeddings(verbose=verbose),
        FillNaNs(["cv", "Motor", "Tracción", "Turbo"]),
        TargetEncoder(["Marca", "Modelo", "Versión"], verbose=verbose),
        CurrencyConverter(verbose=verbose),
        OneHotEncoder(["Tracción", "Tipo de combustible", "Transmisión", "Con cámara de retroceso", "Moneda", "Tipo de vendedor"], verbose=verbose),
        DropColumns(["Tipo de carrocería", "Título", "Descripción", "Color"]),
        OrderColumns(column_order=COLUMNS_ORDER),
    ])


def transform_datasets(apply_to_test: bool = False, val_size: float = 0.2, verbose: bool = True):
    pipe = build_pipeline(verbose)

    # Load the full training data
    full_train_df = pd.read_csv(TRAIN_INPUT_FILE)
    print("Nans in Versión:", full_train_df["Versión"].isna().sum())
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        full_train_df, 
        test_size=val_size, 
        random_state=42,
        stratify=None  # Could stratify by price ranges if needed
    )
    
    print(f"Dataset splits:")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Validation: {len(val_df)} samples")
    
    if apply_to_test:
        test_df = pd.read_csv(TEST_INPUT_FILE)
        print(f"  - Test: {len(test_df)} samples")
    
    # ---- Train: Fit and transform ----
    print("-" * 20, "Train (Fit + Transform)", "-" * 20)
    train_out = pipe.fit_transform(train_df)
    train_out.to_csv(TRAIN_OUTPUT_FILE, index=False)
    
    # ---- Validation: Transform only ----
    print("-" * 20, "Validation (Transform)", "-" * 20)
    val_out = pipe.transform(val_df)
    val_out.to_csv(VAL_OUTPUT_FILE, index=False)

    # ---- Test: Transform only ----
    if apply_to_test:
        print("-" * 20, "Test (Transform)", "-" * 20)
        test_out = pipe.transform(test_df)
        test_out.to_csv(TEST_OUTPUT_FILE, index=False)
        if verbose:
            print("All datasets transformed successfully!")
    else:
        if verbose:
            print("Train and validation datasets transformed successfully!")


if __name__ == "__main__":
    transform_datasets(apply_to_test=True, val_size=0.2, verbose=True)
