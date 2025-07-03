from unicodedata import normalize
from typing import List, Dict, Tuple
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import re
from pathlib import Path

# PyTorch imports for VAE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class VersionClustering:

    def __init__(self, versions_path: str | None, thr: float, verbose: bool = True):
        self.versions_path = versions_path                    # catálogo opcional
        self.thr = thr                                        # umbral de similitud
        self.verbose = verbose
        self._mapping: Dict[Tuple[str, str, str], str] = {}   # lookup exacto
        self._canon_by_bm: Dict[Tuple[str, str], List[str]] = {}  # lista de canónicas
        self._vect = CountVectorizer(analyzer="char", ngram_range=(2, 3), lowercase=True)

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

    def export_mapping(self) -> pd.DataFrame:
        """Export the mapping from (Marca, Modelo, Versión) to canonical version as a DataFrame."""
        rows = [
            {"Marca": k[0], "Modelo": k[1], "Version": k[2], "version_canon": v}
            for k, v in self._mapping.items()
        ]
        return pd.DataFrame(rows)

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
            grp["version_canon"] = grp["Versión"]  # Keep original version
            return grp[["Marca", "Modelo", "Versión", "version_canon"]]

        # Use the pre-fitted global vectorizer for richer feature space
        X = self._vect.transform(grp["_clean"])
        clusters = DBSCAN(eps=0.3, min_samples=max(2, grp.shape[0] // 10), metric="cosine").fit_predict(X)
        grp["_cluster"] = clusters

        canon = grp[grp["_cluster"] != -1].groupby("_cluster")["Versión"].agg(lambda s: s.value_counts().idxmax())
        
        # For unclustered versions (noise), keep the original version instead of "Unassigned"
        grp["version_canon"] = grp["_cluster"].map(canon).fillna(grp["Versión"])

        
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


class VAE(nn.Module):
    """Variational Autoencoder for dimensionality reduction."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None):
        super(VAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """VAE loss: reconstruction loss + KL divergence."""
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * kl_loss


class DescriptionEmbeddings:
    """
    TF-IDF + PCA embeddings for vehicle descriptions.
    """

    def __init__(
        self,
        text_column: str = "Descripción",
        n_components: int = 50,
        max_features: int = 2000,
        min_df: int = 2,
        max_df: float = 0.8,
        verbose: bool = True,
    ):
        self.text_column = text_column
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.verbose = verbose
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(2, 3),
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.feature_names = [f"embed_{i+1}" for i in range(n_components)]

    def fit(self, df: pd.DataFrame):
        if self.text_column not in df.columns:
            raise ValueError(f"Missing column {self.text_column}")
        
        texts = df[self.text_column].fillna("").astype(str).values
        tfidf = self.vectorizer.fit_transform(texts)
        
        if self.verbose:
            print(f"\tDESCRIPTION EMBEDDINGS: TF-IDF shape: {tfidf.shape}")
        
        # Convert to dense array and normalize
        tfidf_dense = tfidf.toarray()
        tfidf_scaled = self.scaler.fit_transform(tfidf_dense)
        
        # Fit PCA
        self.pca.fit(tfidf_scaled)
        
        if self.verbose:
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"\tDESCRIPTION EMBEDDINGS: PCA fitted with {self.n_components} components")
            print(f"\tDESCRIPTION EMBEDDINGS: Explained variance ratio: {explained_variance:.4f}")
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_column not in df.columns:
            raise ValueError(f"Missing column {self.text_column}")
        if self.pca is None:
            raise RuntimeError("DescriptionEmbeddings must be fitted before transform().")
        
        texts = df[self.text_column].fillna("").astype(str).values
        tfidf = self.vectorizer.transform(texts)
        
        # Convert to dense array and normalize
        tfidf_dense = tfidf.toarray()
        tfidf_scaled = self.scaler.transform(tfidf_dense)
        
        # Apply PCA transformation
        features = self.pca.transform(tfidf_scaled)
        
        features_df = pd.DataFrame(features, columns=self.feature_names, index=df.index)
        return pd.concat([df, features_df], axis=1)


class FillVersionNaNs:
    """
    Fill NaN values in the 'Versión' column with the most frequent version 
    for each Marca-Modelo combination.
    """
    
    def __init__(self, 
                 version_column: str = "Versión",
                 brand_column: str = "Marca", 
                 model_column: str = "Modelo",
                 verbose: bool = True):
        """
        Args:
            version_column: Column containing versions to fill
            brand_column: Brand column for grouping
            model_column: Model column for grouping  
            verbose: Whether to print progress information
        """
        self.version_column = version_column
        self.brand_column = brand_column
        self.model_column = model_column
        self.verbose = verbose
        
        # Will store the mode version for each brand-model combination
        self._version_modes: Dict[Tuple[str, str], str] = {}
        
    def fit(self, df: pd.DataFrame):
        """Fit by calculating the mode version for each brand-model combination."""
        if self.version_column not in df.columns:
            raise ValueError(f"Column '{self.version_column}' not found in dataframe")
        if self.brand_column not in df.columns:
            raise ValueError(f"Column '{self.brand_column}' not found in dataframe")
        if self.model_column not in df.columns:
            raise ValueError(f"Column '{self.model_column}' not found in dataframe")
        
        # Calculate mode version for each brand-model combination
        self._version_modes = {}
        
        for (brand, model), group in df.groupby([self.brand_column, self.model_column]):
            # Get non-null versions for this brand-model
            valid_versions = group[self.version_column].dropna()
            
            if len(valid_versions) > 0:
                # Find the most frequent version
                mode_version = valid_versions.mode()
                if len(mode_version) > 0:
                    self._version_modes[(brand, model)] = mode_version.iloc[0]
                else:
                    # If no clear mode, use the first version
                    self._version_modes[(brand, model)] = valid_versions.iloc[0]
        
        if self.verbose:
            filled_combinations = len(self._version_modes)
            print(f"\t VERSION NAN FILLER: Found mode versions for {filled_combinations} brand-model combinations")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN versions with the mode version for each brand-model."""
        if not self._version_modes:
            raise RuntimeError("FillVersionNaNs must be fitted before transform()")
        
        result = df.copy()
        
        # Find rows with NaN versions
        nan_mask = result[self.version_column].isna()
        nan_count_before = nan_mask.sum()
        
        if nan_count_before == 0:
            if self.verbose:
                print(f"\t VERSION NAN FILLER: No NaN versions found")
            return result
        
        # Fill NaN versions
        def fill_version(row):
            if pd.isna(row[self.version_column]):
                key = (row[self.brand_column], row[self.model_column])
                if key in self._version_modes:
                    return self._version_modes[key]
                else:
                    # No mode available for this brand-model, keep NaN
                    return row[self.version_column]
            else:
                return row[self.version_column]
        
        result[self.version_column] = result.apply(fill_version, axis=1)
        
        # Count how many were filled
        nan_count_after = result[self.version_column].isna().sum()
        filled_count = nan_count_before - nan_count_after
        
        if self.verbose:
            print(f"\t VERSION NAN FILLER: Filled {filled_count} NaN versions out of {nan_count_before}")
            if nan_count_after > 0:
                print(f"\t VERSION NAN FILLER: {nan_count_after} versions remain as NaN (no mode available)")
        
        return result


class FillNaNs:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self._modes_by_version: pd.Series | None = None  # Marca + Modelo + Versión
        self._modes_by_model: pd.Series | None = None    # Marca + Modelo (fallback)
        self._modes_by_brand: pd.Series | None = None    # Marca (fallback final)

    def fit(self, df: pd.DataFrame):
        base = ["Marca", "Modelo"]
        ver = "Versión"
        
        # Fit modes by Marca + Modelo + Versión
        grp_version = df.groupby(base + [ver])
        self._modes_by_version = grp_version[self.columns].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        
        # Fit modes by Marca + Modelo (fallback)
        grp_model = df.groupby(base)
        self._modes_by_model = grp_model[self.columns].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        
        # Fit modes by Marca (fallback final)
        grp_brand = df.groupby(["Marca"])
        self._modes_by_brand = grp_brand[self.columns].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        
        return self

    def transform(self, df: pd.DataFrame):
        if self._modes_by_version is None or self._modes_by_model is None or self._modes_by_brand is None:
            raise RuntimeError("FillNaNs must be fitted before transform().")

        base = ["Marca", "Modelo"]
        ver = "Versión"

        def impute(row, col):
            # Try Marca + Modelo + Versión first
            key_version = tuple(row[c] for c in base + [ver])
            if key_version in self._modes_by_version.index:
                mode_value = self._modes_by_version.loc[key_version, col]
                if pd.notna(mode_value):
                    return mode_value
            
            # Fallback to Marca + Modelo
            key_model = tuple(row[c] for c in base)
            if key_model in self._modes_by_model.index:
                mode_value = self._modes_by_model.loc[key_model, col]
                if pd.notna(mode_value):
                    return mode_value
            
            # Final fallback to Marca
            key_brand = row["Marca"]
            if key_brand in self._modes_by_brand.index:
                mode_value = self._modes_by_brand.loc[key_brand, col]
                if pd.notna(mode_value):
                    return mode_value
            
            # If all fail, return original value
            return row[col]

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
                 usd_to_target_rate: float = 1200.0,
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
        self.column_order = column_order
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


class Normalizer:
    """
    Normalize specified columns using StandardScaler.
    Handles the case where some columns might not exist.
    """
    
    def __init__(self, 
                 target_columns: List[str] = None,
                 verbose: bool = True):
        """
        Args:
            target_columns: List of columns to normalize. If None, will use default set.
            verbose: Whether to print progress information
        """
        if target_columns is None:
            default_cols = ["Marca", "Modelo", "Versión", "Motor", "cv", "Kilómetros", "Año"]
            # Add desc_pca columns
            desc_pca_cols = [f"desc_svd_{i}" for i in range(1, 31)]
            target_columns = default_cols + desc_pca_cols
            
        self.target_columns = target_columns
        self.verbose = verbose
        
        # Will store fitted scalers for each column
        self._scalers: Dict[str, StandardScaler] = {}
        self._columns_to_normalize: List[str] = []
        
    def fit(self, df: pd.DataFrame):
        """Fit StandardScaler on each target column that exists in the dataframe."""
        self._scalers = {}
        self._columns_to_normalize = []
        
        for col in self.target_columns:
            if col in df.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    scaler = StandardScaler()
                    # Reshape for sklearn (needs 2D array)
                    scaler.fit(df[col].values.reshape(-1, 1))
                    self._scalers[col] = scaler
                    self._columns_to_normalize.append(col)
                elif self.verbose:
                    print(f"\t NORMALIZER: Column '{col}' is not numeric, skipping")
            elif self.verbose:
                print(f"\t NORMALIZER: Column '{col}' not found, skipping")
        
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the fitted columns using StandardScaler."""
        if not self._scalers:
            raise RuntimeError("Normalizer must be fitted before transform()")
        
        result = df.copy()
        
        for col in self._columns_to_normalize:
            if col in result.columns:
                scaler = self._scalers[col]
                # Transform and reshape back to 1D
                normalized_values = scaler.transform(result[col].values.reshape(-1, 1)).flatten()
                result[col] = normalized_values
        
        if self.verbose:
            print(f"\t NORMALIZER: Normalized {len(self._columns_to_normalize)} columns")
        
        return result

