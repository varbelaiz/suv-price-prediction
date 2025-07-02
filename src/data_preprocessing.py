import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
RAW = Path("data/raw.csv")
CLEAN_ALL = Path("data/clean.csv")
TRAIN_FILE = Path("data/train/cleaned_train.csv")
TEST_FILE = Path("data/test/cleaned_test.csv")


class TurboExtractor:
    """Extract turbo information and create Turbo column."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        result['Turbo'] = result.apply(self._is_turbo, axis=1)
        if verbose:
            turbo_count = result['Turbo'].sum()
            print(f"\t TURBO EXTRACTOR: Found {turbo_count} turbo vehicles")
        return result
    
    def _is_turbo(self, row):
        motor = str(row.get('Motor', ''))
        version = str(row.get('Versión', ''))
        combined = f"{motor} {version}"
        if re.search(r'turbo', combined, re.IGNORECASE):
            return 1
        if re.search(r'(?:\d[\.,]?\d?\s*[tT]|TSI|TFSI|TURBO)', combined, re.IGNORECASE):
            return 1
        return 0


class CVExtractor:
    """Extract CV (horsepower) information from Versión column."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        result['cv'] = result['Versión'].apply(self._extract_cv)
        if verbose:
            cv_found = result['cv'].notna().sum()
            print(f"\t CV EXTRACTOR: Extracted CV values for {cv_found} vehicles")
        return result
    
    def _extract_cv(self, text):
        if not isinstance(text, str):
            return np.nan
        match = re.search(r'(\d+)\s*[cC][vV]\b', text)
        if match:
            return int(match.group(1))
        return np.nan


class TraccionExtractor:
    """Extract traction information from Versión column."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        result['Tracción'] = result['Versión'].apply(self._extract_traccion)
        if verbose:
            traccion_found = result['Tracción'].notna().sum()
            print(f"\t TRACCION EXTRACTOR: Extracted traction info for {traccion_found} vehicles")
        return result
    
    def _extract_traccion(self, text):
        if not isinstance(text, str):
            return np.nan
        if re.search(r'4[xX]4|4wd|awd', text, re.IGNORECASE):
            return '4x4'
        if re.search(r'4[xX]2|2wd|fwd', text, re.IGNORECASE):
            return '4x2'
        return np.nan


class MotorFixer:
    """Fix Motor column by extracting engine size when Motor is not numeric."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        result['Motor'] = result.apply(self._fix_motor, axis=1)
        if verbose:
            numeric_motors = pd.to_numeric(result['Motor'], errors='coerce').notna().sum()
            print(f"\t MOTOR FIXER: {numeric_motors} vehicles have numeric motor values")
        return result
    
    def _extract_engine_size(self, text):
        """Extract engine size (like 1.2, 2.0, 3.6) from any text."""
        if not isinstance(text, str):
            return None
        
        # Look for patterns like 1.2, 2.0, 3.6, etc. anywhere in the text
        patterns = [
            r'\b(\d\.\d)\b',           # 1.2, 2.0, etc.
            r'\b(\d\,\d)\b',           # 1,2, 2,0, etc. (with comma)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                engine_size = match.group(1).replace(',', '.')  # Convert comma to dot
                try:
                    return float(engine_size)
                except ValueError:
                    continue
        
        return None

    def _fix_motor(self, row):
        motor = row['Motor']
        version = row['Versión']
        
        # First, try to convert motor directly to float
        try:
            return float(motor)
        except (ValueError, TypeError):
            pass
        
        # If motor is not numeric, try to extract engine size from motor text
        engine_size = self._extract_engine_size(motor)
        if engine_size is not None:
            return engine_size
        
        # If no engine size found in motor, try version as fallback
        engine_size = self._extract_engine_size(version)
        if engine_size is not None:
            return engine_size
        
        # If nothing found, return NaN
        return np.nan


class VersionCleaner:
    """Clean the Versión column by removing various components."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        result['Versión'] = result.apply(self._clean_version_column, axis=1)
        if verbose:
            cleaned_versions = result['Versión'].notna().sum()
            print(f"\t VERSION CLEANER: Cleaned {cleaned_versions} version strings")
        return result
    
    def _remove_cilinder_size(self, version):
        if not isinstance(version, str):
            return version
        pattern = r'(^|\s)(\d[\.,]\d)\s*[Ll]?(?=\s|$|\W)'
        version = re.sub(pattern, ' ', version)
        version = re.sub(r'\b\d[\.,]?\d?\s*[tT]\b', ' ', version)
        version = re.sub(r'\b(TSI|TFSI|TURBO)\b', ' ', version, flags=re.IGNORECASE)
        version = re.sub(' +', ' ', version).strip()
        return version

    def _remove_transmission(self, text):
        if not isinstance(text, str):
            return text
        text = re.sub(r'\b\d*\s*(at|mt|dct|cvt)\d*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(at|mt|dct|cvt|automática|manual)\b', '', text, flags=re.IGNORECASE)
        return re.sub(' +', ' ', text).strip()

    def _remove_horsepower(self, text):
        if not isinstance(text, str):
            return text
        return re.sub(r'\b\d+\s*(cv|hp)\b', '', text, flags=re.IGNORECASE).replace('  ', ' ').strip()

    def _remove_traction(self, text):
        if not isinstance(text, str):
            return text
        text = re.sub(r'\b4[xX][24]\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bawd\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b4wd\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b2wd\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfwd\b', '', text, flags=re.IGNORECASE)
        return re.sub(' +', ' ', text).strip()

    def _remove_other_data(self, row):
        version = str(row.get('Versión', ''))
        marca = str(row.get('Marca', '')).lower()
        modelo = str(row.get('Modelo', '')).lower()
        combustible = str(row.get('Tipo de combustible', '')).lower()
        carroceria = str(row.get('Tipo de carrocería', '')).lower()
        version_lower = version.lower()
        if combustible and combustible != 'nan':
            version_lower = re.sub(rf'\b{re.escape(combustible)}\b', '', version_lower)
        if marca and marca != 'nan':
            version_lower = re.sub(rf'\b{re.escape(marca)}\b', '', version_lower)
        if modelo and modelo != 'nan':
            version_lower = re.sub(rf'\b{re.escape(modelo)}\b', '', version_lower)
        if carroceria and carroceria != 'nan':
            version_lower = re.sub(rf'\b{re.escape(carroceria)}\b', '', version_lower)
        version_lower = re.sub(r'\b(nafta|gasolina|diesel|gnc|gas|híbrido|eléctrico)\b', '', version_lower)
        version_lower = re.sub(r'\b(suv|sedan|hatchback|pickup|coupe|convertible)\b', '', version_lower)
        version_cleaned = re.sub(r'\s+', ' ', version_lower).strip()
        return version_cleaned

    def _clean_version_column(self, row):
        """Apply all version cleaning functions in sequence."""
        version = str(row.get('Versión', ''))
        
        # Apply all cleaning functions in sequence
        version = self._remove_cilinder_size(version)
        version = self._remove_transmission(version)
        version = self._remove_horsepower(version)
        version = self._remove_traction(version)
        
        # Create a temporary row with the cleaned version for remove_other_data
        temp_row = row.copy()
        temp_row['Versión'] = version
        version = self._remove_other_data(temp_row)
        
        return version


class VersionFromTitleExtractor:
    """Extract version from title when Versión column is empty."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        
        # Check for empty strings after cleaning
        empty_mask = (result['Versión'] == '') | (result['Versión'].isna())
        empty_count = empty_mask.sum()
        
        if verbose:
            print(f"\t VERSION FROM TITLE: Found {empty_count} empty versions")
        
        if empty_count > 0:
            if verbose:
                print("\t VERSION FROM TITLE: Extracting versions from titles...")
            # Apply title extraction only to rows with empty versions
            result.loc[empty_mask, 'Versión'] = result[empty_mask].apply(self._extract_version_from_title, axis=1)
            
            # Check how many are still empty/unknown after title extraction
            final_unknown = (result['Versión'] == 'Unknown').sum()
            if verbose:
                print(f"\t VERSION FROM TITLE: Final 'Unknown' versions: {final_unknown}")
        
        return result
    
    def _extract_version_from_title(self, row):
        """Extract version information from the title by removing brand, model and applying cleaning."""
        title = str(row.get('Título', ''))
        marca = str(row.get('Marca', ''))
        modelo = str(row.get('Modelo', ''))
        
        if not title or title.lower() in ['nan', 'none', '']:
            return 'Unknown'
        
        # Convert to lowercase for processing
        title_lower = title.lower()
        marca_lower = marca.lower() if marca and marca != 'nan' else ''
        modelo_lower = modelo.lower() if modelo and modelo != 'nan' else ''
        
        # Remove brand and model from title
        if marca_lower and marca_lower != 'nan':
            title_lower = re.sub(rf'\b{re.escape(marca_lower)}\b', '', title_lower)
        
        if modelo_lower and modelo_lower != 'nan':
            title_lower = re.sub(rf'\b{re.escape(modelo_lower)}\b', '', title_lower)
        
        # Clean up extra spaces
        title_lower = re.sub(r'\s+', ' ', title_lower).strip()
        
        # Apply the same cleaning functions as version column
        version_cleaner = VersionCleaner()
        temp_row = row.copy()
        temp_row['Versión'] = title_lower
        version = version_cleaner._clean_version_column(temp_row)
        
        # If still empty after cleaning, return 'Unknown'
        if not version or version.strip() == '':
            return 'Unknown'
        
        return version.strip()


class KilometersCleaner:
    """Clean Kilómetros column by converting values like '58.000 km' to numeric."""
    
    def process(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        result = df.copy()
        
        # Count non-numeric values before cleaning
        non_numeric_before = 0
        if 'Kilómetros' in result.columns:
            non_numeric_before = pd.to_numeric(result['Kilómetros'], errors='coerce').isna().sum()
        
        result['Kilómetros'] = result['Kilómetros'].apply(self._clean_kilometers)
        
        if verbose:
            non_numeric_after = pd.to_numeric(result['Kilómetros'], errors='coerce').isna().sum()
            cleaned_count = non_numeric_before - non_numeric_after
            print(f"\t KILOMETERS CLEANER: Cleaned {cleaned_count} non-numeric kilometer values")
        
        return result
    
    def _clean_kilometers(self, value):
        """Convert kilometer values like '58.000 km' to numeric."""
        if pd.isna(value):
            return np.nan
        
        # Convert to string for processing
        value_str = str(value)
        
        # Check if it's already numeric
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Handle values with ' km' suffix and dots as thousands separators
        if isinstance(value_str, str) and value_str.endswith(' km'):
            # Remove ' km' suffix
            number_part = value_str.replace(' km', '')
            
            # Remove dots used as thousands separators
            number_part = number_part.replace('.', '')
            
            # Convert to numeric
            try:
                return float(number_part)
            except ValueError:
                return np.nan
        
        # If not in expected format, try to extract numeric part
        # This handles any other edge cases
        try:
            # Extract any numeric part from the string
            numeric_match = re.search(r'(\d+(?:\.\d+)?)', value_str.replace('.', ''))
            if numeric_match:
                return float(numeric_match.group(1))
        except (ValueError, AttributeError):
            pass
        
        return np.nan


class Pipeline:
    """Pipeline for chaining preprocessors."""
    
    def __init__(self, steps: List):
        self.steps = steps

    def process(self, df: pd.DataFrame, verbose: bool = True):
        tmp = df
        for step in self.steps:
            tmp = step.process(tmp, verbose=verbose)
        return tmp


def build_preprocessing_pipeline() -> Pipeline:
    """Build the preprocessing pipeline with all processors."""
    return Pipeline([
        TurboExtractor(),
        CVExtractor(),
        TraccionExtractor(),
        MotorFixer(),
        VersionCleaner(),
        VersionFromTitleExtractor(),
        KilometersCleaner(),
    ])


def preprocess_data(verbose: bool = True):
    """Main function to process the raw dataset and create cleaned train/test splits."""
    if verbose:
        print("=" * 50)
        print("STARTING DATA PREPROCESSING")
        print("=" * 50)
    
    # Load raw data
    dataset = pd.read_csv(RAW)
    if verbose:
        print(f"Loaded raw dataset: {dataset.shape}")
    
    # Build and apply preprocessing pipeline
    pipe = build_preprocessing_pipeline()
    dataset = pipe.process(dataset, verbose=verbose)
    
    # Check final state
    if verbose:
        print(f"\nFinal NaNs in Versión: {dataset['Versión'].isna().sum()}")
    
    # Save cleaned data
    dataset.to_csv(CLEAN_ALL, index=False)
    if verbose:
        print(f"Cleaned dataset saved to: {CLEAN_ALL}")
    
    # Split into train and test sets
    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    if verbose:
        print(f"Training set saved to: {TRAIN_FILE}")
        print(f"Test set saved to: {TEST_FILE}")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        print("=" * 50)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 50)


if __name__ == "__main__":
    preprocess_data(verbose=True)
