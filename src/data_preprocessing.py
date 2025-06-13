import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path("data/raw.csv")
CLEAN_ALL = Path("data/clean.csv")
TRAIN_FILE = Path("data/train/cleaned_train.csv")
TEST_FILE = Path("data/test/cleaned_test.csv")

def is_turbo(row):
    motor = str(row.get('Motor', ''))
    version = str(row.get('Versión', ''))
    combined = f"{motor} {version}"
    if re.search(r'turbo', combined, re.IGNORECASE):
        return 1
    if re.search(r'(?:\d[\.,]?\d?\s*[tT]|TSI|TFSI|TURBO)', combined, re.IGNORECASE):
        return 1
    return 0

def extract_cv(text):
    if not isinstance(text, str):
        return np.nan
    match = re.search(r'(\d+)\s*[cC][vV]\b', text)
    if match:
        return int(match.group(1))
    return np.nan

def extract_traccion(text):
    if not isinstance(text, str):
        return np.nan
    if re.search(r'4[xX]4|4wd|awd', text, re.IGNORECASE):
        return '4x4'
    if re.search(r'4[xX]2|2wd|fwd', text, re.IGNORECASE):
        return '4x2'
    return np.nan

def extract_engine_size(version):
    if not isinstance(version, str):
        return None
    match = re.match(r'(\d\.\d)', version)
    if match:
        return match.group(1)
    return None

def fix_motor(row):
    motor = row['Motor']
    version = row['Versión']
    try:
        float(motor)
        is_number = True
    except (ValueError, TypeError):
        is_number = False
    if not is_number:
        engine_size = extract_engine_size(version)
        if engine_size:
            return engine_size
    return motor

def remove_cilinder_size(version):
    if not isinstance(version, str):
        return version
    pattern = r'(^|\s)(\d[\.,]\d)\s*[Ll]?(?=\s|$|\W)'
    version = re.sub(pattern, ' ', version)
    version = re.sub(r'\b\d[\.,]?\d?\s*[tT]\b', ' ', version)
    version = re.sub(r'\b(TSI|TFSI|TURBO)\b', ' ', version, flags=re.IGNORECASE)
    version = re.sub(' +', ' ', version).strip()
    return version

def remove_transmission(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\b\d*\s*(at|mt|dct|cvt)\d*\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(at|mt|dct|cvt|automática|manual)\b', '', text, flags=re.IGNORECASE)
    return re.sub(' +', ' ', text).strip()

def remove_horsepower(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'\b\d+\s*(cv|hp)\b', '', text, flags=re.IGNORECASE).replace('  ', ' ').strip()

def remove_traction(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\b4[xX][24]\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bawd\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b4wd\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b2wd\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfwd\b', '', text, flags=re.IGNORECASE)
    return re.sub(' +', ' ', text).strip()

def remove_other_data(row):
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

def clean_version_column(row):
    """
    Apply all version cleaning functions in sequence.
    This function combines all the individual cleaning steps into a single operation.
    """
    version = str(row.get('Versión', ''))
    
    # Apply all cleaning functions in sequence
    version = remove_cilinder_size(version)
    version = remove_transmission(version)
    version = remove_horsepower(version)
    version = remove_traction(version)
    
    # Create a temporary row with the cleaned version for remove_other_data
    temp_row = row.copy()
    temp_row['Versión'] = version
    version = remove_other_data(temp_row)
    
    return version

def main():
    """
    Main function to process the raw dataset and create cleaned train/test splits.
    """
    # Load the raw dataset
    dataset = pd.read_csv(RAW)
    
    # Extract features from the data
    dataset['Turbo'] = dataset.apply(is_turbo, axis=1)
    dataset['cv'] = dataset['Versión'].apply(extract_cv)
    dataset['Tracción'] = dataset['Versión'].apply(extract_traccion)
    dataset['Motor'] = dataset.apply(fix_motor, axis=1)
    
    # Clean the Versión column using the combined function
    dataset['Versión'] = dataset.apply(clean_version_column, axis=1)
    
    # Save the cleaned dataset
    dataset.to_csv(CLEAN_ALL, index=False)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    print(f"Data preprocessing completed successfully!")
    print(f"Cleaned dataset saved to: {CLEAN_ALL}")
    print(f"Training set saved to: {TRAIN_FILE}")
    print(f"Test set saved to: {TEST_FILE}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")


if __name__ == "__main__":
    main()
