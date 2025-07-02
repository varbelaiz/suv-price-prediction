import pandas as pd
import numpy as np
import re

from pathlib import Path
from typing import List, Protocol
from sklearn.model_selection import train_test_split
from transformation.pipes import *

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
TRAIN_INPUT_FILE = Path("data/train/cleaned_train.csv")
TRAIN_OUTPUT_FILE = Path("data/train/transformed_train.csv")

VAL_OUTPUT_FILE = Path("data/train/transformed_val.csv")

TEST_INPUT_FILE = Path("data/test/cleaned_test.csv")
TEST_OUTPUT_FILE = Path("data/test/transformed_test.csv")

VERSIONS_FILE = "data/mercadolibre_versions.csv"  # optional catalogue
SIMILARITY_THRESHOLD = 0.3

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
        DescriptionEmbeddings(verbose=verbose, n_components=50),
        FillVersionNaNs(verbose=verbose),
        FillNaNs(["cv", "Motor", "Tracción", "Turbo"]),
        TargetEncoder(["Marca", "Modelo", "Versión"], verbose=verbose),
        Normalizer(verbose=verbose),
        CurrencyConverter(verbose=verbose),
        OneHotEncoder(["Tracción", "Tipo de combustible", "Transmisión", "Con cámara de retroceso", "Moneda", "Tipo de vendedor"], verbose=verbose),
        DropColumns(["Tipo de carrocería", "Título", "Descripción", "Color", "idx"]),
        OrderColumns(column_order=COLUMNS_ORDER),        
    ])


def transform_datasets(apply_to_test: bool = False, val_size: float = 0.2, verbose: bool = True):
    pipe = build_pipeline(verbose)

    # Load the full training data
    full_train_df = pd.read_csv(TRAIN_INPUT_FILE)
    
    # Print number of rows with "Unknown" versions
    print(f"Nans in Versión: {full_train_df['Versión'].isna().sum()}")
    print(f"Porcentaje de Nans en Versión: {full_train_df['Versión'].isna().sum() / len(full_train_df)}")

    
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
