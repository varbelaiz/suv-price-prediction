#!/usr/bin/env python3
"""
Complete ML Pipeline for SUV Price Prediction

This script automates the entire machine learning workflow:
1. Data preprocessing (cleaning raw dataset)
2. Data transformation (feature engineering and encoding)
3. Model training and evaluation (XGBoost)

Usage: python pipe.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path so we can import modules
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "modelos"))

# Import the modules
from data_preprocessing import preprocess_data
from data_transformation import transform_datasets
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error


def train_xgboost_model(train_file: str, val_file: str, verbose: bool = True):
    """
    Train XGBoost model with the given train and validation files.
    
    Args:
        train_file: Path to training data CSV
        val_file: Path to validation data CSV
        verbose: Whether to print progress information
    """
    # XGBoost hyperparameters (from original script)
    params = {
        "subsample": 1.0,
        "reg_lambda": 1,
        "reg_alpha": 0.01,
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.01,
        "gamma": 0,
        "colsample_bytree": 0.6
    }
    
    if verbose:
        print("=" * 50)
        print("STARTING MODEL TRAINING")
        print("=" * 50)
    
    # Load data
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    
    if verbose:
        print(f"Loaded training data: {df_train.shape}")
        print(f"Loaded validation data: {df_val.shape}")
    
    # Check for target column
    for df, name in [(df_train, 'train'), (df_val, 'validation')]:
        if "Precio" not in df.columns:
            raise ValueError(f"Error: {name} file must contain 'Precio' column as target variable.")
    
    # Prepare features and targets
    y_train = df_train["Precio"]
    X_train = df_train.drop(columns=["Precio"])
    y_val = df_val["Precio"]
    X_val = df_val.drop(columns=["Precio"])
    
    # Verify all columns are numeric
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X_train.dtypes]) or \
       not np.all([np.issubdtype(dtype, np.number) for dtype in X_val.dtypes]):
        raise ValueError("Error: All feature columns must be numeric after transformation.")
    
    if verbose:
        print(f"Training features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
    
    # Train model
    if verbose:
        print("Training XGBoost model...")
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = Path("modelos/modelo_entrenado.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    if verbose:
        print(f"Model saved to: {model_path}")
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    
    if verbose:
        print("\nMODEL EVALUATION RESULTS:")
        print("-" * 30)
        print(f"Train -> R²: {r2_train:.4f}, RMSE: {rmse_train:,.2f}")
        print(f"Val   -> R²: {r2_val:.4f}, RMSE: {rmse_val:,.2f}")
        print("=" * 50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
    
    return {
        'model': model,
        'r2_train': r2_train,
        'r2_val': r2_val,
        'rmse_train': rmse_train,
        'rmse_val': rmse_val
    }


def run_full_pipeline(verbose: bool = True):
    """
    Run the complete ML pipeline from raw data to trained model.
    
    Args:
        verbose: Whether to print detailed progress information
    """
    if verbose:
        print("🚀 STARTING COMPLETE ML PIPELINE")
        print("=" * 60)
        print("Pipeline steps:")
        print("1. Data Preprocessing (cleaning)")
        print("2. Data Transformation (feature engineering)")
        print("3. Model Training & Evaluation")
        print("=" * 60)
    
    try:
        # Step 1: Data Preprocessing
        if verbose:
            print("\n📊 STEP 1: DATA PREPROCESSING")
        preprocess_data(verbose=verbose)
        
        # Step 2: Data Transformation
        if verbose:
            print("\n🔧 STEP 2: DATA TRANSFORMATION")
        transform_datasets(apply_to_test=True, val_size=0.2, verbose=verbose)
        
        # Step 3: Model Training
        if verbose:
            print("\n🤖 STEP 3: MODEL TRAINING")
        
        # File paths for training
        train_file = "data/train/transformed_train.csv"
        val_file = "data/train/transformed_val.csv"
        
        # Check if files exist
        if not Path(train_file).exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not Path(val_file).exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        # Train model
        results = train_xgboost_model(train_file, val_file, verbose=verbose)
        
        if verbose:
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("FINAL RESULTS:")
            print(f"  📈 Training R²: {results['r2_train']:.4f}")
            print(f"  📈 Validation R²: {results['r2_val']:.4f}")
            print(f"  📉 Training RMSE: {results['rmse_train']:,.2f}")
            print(f"  📉 Validation RMSE: {results['rmse_val']:,.2f}")
            print("\n📁 Generated files:")
            print("  - data/clean.csv (cleaned dataset)")
            print("  - data/train/cleaned_train.csv")
            print("  - data/test/cleaned_test.csv")
            print("  - data/train/transformed_train.csv")
            print("  - data/train/transformed_val.csv")
            print("  - data/test/transformed_test.csv")
            print("  - modelos/modelo_entrenado.pkl (trained model)")
            print("=" * 60)
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            print("=" * 60)
        raise


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete ML Pipeline for SUV Price Prediction")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Print detailed progress information")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Run in quiet mode (minimal output)")
    
    args = parser.parse_args()
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    try:
        results = run_full_pipeline(verbose=verbose)
        if not verbose:
            print(f"Pipeline completed. Validation R²: {results['r2_val']:.4f}")
        sys.exit(0)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
