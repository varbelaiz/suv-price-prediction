import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Hiperparámetros del modelo
params = {
    "subsample": 1.0,
    "reg_lambda": 1,
    "reg_alpha": 0.01,
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.1,
    "gamma": 0,
    "colsample_bytree": 0.6
}

# --- Funciones auxiliares ---

def esta_normalizado(df, tolerancia=0.2):
    medias = df.mean()
    stds = df.std()
    return np.all(np.abs(medias) < tolerancia) and np.all(np.abs(stds - 1) < tolerancia)

# --- Código principal ---

def main():
    """
    if len(sys.argv) != 3:
        print("Uso: python entrenar_xgboost.py archivo_train.csv archivo_val.csv")
        sys.exit(1)

    archivo_train = sys.argv[1]
    archivo_val = sys.argv[2]
    """

    archivo_train = "data/train/transformed_train.csv"
    archivo_val = "data/train/transformed_val.csv"
    archivo_test = "data/test/transformed_test.csv"

    df_train = pd.read_csv(archivo_train)
    df_val = pd.read_csv(archivo_val)
    df_test = pd.read_csv(archivo_test)

    for df, nombre in [(df_train, 'train'), (df_val, 'validation')]:
        if "Precio" not in df.columns:
            print(f"Error: el archivo de {nombre} debe contener una columna 'Precio' como variable objetivo.")
            sys.exit(1)

    y_train = df_train["Precio"]
    X_train = df_train.drop(columns=["Precio"])
    y_val = df_val["Precio"]
    X_val = df_val.drop(columns=["Precio"])
    y_test = df_test["Precio"]
    X_test = df_test.drop(columns=["Precio"])

    # Verificación de tipos numéricos
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X_train.dtypes]) or \
       not np.all([np.issubdtype(dtype, np.number) for dtype in X_val.dtypes]):
        print("Error: Los datos deben estar totalmente numerizados.")
        sys.exit(1)

    # Entrenamiento del modelo
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, "modelo_entrenado.pkl")

    # Evaluación
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)

    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"Train -> R²: {r2_train:.4f}, RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}")
    print(f"Val   -> R²: {r2_val:.4f}, RMSE: {rmse_val:.2f}, MAE: {mae_val:.2f}")
    print(f"Test  -> R²: {r2_test:.4f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}")

if __name__ == "__main__":
    main()
