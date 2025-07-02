import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Hiperparámetros del modelo
params = {
    "subsample": 1.0,
    "reg_lambda": 1,
    "reg_alpha": 0.01,
    "n_estimators": 300,
    "max_depth": 6,
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
    if len(sys.argv) != 3:
        print("Uso: python entrenar_xgboost.py archivo_train.csv archivo_val.csv")
        sys.exit(1)

    archivo_train = sys.argv[1]
    archivo_val = sys.argv[2]

    df_train = pd.read_csv(archivo_train)
    df_val = pd.read_csv(archivo_val)

    for df, nombre in [(df_train, 'train'), (df_val, 'validation')]:
        if "Precio" not in df.columns:
            print(f"Error: el archivo de {nombre} debe contener una columna 'Precio' como variable objetivo.")
            sys.exit(1)

    y_train = df_train["Precio"]
    X_train = df_train.drop(columns=["Precio"])
    y_val = df_val["Precio"]
    X_val = df_val.drop(columns=["Precio"])

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

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

    print(f"Train -> R²: {r2_train:.4f}, RMSE: {rmse_train:.2f}")
    print(f"Val   -> R²: {r2_val:.4f}, RMSE: {rmse_val:.2f}")

if __name__ == "__main__":
    main()
