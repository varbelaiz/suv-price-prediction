# SUV price prediction

Predicts the price of SUVs listed on MercadoLibre Argentina, trained on about 18,000 listings published in May 2025. Final project for the Machine Learning course at Universidad de San Andrés, by Eliana Ostrovsky and Valentino Arbelaiz. The full write-up, in Spanish, is [Informe.pdf](Informe.pdf).

## Results

XGBoost was the best model after hyperparameter search, with an R² of 0.944 on the held-out test set.

| Model             | R² before tuning | R² after tuning |
| ----------------- | ---------------- | --------------- |
| Linear regression | 0.799            | n/a             |
| Random Forest     | 0.876            | 0.929           |
| XGBoost           | 0.887            | 0.945           |
| MLP               | 0.803            | 0.318           |

R² on the validation set. The data is split 64% train, 16% validation and 20% test.

## How it works

- **Cleaning.** Text fields are normalized, technical attributes (engine, drivetrain, gearbox, rear camera) are extracted from the version and the title, and missing values are filled from other listings of the same version. Mileage outliers above the 90th percentile are dropped.
- **Version names.** Sellers type the version freely, which left over 1,100 distinct values. Each one is matched against a catalogue of official versions scraped from MercadoLibre by cosine similarity of character n-grams, and the ones without a match are grouped with DBSCAN, leaving about 200.
- **Encoding.** Brand, model and version are target encoded, low-cardinality fields are one-hot encoded, and numeric fields are standardized.
- **Models.** Linear regression as the baseline, then Random Forest, XGBoost and an MLP, tuned with random search, Bayesian optimization and Optuna.
- **Tried and dropped.** VAE embeddings of the listing descriptions, damage detection from the description with an LLM, and a flag for prices that are only a down payment. None of them improved the model.

## Running it

```bash
pip install -r requirements.txt
python pipe.py
```

`pipe.py` cleans the raw data, builds the features and trains the final XGBoost model, saving it to `modelos/modelo_entrenado.pkl`.
