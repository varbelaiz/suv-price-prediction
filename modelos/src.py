from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error, median_absolute_error

def calculate_metrics(y, y_pred, y_val = None, y_pred_val = None, verbose=True):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    evs = explained_variance_score(y, y_pred)
    max_err = max_error(y, y_pred)
    medae = median_absolute_error(y, y_pred)
    metrics = {
        "mse": mse,
        "r2": r2,
        "mae": mae,
        "evs": evs,
        "max_err": max_err,
        "medae": medae
    }

    if y_val is not None and y_pred_val is not None:
        mse_val = mean_squared_error(y_val, y_pred_val)
        r2_val = r2_score(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        evs_val = explained_variance_score(y_val, y_pred_val)
        max_err_val = max_error(y_val, y_pred_val)
        medae_val = median_absolute_error(y_val, y_pred_val)
        metrics.update({
            "mse_val": mse_val,
            "r2_val": r2_val,
            "mae_val": mae_val,
            "evs_val": evs_val,
            "max_err_val": max_err_val,
            "medae_val": medae_val
        })

        if verbose:
            print("Metric       |    Train     |  Validation")
            print("-------------|--------------|---------------")
            print(f"MSE          |   {mse:.2e}   |   {mse_val:.2e}")
            print(f"R²           |   {r2:.4f}     |   {r2_val:.4f}")
            print(f"MAE          |   {mae:.2e}   |   {mae_val:.2e}")
            print(f"EVS          |   {evs:.4f}     |   {evs_val:.4f}")
            print(f"Max Error    |   {max_err:.2e}   |   {max_err_val:.2e}")
            print(f"Median AE    |   {medae:.2e}   |   {medae_val:.2e}")
            print()
            
    elif verbose:
        print(f"MSE          |   {mse:.2e}   ")
        print(f"R²           |   {r2:.4f}     ")
        print(f"MAE          |   {mae:.2e}   ")
        print(f"EVS          |   {evs:.4f}     ")
        print(f"Max Error    |   {max_err:.2e}   ")
        print(f"Median AE    |   {medae:.2e}   ")
        print()
    
    return metrics