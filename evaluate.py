"""Metrics and summaries."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {"MAE_eV": float(mean_absolute_error(y_true, y_pred)),
            "RMSE_eV": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def summarize(runs, by=("representation", "split_type")):
    """Mean and standard deviation over seeds for each representation x split type."""
    g = runs.groupby(list(by))
    out = g[["MAE_eV", "RMSE_eV", "R2"]].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if np.isfinite(std) else f"{mean:.{digits}f}"
