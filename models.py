"""Candidate models, their search spaces, and one way to build and fit any of them.

The seven candidates cover the main families used for molecular property prediction:
    ridge         linear model                       (is the relation simply linear?)
    knn           k-nearest neighbours               (classic similarity-based QSAR)
    random_forest bagged decision trees
    extra_trees   bagged, more randomized trees
    lightgbm      gradient-boosted trees (leaf-wise)
    xgboost       gradient-boosted trees (depth-wise)
    mlp           feed-forward neural network
Not included: SVR / Gaussian processes, whose training cost grows with the square (or cube) of the
number of molecules and is impractical for ~100k molecules on Colab.

Every candidate gets the same tuning budget (config.SEARCH_N_ITER random configurations).
Ridge, kNN and MLP are distance/gradient based, so their inputs are imputed and mapped to a normal
distribution with a quantile transform (robust to the very heavy tails of some RDKit descriptors).
Trees need no scaling.
"""
import importlib.util

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from . import config

HAS = {"lightgbm": importlib.util.find_spec("lightgbm") is not None,
       "xgboost": importlib.util.find_spec("xgboost") is not None}

BOOSTERS = {"lightgbm", "xgboost"}   # use early stopping on a held-out part of the training data

SEARCH_SPACES = {
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "knn": {"n_neighbors": [3, 5, 10, 20], "weights": ["uniform", "distance"]},
    "random_forest": {"max_features": [0.2, 0.33, 0.5, 1.0], "min_samples_leaf": [1, 2, 5]},
    "extra_trees": {"max_features": [0.2, 0.33, 0.5, 1.0], "min_samples_leaf": [1, 2, 5]},
    "lightgbm": {"learning_rate": [0.03, 0.05, 0.1], "num_leaves": [31, 63, 127, 255],
                 "min_child_samples": [10, 20, 50], "colsample_bytree": [0.3, 0.5, 0.8],
                 "reg_lambda": [0.0, 1.0, 5.0]},
    "xgboost": {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [6, 8, 10],
                "min_child_weight": [1, 5, 10], "colsample_bytree": [0.3, 0.5, 0.8],
                "reg_lambda": [0.0, 1.0, 5.0]},
    "mlp": {"hidden_layer_sizes": [(256,), (256, 128), (512, 256), (512, 256, 128)],
            "alpha": [1e-5, 1e-4, 1e-3], "learning_rate_init": [1e-3, 3e-4]},
}
CANDIDATES = tuple(SEARCH_SPACES)


def available(name):
    return HAS.get(name, True)


def _scaled(estimator, seed):
    return make_pipeline(SimpleImputer(strategy="median"),
                         QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                             subsample=200_000, random_state=seed),
                         estimator)


def build(name, params, seed):
    """Return an unfitted model for candidate `name` with hyperparameters `params`."""
    p = dict(params)
    if name == "ridge":
        return _scaled(Ridge(**p), seed)
    if name == "knn":
        return _scaled(KNeighborsRegressor(n_jobs=-1, **p), seed)
    if name == "mlp":
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        net = MLPRegressor(batch_size=256, max_iter=200, early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=10, random_state=seed, **p)
        return TransformedTargetRegressor(regressor=_scaled(net, seed), transformer=StandardScaler())
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=config.FOREST_TREES, n_jobs=-1, random_state=seed, **p))
    if name == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=config.FOREST_TREES, n_jobs=-1, random_state=seed, **p))
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=config.MAX_BOOSTING_ROUNDS, subsample=0.8,
                                 subsample_freq=1, metric="l1", n_jobs=-1, verbosity=-1,
                                 random_state=seed, **p)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=config.MAX_BOOSTING_ROUNDS, subsample=0.8,
                                tree_method="hist", eval_metric="mae",
                                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                                n_jobs=-1, random_state=seed, **p)
    raise ValueError(f"Unknown model '{name}'")


def fit(name, model, X, y, X_stop=None, y_stop=None):
    """Fit a model. Boosters stop adding trees when the MAE on (X_stop, y_stop) stops improving.
    Returns the number of boosting rounds used (0 for other models)."""
    if name == "lightgbm":
        import lightgbm as lgb
        model.fit(X, y, eval_set=[(X_stop, y_stop)],
                  callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS,
                                                first_metric_only=True, verbose=False)])
        return int(model.best_iteration_)
    if name == "xgboost":
        model.fit(X, y, eval_set=[(X_stop, y_stop)], verbose=False)
        return int(model.best_iteration) + 1
    model.fit(X, y)
    return 0
