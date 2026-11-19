"""Shared helpers: data loading, mileage fix, and the final blended model.

Kept in its own module so the notebook, train.py and predict.py all load the
same model class (needed for joblib to save/load it).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SEED = 42
CATEGORICAL = ["MakeName", "Country", "Model", "FuelType", "GearType", "Options", "RegionName"]
NUMERIC = ["Year", "Mileage", "EngineSize"]  # Age is dropped: it is always 2026 - Year
FEATURES = CATEGORICAL + NUMERIC


def load_data(path="cars_export.csv"):
    df = pd.read_csv(path)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    # Access exports prices as "$95000.00" -> plain number (values are SAR)
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True))
    return df


def fix_mileage(df):
    """~8% of cars have mileage under 1,000 (e.g. 230): clearly entered in thousands of km."""
    df = df.copy()
    small = df["Mileage"] < 1000
    df.loc[small, "Mileage"] = df.loc[small, "Mileage"] * 1000
    return df


def make_xgb(max_depth=6, n_estimators=500, learning_rate=0.05):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough")
    reg = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
    return Pipeline([("prep", pre), ("reg", reg)])


def make_mlp():
    """Small neural network: one hidden layer of 64 neurons, trained on log(price).
    Inputs are one-hot categories (rare ones grouped) plus standardised numbers."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3), CATEGORICAL),
         ("num", StandardScaler(), NUMERIC)])
    net = MLPRegressor(hidden_layer_sizes=(64,), alpha=0.01, max_iter=2000, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
    return TransformedTargetRegressor(Pipeline([("prep", pre), ("net", net)]),
                                      func=np.log1p, inverse_func=np.expm1)


class BlendRegressor(BaseEstimator, RegressorMixin):
    """Equal average of a tuned XGBoost, a CatBoost (log price) and a neural network (log price)."""

    def fit(self, X, y):
        X = X[FEATURES]
        self.xgb_ = make_xgb(max_depth=8, n_estimators=800, learning_rate=0.03).fit(X, y)
        self.cat_ = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=6, verbose=0,
                                      random_seed=SEED, cat_features=CATEGORICAL, thread_count=-1)
        self.cat_.fit(X, np.log1p(np.asarray(y)))
        self.nn_ = make_mlp().fit(X, y)
        return self

    def predict(self, X):
        X = X[FEATURES]
        return (self.xgb_.predict(X) + np.expm1(self.cat_.predict(X)) + self.nn_.predict(X)) / 3
