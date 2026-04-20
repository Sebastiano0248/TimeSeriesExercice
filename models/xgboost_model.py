"""
XGBoost with lag-based feature engineering.
Same feature set as random_forest.py — enables direct apples-to-apples comparison.
Uncertainty: fixed residual std from training set (homoscedastic approximation).

Dependencies: xgboost
  pip install xgboost
"""

import numpy as np
from xgboost import XGBRegressor

MODEL_NAME = "XGBoost"

N_LAGS       = 28
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
MAX_DEPTH    = 6


def _features(window: np.ndarray) -> np.ndarray:
    lags = window[-N_LAGS:]
    return np.concatenate([
        lags,
        [np.mean(lags[-7:]),  np.std(lags[-7:])],
        [np.mean(lags[-14:]), np.std(lags[-14:])],
        [np.mean(lags[-28:]), np.std(lags[-28:])],
    ])


class XGBModel:
    def __init__(self, model: XGBRegressor, last_values: np.ndarray, resid_std: float):
        self.model       = model
        self.last_values = last_values.copy()
        self.resid_std   = resid_std

    def predict(self, h: int) -> dict:
        window = list(self.last_values)
        preds  = []
        for _ in range(h):
            feat = _features(np.array(window)).reshape(1, -1)
            p    = self.model.predict(feat)[0]
            preds.append(p)
            window.append(p)

        mean = np.array(preds)
        std  = np.full(h, self.resid_std)
        return {
            "mean":     mean,
            "lower_80": mean - 1.28 * std,
            "upper_80": mean + 1.28 * std,
            "lower_95": mean - 1.96 * std,
            "upper_95": mean + 1.96 * std,
        }


def train(y_train, y_val=None) -> XGBModel:
    values = y_train.values.astype(np.float64)

    # Reservem el 10% final per estimar resid_std out-of-sample (evita overfitting en-sample)
    n_val   = max(N_LAGS + 1, int(len(values) * 0.10))
    tr      = values[:-n_val]
    heldout = values[-n_val:]

    X, Y = [], []
    for i in range(N_LAGS, len(tr)):
        X.append(_features(tr[:i]))
        Y.append(tr[i])
    X, Y = np.array(X), np.array(Y)

    X_v, Y_v = [], []
    all_vals  = np.concatenate([tr, heldout])
    for i in range(len(tr), len(all_vals)):
        X_v.append(_features(all_vals[:i]))
        Y_v.append(all_vals[i])
    X_v, Y_v = np.array(X_v), np.array(Y_v)

    eval_set = [(X_v, Y_v)]
    if y_val is not None:
        val_values = y_val.values.astype(np.float64)
        X_ext, Y_ext = [], []
        ext_all = np.concatenate([values, val_values])
        for i in range(len(values), len(ext_all)):
            X_ext.append(_features(ext_all[:i]))
            Y_ext.append(ext_all[i])
        eval_set = [(np.array(X_ext), np.array(Y_ext))]

    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
        verbosity=0,
    )
    model.fit(X, Y, eval_set=eval_set, verbose=False)

    resid_std = float(np.std(Y_v - model.predict(X_v)))
    return XGBModel(model, values[-N_LAGS:], resid_std)
