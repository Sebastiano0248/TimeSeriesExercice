"""
Random Forest with lag-based feature engineering.
Features per sample: last N_LAGS values + rolling mean/std over 7 and 14 days.
Prediction: recursive (autoregressive) — predict 1 step, feed back as input.
Uncertainty: fixed residual std from training set (homoscedastic approximation).

Dependencies: scikit-learn
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

MODEL_NAME = "RandomForest"

N_LAGS       = 28
N_ESTIMATORS = 300


def _features(window: np.ndarray) -> np.ndarray:
    lags = window[-N_LAGS:]
    return np.concatenate([
        lags,
        [np.mean(lags[-7:]),  np.std(lags[-7:])],
        [np.mean(lags[-14:]), np.std(lags[-14:])],
        [np.mean(lags[-28:]), np.std(lags[-28:])],
    ])


class RFModel:
    def __init__(self, model: RandomForestRegressor, last_values: np.ndarray, resid_std: float):
        self.model       = model
        self.last_values = last_values.copy()
        self.resid_std   = resid_std

    def predict(self, h: int) -> dict:
        window = list(self.last_values)
        preds  = []
        for _ in range(h):
            feat  = _features(np.array(window)).reshape(1, -1)
            p     = self.model.predict(feat)[0]
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


def train(y_train, y_val=None) -> RFModel:
    values = y_train.values.astype(np.float64)

    # Reservem el 10% final per estimar resid_std out-of-sample (evita que RF memoritza)
    n_val   = max(N_LAGS + 1, int(len(values) * 0.10))
    tr      = values[:-n_val]
    heldout = values[-n_val:]

    X, Y = [], []
    for i in range(N_LAGS, len(tr)):
        X.append(_features(tr[:i]))
        Y.append(tr[i])
    X, Y = np.array(X), np.array(Y)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1
    )
    model.fit(X, Y)

    X_v, Y_v = [], []
    all_vals  = np.concatenate([tr, heldout])
    for i in range(len(tr), len(all_vals)):
        X_v.append(_features(all_vals[:i]))
        Y_v.append(all_vals[i])
    resid_std = float(np.std(np.array(Y_v) - model.predict(np.array(X_v))))

    return RFModel(model, values[-N_LAGS:], resid_std)
