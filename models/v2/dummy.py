"""
Dummy model: always predicts the training mean with fixed uncertainty.
Follows the standard model interface required by forecasting_pipeline.ipynb.

Interface contract
------------------
MODULE_NAME : str
    Human-readable model name used for result folders.

train(y_train, y_val=None) -> model object
    Fits the model. y_train and y_val are pd.Series indexed by dates.
    y_val may be used for internal hyperparameter tuning (optional).
    Returns an object with a .predict(h) method.

model.predict(h) -> dict
    Returns a dict with keys:
        "mean"      : np.ndarray of shape (h,)
        "lower_80"  : np.ndarray of shape (h,)
        "upper_80"  : np.ndarray of shape (h,)
        "lower_95"  : np.ndarray of shape (h,)
        "upper_95"  : np.ndarray of shape (h,)
"""

import numpy as np

MODEL_NAME = "DummyConstant"


class DummyModel:
    def __init__(self, constant: float, std: float):
        self.constant = constant
        self.std = std

    def predict(self, h: int) -> dict:
        mean = np.full(h, self.constant)
        return {
            "mean":     mean,
            "lower_80": mean - 1.28 * self.std,
            "upper_80": mean + 1.28 * self.std,
            "lower_95": mean - 1.96 * self.std,
            "upper_95": mean + 1.96 * self.std,
        }


def train(y_train, y_val=None) -> DummyModel:
    constant = float(y_train.mean())
    std = float(y_train.std())
    return DummyModel(constant, std)
