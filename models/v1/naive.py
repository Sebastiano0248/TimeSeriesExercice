"""
Naïve model: predicts ŷ_{t+h} = y_t (random walk).
Uncertainty grows as σ·√h (standard random walk property).
Mandatory benchmark — any useful model must beat this.
"""

import numpy as np

MODEL_NAME = "Naive"


class NaiveModel:
    def __init__(self, last_value: float, return_std: float):
        self.last_value = last_value
        self.return_std = return_std

    def predict(self, h: int) -> dict:
        mean  = np.full(h, self.last_value)
        std_h = self.return_std * np.sqrt(np.arange(1, h + 1))
        return {
            "mean":     mean,
            "lower_80": mean - 1.28 * std_h,
            "upper_80": mean + 1.28 * std_h,
            "lower_95": mean - 1.96 * std_h,
            "upper_95": mean + 1.96 * std_h,
        }


def train(y_train, y_val=None) -> NaiveModel:
    last       = float(y_train.iloc[-1])
    return_std = float(y_train.diff().dropna().std())
    return NaiveModel(last, return_std)
