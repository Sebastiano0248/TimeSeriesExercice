"""
Drift model: random walk + constant slope estimated from full training history.
slope = (y_last - y_first) / (n - 1)
Uncertainty grows as σ·√h, identical to Naïve.
"""

import numpy as np

MODEL_NAME = "Drift"


class DriftModel:
    def __init__(self, last_value: float, slope: float, return_std: float):
        self.last_value = last_value
        self.slope      = slope
        self.return_std = return_std

    def predict(self, h: int) -> dict:
        t     = np.arange(1, h + 1)
        mean  = self.last_value + self.slope * t
        std_h = self.return_std * np.sqrt(t)
        return {
            "mean":     mean,
            "lower_80": mean - 1.28 * std_h,
            "upper_80": mean + 1.28 * std_h,
            "lower_95": mean - 1.96 * std_h,
            "upper_95": mean + 1.96 * std_h,
        }


def train(y_train, y_val=None) -> DriftModel:
    n          = len(y_train)
    last       = float(y_train.iloc[-1])
    first      = float(y_train.iloc[0])
    slope      = (last - first) / (n - 1)
    return_std = float(y_train.diff().dropna().std())
    return DriftModel(last, slope, return_std)
