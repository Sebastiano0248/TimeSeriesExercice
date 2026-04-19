"""
GARCH(1,1) model on log returns with constant mean.
Mean forecast: Naïve (random walk in log prices → geometric random walk in levels).
Intervals: derived from GARCH h-step ahead conditional volatility.

This captures the key GARCH feature — time-varying volatility (volatility clustering)
— without requiring the mean forecast to be more than a random walk.

Dependencies: arch
  pip install arch
"""

import warnings
import numpy as np
from arch import arch_model

MODEL_NAME = "GARCH"


class GARCHModel:
    def __init__(self, fitted, last_price: float):
        self.fitted     = fitted
        self.last_price = last_price

    def predict(self, h: int) -> dict:
        fc = self.fitted.forecast(horizon=h, reindex=False)

        # Conditional volatility in % log-return units → convert to price units
        vol_pct   = np.sqrt(fc.variance.values[-1])          # shape (h,)
        std_price = self.last_price * (vol_pct / 100.0)

        # Naïve mean (random walk)
        mean = np.full(h, self.last_price)

        return {
            "mean":     mean,
            "lower_80": mean - 1.28 * std_price,
            "upper_80": mean + 1.28 * std_price,
            "lower_95": mean - 1.96 * std_price,
            "upper_95": mean + 1.96 * std_price,
        }


def train(y_train, y_val=None) -> GARCHModel:
    values      = y_train.values.astype(np.float64)
    log_returns = np.diff(np.log(values)) * 100.0  # % log returns, scale for numerics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch  = arch_model(log_returns, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
        fitted = garch.fit(disp="off", show_warning=False)

    return GARCHModel(fitted, float(values[-1]))
