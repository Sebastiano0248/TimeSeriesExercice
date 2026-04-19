"""
ETS / Holt's Exponential Smoothing with additive damped trend.
Mean forecast via statsmodels ExponentialSmoothing.
Intervals via bootstrap simulation of the fitted model.

Dependencies: statsmodels
"""

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MODEL_NAME = "ETS_Holt"

N_SIM = 500   # simulation repetitions for prediction intervals


class ETSModel:
    def __init__(self, fitted_model):
        self.fitted = fitted_model

    def predict(self, h: int) -> dict:
        mean = self.fitted.forecast(h).values

        # Bootstrap simulation for honest prediction intervals
        sim = self.fitted.simulate(
            nsimulations=h, repetitions=N_SIM, error="add"
        ).values  # shape (h, N_SIM)

        lower_80 = np.percentile(sim, 10, axis=1)
        upper_80 = np.percentile(sim, 90, axis=1)
        lower_95 = np.percentile(sim, 2.5, axis=1)
        upper_95 = np.percentile(sim, 97.5, axis=1)

        return {
            "mean":     mean,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
        }


def train(y_train, y_val=None) -> ETSModel:
    model  = ExponentialSmoothing(
        y_train,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    return ETSModel(fitted)
