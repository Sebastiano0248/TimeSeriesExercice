"""
ARIMA model with automatic order selection via pmdarima (auto_arima).
Falls back to ARIMA(2,1,2) if pmdarima is not installed.
Prediction intervals come directly from the statsmodels ARIMA analytical formula.

Dependencies: statsmodels, pmdarima (optional but recommended)
  pip install pmdarima
"""

import warnings
import numpy as np

MODEL_NAME = "ARIMA"

try:
    from pmdarima import auto_arima as _auto_arima
    _HAS_PMDARIMA = True
except ImportError:
    _HAS_PMDARIMA = False

from statsmodels.tsa.arima.model import ARIMA as _ARIMA


class ARIMAModel:
    def __init__(self, fitted):
        self.fitted = fitted

    def predict(self, h: int) -> dict:
        fc     = self.fitted.get_forecast(h)
        mean   = fc.predicted_mean.values
        ci_95  = fc.conf_int(alpha=0.05)
        ci_80  = fc.conf_int(alpha=0.20)
        return {
            "mean":     mean,
            "lower_80": ci_80.iloc[:, 0].values,
            "upper_80": ci_80.iloc[:, 1].values,
            "lower_95": ci_95.iloc[:, 0].values,
            "upper_95": ci_95.iloc[:, 1].values,
        }


def train(y_train, y_val=None) -> ARIMAModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if _HAS_PMDARIMA:
            auto   = _auto_arima(
                y_train, seasonal=False, stepwise=True,
                error_action="ignore", suppress_warnings=True,
                information_criterion="aic",
            )
            order  = auto.order
            print(f"  ARIMA order selected: {order}")
        else:
            order  = (2, 1, 2)
            print(f"  pmdarima not found — using default ARIMA{order}")

        fitted = _ARIMA(y_train, order=order).fit()

    return ARIMAModel(fitted)
