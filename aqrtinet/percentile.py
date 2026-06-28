"""
AQRTINet — Cross-Sectional PercentileRanker

Converts raw feature values to within-date percentile ranks (0.0–1.0).

Instead of "RELIANCE RSI = 65" the model sees "RELIANCE RSI is at the
78th percentile of all stocks in the training distribution."

This makes features:
  - Scale-invariant across time (market conditions change, ranks don't)
  - Comparable across stocks with different price levels
  - Capturing cross-sectional alpha that single-stock models miss
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PercentileRanker:
    """
    Fits per-feature empirical CDFs from training data, then maps values to
    percentile ranks at inference time.

    No data leakage: rank boundaries are fitted only on training data.
    At inference: values are ranked against the training distribution,
    not against the current day's universe (which is unavailable at prediction time).

    Usage:
        ranker = PercentileRanker()
        ranker.fit(X_train)
        X_ranked = ranker.transform(X_test)   # values in [0, 1]
    """

    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self._quantiles: dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "PercentileRanker":
        """Learn empirical distribution of each feature from training data."""
        self._quantiles = {}
        q_points = np.linspace(0, 100, self.n_quantiles + 1)

        for col in X.columns:
            vals = X[col].dropna().values
            if len(vals) == 0:
                continue
            self._quantiles[col] = np.percentile(vals, q_points)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map each feature value to its percentile rank in the training distribution.

        Returns DataFrame with same shape and columns as X, values in [0, 1].
        NaN values remain NaN. Unknown columns pass through unchanged.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        result = X.copy().astype(float)

        for col in X.columns:
            if col not in self._quantiles:
                continue

            breaks = self._quantiles[col]
            vals = X[col].values.astype(float)
            ranked = np.where(
                np.isnan(vals),
                np.nan,
                np.searchsorted(breaks, vals, side="right") / len(breaks),
            )
            result[col] = ranked

        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def is_fitted(self) -> bool:
        return self._fitted

    def feature_count(self) -> int:
        return len(self._quantiles)
