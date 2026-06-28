"""
AQRTINet — Standalone Model

This is the standalone version with no AQRTI project dependencies.
It can be used with any DataFrame of features and binary labels.

For AQRTI integration, use backend/ml/models/aqrtinet_model.py which
wraps this with the BaseModel adapter and DB regime routing.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from aqrtinet.percentile import PercentileRanker
from aqrtinet.loss import CLASS_WEIGHT

REGIMES = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
FALLBACK_REGIME = "BULL"
MIN_REGIME_ROWS = 80


def _build_expert(hyperparams: dict) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter            = hyperparams.get("max_iter", 300),
        learning_rate       = hyperparams.get("learning_rate", 0.05),
        max_depth           = hyperparams.get("max_depth", 6),
        min_samples_leaf    = hyperparams.get("min_samples_leaf", 20),
        l2_regularization   = hyperparams.get("l2_regularization", 0.1),
        max_features        = hyperparams.get("max_features", 0.8),
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = 30,
        class_weight        = CLASS_WEIGHT,
        random_state        = 42,
        verbose             = 0,
    )


class AQRTINet:
    """
    Custom gradient-boosted tree ensemble for stock direction prediction.

    Three core innovations:
      1. Asymmetric trading loss (2× penalty for false positives)
      2. Regime-aware mixture of experts (BULL/BEAR/SIDEWAYS/VOLATILE)
      3. Cross-sectional percentile ranking (features as distribution ranks)

    Args:
        hyperparams: Optional dict to override defaults
        regime_col:  Column name in X containing the market regime label.
                     If None, all rows use the BULL expert (no regime routing).

    Example:
        import pandas as pd
        from aqrtinet import AQRTINet

        model = AQRTINet()
        model.fit(X_train, y_train, regime_col="market_regime")
        probas = model.predict_proba(X_test, regime="BULL")
    """

    DEFAULT_HYPERPARAMS = {
        "max_iter":          300,
        "learning_rate":     0.05,
        "max_depth":         6,
        "min_samples_leaf":  20,
        "l2_regularization": 0.1,
        "max_features":      0.8,
    }

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._experts: dict[str, Any] = {}
        self._ranker: Optional[PercentileRanker] = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_col: Optional[str] = None,
    ) -> "AQRTINet":
        """
        Train AQRTINet.

        Args:
            X:          Feature DataFrame (rows = samples)
            y:          Binary labels (1 = UP, 0 = DOWN)
            regime_col: Column name in X with market regime
                        (BULL/BEAR/SIDEWAYS/VOLATILE). If None, single expert.
        """
        # Extract regime series before ranking (it's categorical, not a feature)
        if regime_col and regime_col in X.columns:
            regimes = X[regime_col].fillna(FALLBACK_REGIME).str.upper()
            X_feat = X.drop(columns=[regime_col])
        else:
            regimes = pd.Series([FALLBACK_REGIME] * len(X), index=X.index)
            X_feat = X

        self._feature_cols = list(X_feat.columns)

        # Cross-sectional percentile ranking
        self._ranker = PercentileRanker(n_quantiles=100)
        X_ranked = self._ranker.fit_transform(X_feat)

        # Train one expert per regime
        self._experts = {}
        for regime in REGIMES:
            mask = regimes == regime
            n = mask.sum()

            if n < MIN_REGIME_ROWS:
                continue

            expert = _build_expert(self.hyperparams)
            expert.fit(X_ranked[mask], y[mask])
            self._experts[regime] = expert

        # Fallback expert on all data
        if FALLBACK_REGIME not in self._experts:
            expert = _build_expert(self.hyperparams)
            expert.fit(X_ranked, y)
            self._experts[FALLBACK_REGIME] = expert

        return self

    def predict_proba(self, X: pd.DataFrame, regime: str = FALLBACK_REGIME) -> np.ndarray:
        """
        Returns P(UP) for each row. Values in [0, 1].

        Args:
            X:      Feature DataFrame (same columns as fit)
            regime: Market regime for expert routing (BULL/BEAR/SIDEWAYS/VOLATILE)
        """
        X_feat = X[self._feature_cols] if set(self._feature_cols).issubset(X.columns) else X
        X_ranked = self._ranker.transform(X_feat)
        expert = self._experts.get(regime.upper()) or self._experts[FALLBACK_REGIME]
        proba = expert.predict_proba(X_ranked)
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, X: pd.DataFrame, regime: str = FALLBACK_REGIME, threshold: float = 0.5) -> np.ndarray:
        """Returns class labels (1=UP, 0=DOWN)."""
        return (self.predict_proba(X, regime) >= threshold).astype(int)

    def feature_importance(self) -> dict[str, float]:
        """Average feature importances across all regime experts."""
        all_imp = [
            e.feature_importances_
            for e in self._experts.values()
            if hasattr(e, "feature_importances_")
        ]
        if not all_imp:
            return {col: 1.0 / len(self._feature_cols) for col in self._feature_cols}
        avg = np.mean(all_imp, axis=0)
        total = avg.sum()
        normalized = avg / total if total > 0 else avg
        return dict(zip(self._feature_cols, normalized.tolist()))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "hyperparams":   self.hyperparams,
            "experts":       self._experts,
            "ranker":        self._ranker,
            "feature_cols":  self._feature_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AQRTINet":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(hyperparams=payload["hyperparams"])
        instance._experts      = payload["experts"]
        instance._ranker       = payload["ranker"]
        instance._feature_cols = payload["feature_cols"]
        return instance

    def is_trained(self) -> bool:
        return bool(self._experts)

    @property
    def experts_trained(self) -> list[str]:
        return list(self._experts.keys())
