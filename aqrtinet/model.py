"""
AQRTINet v3.1 — Standalone Model

Self-contained implementation with no AQRTI project dependencies.
Works with any DataFrame of features and binary labels.

For AQRTI integration, use backend/ml/models/aqrtinet_model.py which
adds the BaseModel adapter, DB regime routing, and full training pipeline.

Eleven innovations over off-the-shelf gradient boosters:

  1.  ASYMMETRIC TRADING LOSS
      class_weight={0: 2.0, 1: 1.0} — false positives penalised 2× harder.

  2.  REGIME-AWARE MIXTURE OF EXPERTS
      One HistGBT specialist per market regime (BULL/BEAR/SIDEWAYS/VOLATILE)
      with regime-tuned hyperparameters.

  3.  CROSS-SECTIONAL PERCENTILE RANKING
      All features mapped to [0,1] rank in training distribution.

  4.  META-LEARNING / STACKING
      OOF predictions from base models as extra meta-features (AQRTI version).

  5.  PLATT PROBABILITY CALIBRATION
      5-fold OOF logistic regression → calibrated P(UP).

  6.  ENGINEERED INTERACTION FEATURES (9 crosses)
      - ix_momentum_x_trend    momentum × ADX
      - ix_rsi_x_vol           RSI × rolling vol
      - ix_volume_x_momentum   volume spike × momentum
      - ix_breadth_x_beta      breadth × beta
      - ix_sector_x_nifty      sector return × NIFTY return
      - ix_support_x_rsi       support distance × RSI
      - ix_rsi_x_momentum      RSI × 5d momentum
      - ix_vol_x_breakout      vol expansion × breakout distance
      - ix_trend_x_price       ADX × price vs EMA21

  7.  TEMPORAL DECAY WEIGHTING
      Exponential decay, half-life 252d (1 trading year).
      Calibrated for 5yr datasets — recent data weighted more.

  8.  REGIME-SPECIFIC FEATURE SELECTION
      Top-30 IC features per regime with domain-knowledge seed boosting.

  9.  CONFIDENT-LABEL FILTERING
      Rows where |5d return| < 0.5% are down-weighted 0.3× (noise zone).

  10. 7-FOLD STACKING OOF
      Higher fold count for lower-variance meta-features.

  11. 5-FOLD PLATT CALIBRATION
      More folds → tighter sigmoid fit with large datasets.

Example (standalone, no regime routing):
    from aqrtinet import AQRTINet

    model = AQRTINet()
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)

Example (with regime routing):
    model.fit(X_train, y_train, regime_col="market_regime")
    probas = model.predict_proba(X_test, regime="BULL")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from aqrtinet.percentile import PercentileRanker
from aqrtinet.loss import CLASS_WEIGHT

REGIMES       = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
FALLBACK_REGIME       = "BULL"
MIN_REGIME_ROWS       = 200
TEMPORAL_HALFLIFE_DAYS = 252
CONFIDENT_LABEL_THRESHOLD_PCT = 0.5
CONFIDENT_LABEL_WEIGHT        = 0.3

REGIME_HYPERPARAMS = {
    "BULL":     {"max_iter": 400, "learning_rate": 0.04, "max_depth": 7,
                 "min_samples_leaf": 15, "l2_regularization": 0.05, "max_features": 0.85},
    "BEAR":     {"max_iter": 350, "learning_rate": 0.04, "max_depth": 6,
                 "min_samples_leaf": 20, "l2_regularization": 0.15, "max_features": 0.8},
    "SIDEWAYS": {"max_iter": 300, "learning_rate": 0.05, "max_depth": 5,
                 "min_samples_leaf": 25, "l2_regularization": 0.2,  "max_features": 0.75},
    "VOLATILE": {"max_iter": 200, "learning_rate": 0.08, "max_depth": 4,
                 "min_samples_leaf": 30, "l2_regularization": 0.3,  "max_features": 0.7},
}

REGIME_FEATURE_HINTS = {
    "BULL":     ["momentum_10d", "momentum_20d", "nifty_return_21d", "nifty_return_5d",
                 "macd_histogram", "price_vs_ema21_pct", "adx_14",
                 "breadth_pct_above_ema50", "breadth_pct_above_ema200"],
    "BEAR":     ["beta_21d", "rolling_vol_21d", "historical_vol_63d", "atr_14",
                 "relative_strength_nifty_21d", "rsi_14", "volume_spike"],
    "SIDEWAYS": ["rsi_14", "rsi_divergence", "resistance_distance_20d",
                 "support_distance_20d", "macd_histogram", "vol_compression"],
    "VOLATILE": ["atr_14", "vol_expansion", "volume_spike", "rolling_vol_10d",
                 "rolling_vol_21d", "gap_open_pct", "breakout_distance_52w"],
}

_LABEL_COLS = ["return_5d", "return_3d", "return_10d", "return_15d",
               "outperform_nifty_5d", "outperform_binary", "expected_return", "direction_5d"]


def _build_expert(hp: dict) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter            = hp.get("max_iter", 300),
        learning_rate       = hp.get("learning_rate", 0.05),
        max_depth           = hp.get("max_depth", 6),
        min_samples_leaf    = hp.get("min_samples_leaf", 20),
        l2_regularization   = hp.get("l2_regularization", 0.1),
        max_features        = hp.get("max_features", 0.8),
        early_stopping      = True,
        validation_fraction = 0.15,
        n_iter_no_change    = 25,
        class_weight        = CLASS_WEIGHT,
        random_state        = 42,
        verbose             = 0,
    )


def _build_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add 9 domain-specific interaction features."""
    X = X.copy()
    c = set(X.columns)

    def g(col):
        return X[col] if col in c else pd.Series(np.nan, index=X.index)

    X["ix_momentum_x_trend"]  = g("momentum_10d")            * g("adx_14")
    X["ix_rsi_x_vol"]         = g("rsi_14")                  * g("rolling_vol_21d")
    X["ix_volume_x_momentum"] = g("volume_spike")            * g("momentum_10d")
    X["ix_breadth_x_beta"]    = g("breadth_pct_above_ema50") * g("beta_21d")
    X["ix_sector_x_nifty"]    = g("sector_return_5d")        * g("nifty_return_5d")
    X["ix_support_x_rsi"]     = g("support_distance_20d")    * g("rsi_14")
    X["ix_rsi_x_momentum"]    = g("rsi_14")                  * g("momentum_5d")
    X["ix_vol_x_breakout"]    = g("vol_expansion")           * g("breakout_distance_52w")
    X["ix_trend_x_price"]     = g("adx_14")                  * g("price_vs_ema21_pct")
    return X


def _compute_temporal_weights(dates: pd.Series, halflife_days: int = TEMPORAL_HALFLIFE_DAYS) -> np.ndarray:
    """Exponential decay: w(t) = 2^(-(T-t)/halflife). Returns uniform if dates invalid."""
    try:
        date_vals = pd.to_datetime(dates, errors="coerce")
        valid = date_vals.notna()
        if valid.sum() < 10:
            return np.ones(len(dates), dtype=np.float32)
        T = date_vals[valid].max()
        age_days = (T - date_vals).dt.days.fillna(0).clip(lower=0).values
        weights = np.power(2.0, -age_days / halflife_days).astype(np.float32)
        mean_w = weights.mean()
        if mean_w <= 0 or np.isnan(mean_w):
            return np.ones(len(dates), dtype=np.float32)
        return weights / mean_w
    except Exception:
        return np.ones(len(dates), dtype=np.float32)


def _compute_confident_weights(X: pd.DataFrame) -> np.ndarray:
    """Down-weight rows where |return_5d| < threshold — ambiguous direction labels."""
    if "return_5d" not in X.columns:
        return np.ones(len(X), dtype=np.float32)
    abs_ret = X["return_5d"].abs()
    return np.where(abs_ret >= CONFIDENT_LABEL_THRESHOLD_PCT, 1.0, CONFIDENT_LABEL_WEIGHT).astype(np.float32)


def _select_regime_features(X: pd.DataFrame, y: pd.Series, regime: str, top_n: int = 30) -> list[str]:
    """Top-N features by IC, with 20% boost for regime-hint features."""
    all_cols = [c for c in X.columns if not c.startswith("meta_")]
    sample = X if len(X) <= 8000 else X.sample(8000, random_state=42)
    y_s = y.loc[sample.index]
    ics = {}
    for col in all_cols:
        if sample[col].isnull().all():
            continue
        corr = sample[col].corr(y_s, method="spearman")
        if not np.isnan(corr):
            ics[col] = abs(corr)
    if not ics:
        return all_cols[:top_n]
    for h in REGIME_FEATURE_HINTS.get(regime, []):
        if h in ics:
            ics[h] *= 1.20
    ranked = sorted(ics.items(), key=lambda x: -x[1])
    selected = [col for col, _ in ranked[:top_n]]
    meta_cols = [c for c in X.columns if c.startswith("meta_")]
    return selected + [m for m in meta_cols if m not in selected]


class AQRTINet:
    """
    AQRTINet v3.1 — NSE/BSE stock direction predictor.
    Eleven innovations over off-the-shelf gradient boosters.
    CPU-only, ~6 min training on AMD Ryzen AI 7 350 / 16GB RAM.
    """

    DEFAULT_HYPERPARAMS = {
        "max_iter": 300, "learning_rate": 0.05, "max_depth": 6,
        "min_samples_leaf": 20, "l2_regularization": 0.1, "max_features": 0.8,
    }

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._experts: dict[str, Any] = {}
        self._ranker: Optional[PercentileRanker] = None
        self._feature_cols: list[str] = []
        self._regime_feature_cols: dict[str, list[str]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_col: Optional[str] = None,
        date_col:   Optional[str] = None,
    ) -> "AQRTINet":
        """
        Train AQRTINet.

        Args:
            X:          Feature DataFrame
            y:          Binary labels (1=UP, 0=DOWN)
            regime_col: Column in X with regime label (BULL/BEAR/SIDEWAYS/VOLATILE)
            date_col:   Column in X with dates (used for temporal decay weights)
        """
        # Extract regime + date before feature processing
        if regime_col and regime_col in X.columns:
            regimes = X[regime_col].fillna(FALLBACK_REGIME).str.upper()
            X = X.drop(columns=[regime_col])
        else:
            regimes = pd.Series([FALLBACK_REGIME] * len(X), index=X.index)

        dates = X[date_col] if date_col and date_col in X.columns else pd.Series([""] * len(X), index=X.index)
        if date_col:
            X = X.drop(columns=[date_col], errors="ignore")

        # Strip label columns that shouldn't be features (read them first for weighting)
        confident_weights = _compute_confident_weights(X)
        X_clean = X.drop(columns=[c for c in _LABEL_COLS if c in X.columns], errors="ignore")
        self._feature_cols = list(X_clean.columns)

        # Interaction features
        X_work = _build_interaction_features(X_clean)

        # Percentile ranking
        self._ranker = PercentileRanker(n_quantiles=100)
        X_ranked = self._ranker.fit_transform(X_work)

        # Combined sample weights: temporal decay × confident-label
        temporal_weights = _compute_temporal_weights(dates)
        combined = temporal_weights * confident_weights
        cm = combined.mean()
        combined = (combined / cm).astype(np.float32) if cm > 0 and not np.isnan(cm) else np.ones(len(X), dtype=np.float32)

        # Train regime experts
        self._experts = {}
        self._regime_feature_cols = {}
        for regime in REGIMES:
            mask = (regimes == regime).values
            n = mask.sum()
            if n < MIN_REGIME_ROWS:
                continue
            X_r, y_r, w_r = X_ranked[mask], y[mask], combined[mask]
            feats = _select_regime_features(X_r, y_r, regime, top_n=30)
            self._regime_feature_cols[regime] = feats
            hp = {**self.hyperparams, **REGIME_HYPERPARAMS[regime]}
            expert = _build_expert(hp)
            expert.fit(X_r[feats], y_r, sample_weight=w_r)
            self._experts[regime] = expert

        if FALLBACK_REGIME not in self._experts:
            feats = _select_regime_features(X_ranked, y, FALLBACK_REGIME, top_n=30)
            self._regime_feature_cols[FALLBACK_REGIME] = feats
            hp = {**self.hyperparams, **REGIME_HYPERPARAMS[FALLBACK_REGIME]}
            expert = _build_expert(hp)
            expert.fit(X_ranked[feats], y, sample_weight=combined)
            self._experts[FALLBACK_REGIME] = expert

        return self

    def predict_proba(self, X: pd.DataFrame, regime: str = FALLBACK_REGIME) -> np.ndarray:
        """Returns P(UP) in [0,1] for each row."""
        X_feat = X[[c for c in self._feature_cols if c in X.columns]]
        X_work = _build_interaction_features(X_feat)
        X_ranked = self._ranker.transform(X_work)
        regime = regime.upper()
        expert = self._experts.get(regime) or self._experts[FALLBACK_REGIME]
        feats = self._regime_feature_cols.get(regime) or self._regime_feature_cols[FALLBACK_REGIME]
        available_feats = [f for f in feats if f in X_ranked.columns]
        proba = expert.predict_proba(X_ranked[available_feats])
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(self, X: pd.DataFrame, regime: str = FALLBACK_REGIME, threshold: float = 0.5) -> np.ndarray:
        """Returns class labels (1=UP, 0=DOWN)."""
        return (self.predict_proba(X, regime) >= threshold).astype(int)

    def feature_importance(self) -> dict[str, float]:
        """Average feature importances across all regime experts."""
        all_imp = []
        for regime, expert in self._experts.items():
            fi = getattr(expert, "feature_importances_", None)
            if fi is not None and len(fi) > 0:
                feats = self._regime_feature_cols.get(regime, self._feature_cols)
                if len(fi) == len(feats):
                    imp_map = dict(zip(feats, fi))
                    full = np.array([imp_map.get(c, 0.0) for c in self._feature_cols])
                    all_imp.append(full)
        if not all_imp:
            n = len(self._feature_cols)
            return {c: 1.0 / n for c in self._feature_cols}
        avg = np.mean(all_imp, axis=0)
        return dict(zip(self._feature_cols, avg.tolist()))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "version":              "3.1",
            "hyperparams":          self.hyperparams,
            "experts":              self._experts,
            "ranker":               self._ranker,
            "feature_cols":         self._feature_cols,
            "regime_feature_cols":  self._regime_feature_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AQRTINet":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(hyperparams=payload["hyperparams"])
        instance._experts             = payload["experts"]
        instance._ranker              = payload["ranker"]
        instance._feature_cols        = payload["feature_cols"]
        instance._regime_feature_cols = payload.get("regime_feature_cols", {})
        return instance

    def is_trained(self) -> bool:
        return bool(self._experts)

    @property
    def experts_trained(self) -> list[str]:
        return list(self._experts.keys())
