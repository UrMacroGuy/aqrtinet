"""
AQRTINet — Basic sanity tests.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import pathlib

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from aqrtinet import AQRTINet, PercentileRanker


def make_data(n=300, f=15, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame(np.random.randn(n, f), columns=[f"feat_{i}" for i in range(f)])
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


class TestPercentileRanker:
    def test_fit_transform_shape(self):
        X, _ = make_data()
        ranker = PercentileRanker()
        X_ranked = ranker.fit_transform(X)
        assert X_ranked.shape == X.shape

    def test_values_in_range(self):
        X, _ = make_data()
        ranker = PercentileRanker()
        X_ranked = ranker.fit_transform(X)
        assert X_ranked.min().min() >= 0.0
        assert X_ranked.max().max() <= 1.0

    def test_nan_preserved(self):
        X, _ = make_data(n=100, f=3)
        X.iloc[5, 0] = np.nan
        ranker = PercentileRanker()
        X_ranked = ranker.fit_transform(X)
        assert np.isnan(X_ranked.iloc[5, 0])

    def test_unknown_column_passthrough(self):
        X, _ = make_data(n=100, f=3)
        ranker = PercentileRanker()
        ranker.fit(X)
        X_new = X.copy()
        X_new["unknown_feat"] = 99.9
        X_ranked = ranker.transform(X_new)
        # Unknown col should pass through raw
        assert (X_ranked["unknown_feat"] == 99.9).all()

    def test_not_fitted_raises(self):
        X, _ = make_data()
        ranker = PercentileRanker()
        with pytest.raises(RuntimeError):
            ranker.transform(X)


class TestAQRTINet:
    def test_fit_predict_shape(self):
        X, y = make_data()
        model = AQRTINet()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (len(X),)

    def test_probas_in_range(self):
        X, y = make_data()
        model = AQRTINet()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_predict_binary(self):
        X, y = make_data()
        model = AQRTINet()
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_regime_routing(self):
        X, y = make_data(n=600)
        regimes = pd.Series(
            ["BULL"] * 200 + ["BEAR"] * 200 + ["SIDEWAYS"] * 200,
            index=X.index,
        )
        X_with_regime = X.copy()
        X_with_regime["regime"] = regimes

        model = AQRTINet()
        model.fit(X_with_regime, y, regime_col="regime")

        # All three experts should be trained (200 rows each > MIN_REGIME_ROWS=80)
        assert "BULL" in model.experts_trained
        assert "BEAR" in model.experts_trained
        assert "SIDEWAYS" in model.experts_trained

        # Predictions should differ by regime
        p_bull = model.predict_proba(X[:10], regime="BULL")
        p_bear = model.predict_proba(X[:10], regime="BEAR")
        assert not np.allclose(p_bull, p_bear), "BULL and BEAR experts should differ"

    def test_fallback_regime(self):
        X, y = make_data(n=200)
        model = AQRTINet()
        model.fit(X, y)
        # VOLATILE not in training → should use BULL fallback without error
        probas = model.predict_proba(X[:5], regime="VOLATILE")
        assert probas.shape == (5,)

    def test_feature_importance_sums_to_one(self):
        X, y = make_data()
        model = AQRTINet()
        model.fit(X, y)
        imp = model.feature_importance()
        assert abs(sum(imp.values()) - 1.0) < 1e-6

    def test_save_load_consistency(self):
        X, y = make_data()
        model = AQRTINet()
        model.fit(X, y)
        p1 = model.predict_proba(X[:20])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = pathlib.Path(f.name)

        try:
            model.save(path)
            model2 = AQRTINet.load(path)
            p2 = model2.predict_proba(X[:20])
            assert np.allclose(p1, p2, atol=1e-6), "Predictions differ after save/load"
        finally:
            path.unlink(missing_ok=True)

    def test_is_trained(self):
        X, y = make_data()
        model = AQRTINet()
        assert not model.is_trained()
        model.fit(X, y)
        assert model.is_trained()

    def test_asymmetric_loss_shifts_precision(self):
        """
        With asymmetric loss (FP penalised 2×), precision on UP class should be
        >= precision from a standard balanced model at the same threshold.
        This is a statistical test with fixed seed — not guaranteed always but
        holds with high probability over 500 rows.
        """
        np.random.seed(0)
        n = 500
        # Weak signal: add some noise
        X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
        # True signal in f0
        y = pd.Series((X["f0"] + np.random.randn(n) * 0.5 > 0).astype(int))

        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import precision_score

        split = 400
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # Standard model
        std_model = HistGradientBoostingClassifier(random_state=42)
        std_model.fit(X_tr, y_tr)
        std_preds = std_model.predict(X_te)

        # AQRTINet (asymmetric)
        model = AQRTINet()
        model.fit(X_tr, y_tr)
        aqrti_preds = model.predict(X_te)

        std_prec  = precision_score(y_te, std_preds,  zero_division=0)
        aqrti_prec = precision_score(y_te, aqrti_preds, zero_division=0)

        # AQRTINet should have >= precision (it's designed to be more conservative)
        # Allow small slack since both are approximate
        assert aqrti_prec >= std_prec - 0.05, (
            f"AQRTINet precision ({aqrti_prec:.3f}) should be >= standard ({std_prec:.3f})"
        )
