# AQRTINet

A gradient-boosted ensemble for 5-day stock direction prediction on NSE/BSE equities — an applied research project exploring whether trading-domain-specific structure (regime experts, asymmetric loss, cross-sectional ranking) can beat off-the-shelf gradient boosters on real market data.

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![scikit-learn 1.9+](https://img.shields.io/badge/sklearn-1.9%2B-orange.svg)](https://scikit-learn.org)
[![CPU Only](https://img.shields.io/badge/Training-CPU%20Only-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/tests-10%2B%20passing-brightgreen.svg)](tests/test_aqrtinet.py)
[![Status: Research](https://img.shields.io/badge/Status-Research%20%2F%20Case%20Study-yellow.svg)]()

**[Overview](#overview)** · **[Status](#status)** · **[Benchmarks](#benchmarks)** · **[Architecture](#architecture)** · **[Design details](#design-details)** · **[Quick start](#quick-start)** · **[License](#license)**

---

## Status

**This is a research project and case study, not an actively-maintained production model.** [AQRTI](https://github.com/UrMacroGuy/Project-AQRTI), the trading intelligence platform this was built for, now runs plain **CatBoost** in production — it consistently matched or beat AQRTINet on accuracy/AUC-ROC at a fraction of the training cost (see [Benchmarks](#benchmarks)), and a single well-understood model is easier to trust for real capital than a more complex one that doesn't clearly win.

That outcome is itself the useful finding: five plausible trading-domain improvements over a stock gradient booster, honestly measured, mostly didn't pay for their added complexity — and one of them (regime routing) had a real bug for months before an evaluation harness fix surfaced it. Both the wins and the negative results are documented below, code included, in case they save someone else the time.

---

## Overview

AQRTINet is a gradient-boosted ensemble, not a neural network or transformer. It treats 5-day direction prediction as a domain-specific problem rather than generic binary classification, and differs from off-the-shelf models (CatBoost, LightGBM, XGBoost, NGBoost) in five respects:

| # | Component | Purpose |
|---|---|---|
| 1 | Asymmetric trading loss | Penalizes false positives (bad trade entries) more than false negatives (missed trades) |
| 2 | Regime-aware mixture of experts | Trains a separate specialist per market regime (BULL/BEAR/SIDEWAYS/VOLATILE) instead of one blended model |
| 3 | Cross-sectional percentile ranking | Converts absolute feature values to within-universe percentile ranks before training |
| 4 | Meta-learning / stacking | Uses out-of-fold predictions from CatBoost and NGBoost as additional input features |
| 5 | Platt probability calibration | Produces calibrated probabilities suitable for position sizing, not raw uncalibrated scores |

---

## Benchmarks

Two independent evaluation runs, both reported in full — including the one where this model underperformed a baseline.

### 3-year walk-forward validation, full NSE universe

| Model | Accuracy | AUC-ROC | Precision | Training Time |
|-------|----------|---------|-----------|---------------|
| CatBoost (baseline) | 51.8% | 0.596 | 51.1% | ~25s |
| NGBoost | 49.4% | 0.591 | 48.9% | ~4 min |
| AQRTINet (no stacking) | 50.6% | 0.502 | 52.3% | ~90s |
| AQRTINet (full ensemble) | 49.0% | 0.572 | 53.1% | ~5 min |

Raw accuracy near 50% is expected for this task — 5-day equity direction on liquid names is close to a random walk, and claims of substantially higher accuracy usually indicate information leakage. The relevant metrics are AUC-ROC (ranking quality) and precision (signal quality when the model does act), both of which AQRTINet leads on this benchmark.

### Fast comparison on a smaller, single-regime sample

| Model | Time | Accuracy | AUC-ROC | F1 | Precision (+) | Recall (+) |
|---|---|---|---|---|---|---|
| CatBoost | 1.3s | 61.0% | 0.634 | 0.660 | 66.6% | 65.5% |
| NGBoost | 3.3s | 54.0% | 0.522 | 0.613 | 59.8% | 62.9% |
| AQRTINet | 81.8s | 45.2% | 0.645 | 0.210 | 63.5% | 12.6% |

On this run, AQRTINet underperformed CatBoost on accuracy and F1. The sample (5,000 recent rows) contained only one market regime (BULL), meaning three of AQRTINet's four regime experts were never exercised, and its default decision threshold was not well calibrated for this slice. Its AUC-ROC remained the highest of the three, indicating the underlying ranking was still sound.

### Postmortem: two real bugs the benchmarks above were hiding

Following up on the threshold-calibration note above surfaced two structural bugs in the evaluation path — worth documenting since they're a good example of how a model can look worse (or falsely better) than it actually is for reasons that have nothing to do with its architecture:

**1. Regime mis-routing during historical evaluation.** `predict()`/`predict_proba()` routed every row in a batch to whichever regime was "current" in the database *right now*, rather than the regime that was actually in effect on each row's historical date. That's correct for live single-day inference (there's only one "today"), but wrong for a backtest spanning years of dates across all four regimes — nearly every row got scored by the wrong specialist expert using the wrong decision threshold. Fixed by threading per-row dates through inference so historical evaluation can look up each row's regime from the same map used at training time, falling back to "current regime" only when no dates are supplied.

**2. A decision-threshold selector that could pick a degenerate cutoff.** The per-regime F1-optimal threshold sweep maximized plain F1, which — on a positive-majority label — is maximized by a threshold that calls almost every row "UP" (high recall, mediocre precision, near-zero real discrimination). After the routing fix above, exactly this happened: 100% recall on a 61%-positive test set, which is barely better than always guessing the majority class. Fixed by switching the sweep to balanced accuracy and explicitly rejecting any threshold where fewer than 2% or more than 98% of predictions land in one class.

Neither bug was in the core ideas (regime experts, percentile ranking, stacking) — both were in the plumbing around them. It's a reminder that an evaluation harness bug and a real architectural weakness produce the same symptom (worse benchmark numbers), and it's worth ruling out the former before concluding the latter.

---

## Architecture

```
Input Features (40 technical indicators)
         │
         ▼
┌─────────────────────────────────────────┐
│  Stage 0: Meta-Feature Generation        │
│  ┌──────────────┐  ┌─────────────────┐   │
│  │   CatBoost   │  │    NGBoost      │   │
│  │  (5-fold OOF)│  │  (5-fold OOF)   │   │
│  └──────┬───────┘  └───────┬─────────┘   │
│         └──────┬───────────┘             │
│          P(UP)_catboost, P(UP)_ngboost   │
└─────────────────────────────────────────┘
         │  42 features total
         ▼
┌─────────────────────────────────────────┐
│  Stage 1: Cross-Sectional Ranking        │
│  PercentileRanker — maps each feature    │
│  value to its rank (0–1) in the training │
│  distribution. Scale-invariant, no       │
│  leakage (boundaries fit on train only). │
└─────────────────────────────────────────┘
         │  42 ranked features [0, 1]
         ▼
┌─────────────────────────────────────────┐
│  Stage 2: Regime Router                  │
│  Reads current regime: BULL │ BEAR │     │
│  SIDEWAYS │ VOLATILE. Routes to the      │
│  matching expert.                        │
└─────────────────────────────────────────┘
         │  regime label
         ▼
┌─────────────────────────────────────────┐
│  Stage 3: Regime Expert                  │
│  HistGradientBoostingClassifier          │
│  class_weight = {0: 2.0, 1: 1.0}         │
│  (asymmetric loss: FP penalized 2×)      │
│  Trained only on rows from the current   │
│  market regime.                          │
└─────────────────────────────────────────┘
         │  raw P(UP)
         ▼
┌─────────────────────────────────────────┐
│  Stage 4: Platt Calibration              │
│  Logistic regression on 3-fold OOF       │
│  raw scores → calibrated P(UP)           │
└─────────────────────────────────────────┘
         │
         ▼
    P(UP) ∈ [0, 1]  (well-calibrated)
```

---

## Design Details

<details>
<summary><strong>1. Asymmetric Trading Loss</strong></summary>

Standard binary cross-entropy penalizes false positives and false negatives equally:

```
L_standard = -y·log(p) - (1-y)·log(1-p)
```

In trading this is incorrect. A false positive (model says UP, stock goes DOWN) means entering a losing trade — real capital lost. A false negative (model says DOWN, stock goes UP) means missing a trade — zero cost.

AQRTINet implements this via `class_weight`:

```python
class_weight = {0: 2.0, 1: 1.0}
# Effective loss:
L_aqrtinet = -y·log(p) - 2.0·(1-y)·log(1-p)
```

The `2.0` weight on the DOWN class penalizes the model twice as hard for predicting UP when the stock actually goes DOWN, shifting the decision boundary toward higher precision at lower recall.

`HistGradientBoostingClassifier` with `class_weight` achieves the same directional effect as a custom loss implementation, with better performance and stability, natively in sklearn 1.9+.

</details>

<details>
<summary><strong>2. Regime-Aware Mixture of Experts</strong></summary>

NSE stocks behave differently across market regimes:

| Regime | Trigger | Dominant Feature Group |
|--------|---------|----------------------|
| BULL | avg 20d return > +0.2%/day | Momentum: `nifty_return_21d`, `momentum_10d`, `price_vs_ema21_pct` |
| BEAR | avg 20d return < -0.1%/day | Risk: `beta_21d`, `rolling_vol_21d`, `historical_vol_63d` |
| SIDEWAYS | low mean, low vol | Mean-reversion: `rsi_14`, `rsi_divergence`, `resistance_distance_20d` |
| VOLATILE | 20d stdev > 1.8%/day | Volatility: `atr_14`, `vol_expansion`, `volume_spike` |

A single model trained across all regimes learns a blurred average of all four behaviors. AQRTINet trains one HistGBT expert per regime on only that regime's rows, and routes inference to the matching specialist based on the current regime, falling back to the BULL expert when a regime has insufficient history.

```python
def _price_regime(mean_20d_return, stdev_20d_return):
    if stdev_20d_return > 1.8:
        return "VOLATILE"
    if mean_20d_return > 0.2:
        return "BULL"
    if mean_20d_return < -0.1:
        return "BEAR"
    return "SIDEWAYS"
```

</details>

<details>
<summary><strong>3. Cross-Sectional Percentile Ranking</strong></summary>

An absolute feature value is not meaningful without context — an RSI of 65 could be the 80th percentile of today's universe or the 20th, depending on market-wide conditions. `PercentileRanker` converts every feature value to its rank in the training distribution:

```python
ranker = PercentileRanker(n_quantiles=100)
ranker.fit(X_train)                 # quantile boundaries from training data only
X_ranked = ranker.transform(X_test) # maps each value to [0, 1] rank
```

Properties: no data leakage (boundaries fit on train only), scale-invariant, NaN-safe, and stable over time as absolute feature values drift.

</details>

<details>
<summary><strong>4. Meta-Learning / Stacking</strong></summary>

CatBoost and NGBoost make systematic, model-specific errors. AQRTINet uses out-of-fold (OOF) predictions from both as additional input features:

```
Training:
  1. Split X_train into 5 folds
  2. For each fold: train CatBoost on the other 4, predict on the held-out fold
  3. Concatenate OOF predictions → meta_catboost
  4. Repeat for NGBoost → meta_ngboost
  5. Stack [X_train | meta_catboost | meta_ngboost] → augmented training set
  6. Train AQRTINet regime experts on the augmented set

Inference:
  Load CatBoost + NGBoost → predict on input → append as meta features
  → PercentileRanker → regime expert
```

OOF predictions are used rather than direct fitted predictions because training-set predictions from a fully-fit model have near-zero error and would not generalize as meta-features. OOF predictions mimic real inference-time accuracy.

</details>

<details>
<summary><strong>5. Platt Probability Calibration</strong></summary>

Raw gradient-boosted-tree probabilities are typically poorly calibrated, tending toward overconfidence at the extremes. AQRTINet applies Platt scaling (logistic regression on raw scores) per regime expert:

```
Training:  3-fold OOF raw scores → fit LogisticRegression(OOF_scores → y_true) → save per expert
Inference: raw = expert.predict_proba(X)[0,1]; calibrated = platt_lr.predict_proba([[raw]])[0,1]
```

This makes AQRTINet's confidence scores usable directly for Kelly-criterion position sizing, confidence-gated entry, and ensemble weighting by expected accuracy.

</details>

---

## Installation

```bash
git clone https://github.com/UrMacroGuy/aqrtinet.git
cd aqrtinet
pip install -r requirements.txt
pip install -e .
```

```
scikit-learn>=1.9.0
numpy>=1.24.0
pandas>=2.0.0
```

No GPU or deep learning framework required.

---

## Quick Start

```python
import pandas as pd
import numpy as np
from aqrtinet import AQRTINet

# X must be a DataFrame with numeric features (NaN allowed)
# y must be binary: 0 = DOWN, 1 = UP
X_train = pd.DataFrame(np.random.randn(1000, 40), columns=[f"feat_{i}" for i in range(40)])
y_train = pd.Series(np.random.randint(0, 2, 1000))
X_test  = pd.DataFrame(np.random.randn(200, 40),  columns=[f"feat_{i}" for i in range(40)])

model = AQRTINet(task="direction", label_col="direction_5d")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)   # calibrated P(UP) per sample
signals       = model.predict(X_test)          # 0 or 1

importance = model.feature_importance()
top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
print("Top 5 features:", top5)

path = model.save()
model_loaded = AQRTINet.load(path)
```

See [`examples/train_on_nse.py`](examples/train_on_nse.py) for an end-to-end example using `yfinance` to fetch NSE price data.

Notes for training on real data:
- Features: momentum (10d, 21d), RSI, MACD, ATR, volume ratios, sector returns, NIFTY beta
- Label: 5-day forward return > 0 → 1, else → 0
- Minimum ~500 rows per regime for regime experts to specialize
- Apply `RobustScaler` before AQRTINet — `PercentileRanker` is additive, not a scaler replacement

---

## Model Parameters

```python
AQRTINet(
    task="direction",         # "direction" (classification) or "expected_return" (regression)
    label_col="direction_5d", # target column name
    hyperparams={
        "max_iter":          300,
        "learning_rate":     0.05,
        "max_depth":         6,
        "min_samples_leaf":  20,
        "l2_regularization": 0.1,
        "max_features":      0.8,
    },
    version=1,
)
```

---

## Directory Structure

```
aqrtinet/
├── LICENSE                    ← BUSL-1.1 (source-available, non-commercial)
├── README.md
├── requirements.txt
├── setup.py
├── aqrtinet/
│   ├── __init__.py            ← from aqrtinet.model import AQRTINet
│   ├── model.py                 AQRTINet class (standalone)
│   ├── percentile.py            PercentileRanker
│   └── loss.py                  Loss function documentation
├── examples/
│   ├── train_on_nse.py          End-to-end training with yfinance
│   └── predict_example.py
└── tests/
    └── test_aqrtinet.py          Test suite
```

---

## Roadmap

Not under active development — this repo is preserved as a research reference, not a live project. If you fork it, ideas worth exploring:

- [ ] `TemporalPercentileRanker` — ranks within a rolling window rather than the full training distribution
- [ ] Regime auto-detection from price data
- [ ] Sector-aware stacking
- [ ] ONNX export for low-latency inference
- [ ] Hyperparameter tuning via Optuna with walk-forward CV
- [ ] Pre-trained weights for the NIFTY 50 universe

---

## Contributing

This project is source-available under BUSL-1.1. You may read, fork, and modify the code for non-commercial, non-production purposes. For production or commercial use, contact `pratyushdave80@gmail.com` for a commercial license. Pull requests that improve documentation, tests, or examples without changing core model behavior are welcome.

---

## License

Licensed under the Business Source License 1.1 (BUSL-1.1).

- Free to read, fork, study, and run locally
- Free for personal research, education, and non-commercial projects
- Not licensed for use in a live trading system, commercial product, or revenue-generating service without a commercial license
- Converts to Apache 2.0 on January 1, 2030

See [LICENSE](LICENSE) for the full text. Commercial licensing inquiries: `pratyushdave80@gmail.com`

---

*Trained and validated with scikit-learn 1.9. CPU only — no GPU or cloud infrastructure required.*
