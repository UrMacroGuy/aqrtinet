# AQRTINet

> A custom ML model for NSE/BSE stock direction prediction — purpose-built for autonomous trading systems.

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![scikit-learn 1.9+](https://img.shields.io/badge/sklearn-1.9%2B-orange.svg)](https://scikit-learn.org)
[![CPU Only](https://img.shields.io/badge/Training-CPU%20Only-lightgrey.svg)]()

---

> **Version note (2026-07-05):** this README describes an earlier snapshot
> of the architecture (5-fold stacking, 42 features, 6 interaction
> features). The version running inside AQRTI today is v3.1: 7-fold
> stacking, 5-fold Platt calibration (3-fold below 500 samples), 9
> interaction features (6 original + 3 new), 252-day temporal decay
> half-life, and 0.3× down-weighting of ambiguous-direction training rows.
> The core five innovations below are still accurate in spirit — the
> specific fold counts and feature counts have moved on since this public
> snapshot was written.

## What is AQRTINet?

AQRTINet is a **gradient-boosted ensemble** designed specifically for predicting 5-day stock direction (UP / DOWN) on NSE and BSE equities. It is not a neural network, not a transformer, and not a fine-tuned LLM. It is a carefully engineered decision-tree ensemble that encodes domain knowledge about how financial markets actually behave.

Off-the-shelf models (CatBoost, LightGBM, XGBoost, NGBoost) treat stock prediction as generic binary classification. AQRTINet adds five architectural innovations that are specific to the trading domain:

1. **Asymmetric Trading Loss** — penalties are not symmetric; missing a trade is free, entering a bad trade is expensive
2. **Regime-Aware Mixture of Experts** — BULL markets and BEAR markets require different features; a single model learns a blurred average of both
3. **Cross-Sectional Percentile Ranking** — raw feature values are meaningless without context; RSI of 65 means nothing unless you know it is the 80th percentile of today's universe
4. **Meta-Learning / Stacking** — AQRTINet sees where CatBoost and NGBoost disagree or are wrong, and learns to correct their systematic errors
5. **Platt Probability Calibration** — raw GBT probability scores are not well-calibrated; AQRTINet outputs true P(UP) values using isotonic regression

These five innovations compound. The result is a model that:
- Makes **fewer but higher-quality** trade entry signals (precision bias)
- Adapts to the current **market regime** automatically
- Produces **calibrated confidence scores** usable for position sizing
- Corrects the **known failure modes** of other models in the ensemble

---

## Architecture

```
Input Features (40 technical indicators)
         │
         ▼
┌─────────────────────────────────────────┐
│  Stage 0: Meta-Feature Generation       │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │   CatBoost   │  │    NGBoost      │  │
│  │  (5-fold OOF)│  │  (5-fold OOF)  │  │
│  └──────┬───────┘  └───────┬─────────┘  │
│         └──────┬───────────┘            │
│          P(UP)_catboost, P(UP)_ngboost  │
└─────────────────────────────────────────┘
         │  42 features total
         ▼
┌─────────────────────────────────────────┐
│  Stage 1: Cross-Sectional Ranking       │
│  PercentileRanker                       │
│  Maps each feature value to its         │
│  rank (0–1) in the training             │
│  distribution. Scale-invariant.         │
│  No leakage: boundaries fitted on       │
│  training data only.                    │
└─────────────────────────────────────────┘
         │  42 ranked features [0, 1]
         ▼
┌─────────────────────────────────────────┐
│  Stage 2: Regime Router                 │
│  Reads current regime from DB:          │
│  BULL │ BEAR │ SIDEWAYS │ VOLATILE      │
│  Routes input to matching expert.       │
└─────────────────────────────────────────┘
         │  regime label
         ▼
┌─────────────────────────────────────────┐
│  Stage 3: Regime Expert                 │
│  HistGradientBoostingClassifier         │
│  class_weight = {0: 2.0, 1: 1.0}       │
│  (asymmetric loss: FP penalized 2×)     │
│  Trained on only the rows matching      │
│  the current market regime              │
└─────────────────────────────────────────┘
         │  raw P(UP)
         ▼
┌─────────────────────────────────────────┐
│  Stage 4: Platt Calibration             │
│  Logistic regression on 3-fold OOF      │
│  raw scores → calibrated P(UP)          │
└─────────────────────────────────────────┘
         │
         ▼
    P(UP) ∈ [0, 1]  (well-calibrated)
```

---

## The Five Innovations — Deep Dive

### 1. Asymmetric Trading Loss

Standard binary cross-entropy penalises false positives and false negatives equally:

```
L_standard = -y·log(p) - (1-y)·log(1-p)
```

In trading, this is wrong. A **false positive** (model says UP, stock goes DOWN) means we enter a losing trade — real capital lost. A **false negative** (model says DOWN, stock goes UP) means we miss a trade — zero cost.

AQRTINet implements this via `class_weight`:

```python
class_weight = {0: 2.0, 1: 1.0}

# Effective loss:
L_aqrtinet = -y·log(p) - 2.0·(1-y)·log(1-p)
```

The `2.0` weight on the DOWN class means the model is penalised twice as hard for predicting UP when the stock goes DOWN. This shifts the decision boundary: the model only predicts UP when it is significantly more confident than 50%. The result is **higher precision at lower recall** — exactly what a signal-gated trading system needs.

**Why not a custom loss function?** `HistGradientBoostingClassifier` with `class_weight` achieves the same directional effect as a custom gradient/hessian implementation. It is faster, more stable, and available natively in sklearn 1.9+.

---

### 2. Regime-Aware Mixture of Experts

NSE stocks exhibit clearly different statistical behaviour in different market regimes:

| Regime | Trigger | Dominant Feature Group |
|--------|---------|----------------------|
| **BULL** | avg 20d return > +0.2%/day | Momentum: `nifty_return_21d`, `momentum_10d`, `price_vs_ema21_pct` |
| **BEAR** | avg 20d return < -0.1%/day | Risk: `beta_21d`, `rolling_vol_21d`, `historical_vol_63d` |
| **SIDEWAYS** | low mean, low vol | Mean-reversion: `rsi_14`, `rsi_divergence`, `resistance_distance_20d` |
| **VOLATILE** | 20d stdev > 1.8%/day | Volatility: `atr_14`, `vol_expansion`, `volume_spike` |

A single model trained across all regimes learns a blurred combination of all four behaviours. It will underweight momentum in BULL (because BEAR rows pull the weights back) and underweight volatility in VOLATILE (because SIDEWAYS rows dominate by count).

AQRTINet trains **one HistGBT expert per regime** on only the rows from that regime. At inference, it reads the current regime from the database and routes the input to the matching specialist. If the current regime has no specialist (too few historical rows), it falls back to the BULL expert.

**Regime computation** (from NIFTY50 index returns):
```python
def _price_regime(mean_20d_return, stdev_20d_return):
    if stdev_20d_return > 1.8:  # % per day
        return "VOLATILE"
    if mean_20d_return > 0.2:
        return "BULL"
    if mean_20d_return < -0.1:
        return "BEAR"
    return "SIDEWAYS"
```

---

### 3. Cross-Sectional Percentile Ranking

Consider two features on the same day:
- **RELIANCE RSI = 65**
- **INFY RSI = 65**

In isolation, both are the same. But if today 90% of NSE stocks have RSI above 65, both are actually in the bottom 10% — oversold relative to the market. If instead 90% of stocks have RSI below 65, both are in the top 10% — overbought.

The **absolute value** is meaningless for cross-sectional prediction. The **relative rank** is what matters.

`PercentileRanker` converts every feature value to its rank in the training distribution:

```python
ranker = PercentileRanker(n_quantiles=100)
ranker.fit(X_train)     # computes per-feature quantile boundaries from training data
X_ranked = ranker.transform(X_test)   # maps each value to [0, 1] rank
```

Key properties:
- **No data leakage**: quantile boundaries are fitted on training data only, applied to test
- **Scale-invariant**: works identically regardless of the absolute scale of features
- **NaN-safe**: NaN values pass through unchanged (handled by HistGBT natively)
- **Time-stable**: rank distributions shift slowly; a model trained on 2023 data still works on 2026 data at the 70th-percentile level, even if absolute values have drifted

---

### 4. Meta-Learning / Stacking

CatBoost and NGBoost make systematic errors. They consistently get certain patterns wrong:
- NGBoost is overconfident in low-volatility stocks
- CatBoost underestimates downside risk in BEAR regimes

AQRTINet uses **out-of-fold (OOF) predictions** from both models as additional input features:

```
Training phase:
  1. Split X_train into 5 folds
  2. For each fold:
     - Train CatBoost on 4 folds
     - Predict P(UP) on held-out fold
  3. Concatenate 5 OOF prediction arrays → meta_catboost (shape: n_train,)
  4. Repeat for NGBoost → meta_ngboost
  5. Stack [X_train | meta_catboost | meta_ngboost] → X_train_augmented
  6. Train AQRTINet regime experts on X_train_augmented

Inference phase:
  - Load latest CatBoost pkl → predict P(UP) on input → append as meta_catboost
  - Load latest NGBoost pkl → predict P(UP) on input → append as meta_ngboost
  - Pass [X | meta_catboost | meta_ngboost] through PercentileRanker → regime expert
```

**Why OOF and not direct predictions?**

If you train CatBoost on the full training set and use its training-set predictions as meta-features, those predictions have near-zero error (the model memorised training data). AQRTINet would learn "when meta_catboost = 0.99, the label is 1" which is useless at inference (CatBoost won't be 0.99 confident on new data). OOF predictions mimic the actual accuracy CatBoost will have at inference time, making the meta-features informative without leaking.

---

### 5. Platt Probability Calibration

Raw gradient boosted tree probabilities are poorly calibrated. A GBT saying "P(UP) = 0.72" does not mean the stock goes up 72% of the time — GBTs tend to be overconfident near the extremes.

AQRTINet applies **Platt scaling** (logistic regression on raw scores) per regime expert:

```
Training:
  1. For each regime expert:
     a. 3-fold OOF: get raw GBT probability scores on training data (out-of-fold)
     b. Fit LogisticRegression on OOF_scores → y_true
     c. Save Platt logistic regression as part of the expert

Inference:
  raw_score = expert.predict_proba(X)[0, 1]         # raw GBT output
  calibrated = platt_lr.predict_proba([[raw_score]])[0, 1]  # calibrated
```

The result: **the confidence scores produced by AQRTINet are true probabilities** (a score of 0.70 means roughly 70% of predictions at that confidence level are correct). This makes AQRTINet's output directly usable for:
- **Kelly criterion position sizing** (bet size proportional to true edge)
- **Confidence-gated entry** (only enter when P(UP) > threshold with calibrated meaning)
- **Ensemble weighting** (models with calibrated outputs can be weighted by expected accuracy)

---

## Installation

```bash
git clone https://github.com/pratyushdave80/aqrtinet.git
cd aqrtinet
pip install -r requirements.txt
pip install -e .
```

**Requirements:**
```
scikit-learn>=1.9.0
numpy>=1.24.0
pandas>=2.0.0
```

No GPU required. No CUDA. No deep learning frameworks.

---

## Quick Start

```python
import pandas as pd
import numpy as np
from aqrtinet import AQRTINet

# Prepare your data
# X must be a DataFrame with numeric features (NaN allowed)
# y must be binary: 0 = DOWN, 1 = UP
X_train = pd.DataFrame(np.random.randn(1000, 40), columns=[f"feat_{i}" for i in range(40)])
y_train = pd.Series(np.random.randint(0, 2, 1000))
X_test  = pd.DataFrame(np.random.randn(200, 40),  columns=[f"feat_{i}" for i in range(40)])

# Train
model = AQRTINet(task="direction", label_col="direction_5d")
model.fit(X_train, y_train)

# Predict
probabilities = model.predict_proba(X_test)   # calibrated P(UP) per sample
signals       = model.predict(X_test)          # 0 or 1

# Feature importance (averaged across regime experts)
importance = model.feature_importance()
top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
print("Top 5 features:", top5)

# Save / Load
path = model.save()
model_loaded = AQRTINet.load(path)
```

---

## Training on Real NSE Data

See `examples/train_on_nse.py` for a complete example using `yfinance` to fetch 3 years of NSE price data.

Key considerations for real training:
- **Features**: momentum (10d, 21d), RSI, MACD, ATR, volume ratios, sector returns, NIFTY beta
- **Label**: 5-day forward return > 0 → 1, else → 0
- **Min data**: at least 500 rows per regime for regime experts to specialize
- **Scale**: apply `RobustScaler` before passing to AQRTINet (the PercentileRanker is additive, not a replacement for scaling)

---

## Model Parameters

```python
AQRTINet(
    task="direction",         # "direction" (classification) or "expected_return" (regression)
    label_col="direction_5d", # target column name
    hyperparams={
        "max_iter":          300,   # max boosting rounds per expert
        "learning_rate":     0.05,  # step size (lower = slower, more robust)
        "max_depth":         6,     # tree depth (6 = deep enough for interactions)
        "min_samples_leaf":  20,    # min samples per leaf (prevents overfitting)
        "l2_regularization": 0.1,   # L2 regularization on leaf weights
        "max_features":      0.8,   # column subsampling per split
    },
    version=1,                # version tag for artifact naming
)
```

---

## Performance on NSE Universe (Internal Benchmarks)

Tested on the backtest-eligible NSE/BSE universe (see the main AQRTI project's docs for how this count is defined and its current value — it has changed since this benchmark was run), 3-year walk-forward validation (2023–2026):

| Model | Accuracy | AUC-ROC | Precision | Training Time |
|-------|----------|---------|-----------|---------------|
| CatBoost (baseline) | 51.8% | 0.596 | 51.1% | ~25s |
| NGBoost | 49.4% | 0.591 | 48.9% | ~4 min |
| AQRTINet (no stacking) | 50.6% | 0.502 | 52.3% | ~90s |
| **AQRTINet (full)** | **49.0%** | **0.572** | **53.1%** | ~5 min |

Notes:
- Raw accuracy is near-random (50%) across all models — this is expected and correct for financial data
- **AUC-ROC** is the real signal: 0.57+ means the model ranks stocks that go UP higher than stocks that go DOWN
- **Precision** (fraction of UP signals that are actually UP) is what matters for trading; AQRTINet's asymmetric loss shifts it higher than CatBoost
- Training time includes stacking OOF and Platt calibration

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
│   ├── model.py               ← AQRTINet class (standalone, no AQRTI imports)
│   ├── percentile.py          ← PercentileRanker
│   └── loss.py                ← Loss function documentation + formulas
├── examples/
│   ├── train_on_nse.py        ← End-to-end training with yfinance
│   └── predict_example.py
└── tests/
    └── test_aqrtinet.py       ← 10+ tests including precision-bias verification
```

---

## Roadmap

- [ ] Add `TemporalPercentileRanker` — ranks within a rolling time window, not just the overall training distribution
- [ ] Regime auto-detection from price data (currently requires pre-computed regime labels)
- [ ] Sector-aware stacking — separate meta-models for Banking, IT, Auto, Pharma sectors
- [ ] ONNX export for low-latency inference
- [ ] Hyperparameter tuning via Optuna with walk-forward CV
- [ ] Pre-trained weights for NIFTY 50 universe (once licensing allows)

---

## Contributing

This project is source-available under BUSL-1.1. You may read, fork, and modify the code for **non-commercial, non-production** purposes. For production or commercial use, contact `pratyushdave80@gmail.com` for a commercial license.

Pull requests that improve documentation, tests, or examples (without changing core model behavior) are welcome.

---

## License

AQRTINet is licensed under the **Business Source License 1.1 (BUSL-1.1)**.

**In plain English:**
- You CAN read, fork, study, and run this code locally
- You CAN use it for personal research, education, or non-commercial projects
- You CANNOT use it in a live trading system, commercial product, or revenue-generating service without a commercial license
- The license converts to **Apache 2.0** on **January 1, 2030**

See [LICENSE](LICENSE) for the full text.

Commercial licensing inquiries: `pratyushdave80@gmail.com`

---

## About AQRTI

AQRTINet is the ML core of **AQRTI** — an Autonomous Quantitative Research and Trading Intelligence system built for NSE/BSE markets.

AQRTI runs continuously on desktop hardware (AMD Ryzen AI 7 350, 16GB RAM, no GPU) and performs:
- **Automated feature engineering** from 640+ NSE/BSE symbols and 5 years of price history
- **Walk-forward ML training** with rolling regime-aware datasets
- **Autonomous strategy generation** via a genetic/evolutionary arena (population size varies as evolution runs and gates tighten/loosen — see the main AQRTI project's dashboard for the current count, not a number frozen here)
- **Paper trading simulation** to validate strategies before promotion
- **Self-learning loop** — the system detects its own prediction failures, retrains models, and adjusts confidence thresholds automatically

AQRTINet was created because no off-the-shelf model adequately handles the peculiarities of Indian equity markets: the regime shifts driven by FII flows, the cross-sectional nature of stock selection, and the asymmetric cost of false signals in a system that can afford to wait for high-confidence opportunities.

The model you see here is the distilled, standalone version — extractable from AQRTI's internal infrastructure and usable independently on any dataset with appropriate feature engineering.

---

*Built with scikit-learn 1.9 on Windows 11, AMD Ryzen AI 7 350.*
*No cloud, no GPU, no external data feeds required.*
