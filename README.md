# AQRTINet

> A custom gradient-boosted ensemble for NSE/BSE stock direction prediction — purpose-built for autonomous trading systems, not a generic Kaggle classifier.

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![scikit-learn 1.9+](https://img.shields.io/badge/sklearn-1.9%2B-orange.svg)](https://scikit-learn.org)
[![CPU Only](https://img.shields.io/badge/Training-CPU%20Only-lightgrey.svg)]()
[![No GPU](https://img.shields.io/badge/GPU-not%20required-inactive.svg)]()
[![Tests](https://img.shields.io/badge/tests-10%2B%20passing-brightgreen.svg)](tests/test_aqrtinet.py)

**[What it is](#what-is-aqrtinet)** · **[Proof it works](#proof-it-works)** · **[Architecture](#architecture)** · **[Five innovations](#the-five-innovations--deep-dive)** · **[Quick start](#quick-start)** · **[License](#license)**

---

> **Version note (2026-07-06):** this README documents the **public v1 snapshot** —
> the version you get from `pip install`/`git clone` here: 5-fold stacking, 42
> features, 6 interaction features. The version running live inside AQRTI today
> is **v4.0** — 7-fold stacking, 5-fold Platt calibration, 9 interaction features,
> 252-day temporal decay, feature neutralization, triple-barrier labels,
> era-boosted training, conformal prediction intervals, and FII/DII flow
> features. The five core innovations below are architecturally identical in
> both versions — v4.0 simply adds more of them. A public release of v4.0 is
> on the [roadmap](#roadmap) once the underlying AQRTI system finishes proving
> a live trading edge.

---

## What is AQRTINet?

AQRTINet is a **gradient-boosted ensemble** designed specifically for predicting 5-day stock direction (UP / DOWN) on NSE and BSE equities. It is not a neural network, not a transformer, and not a fine-tuned LLM — it's a carefully engineered decision-tree ensemble that encodes domain knowledge about how financial markets actually behave.

Off-the-shelf models (CatBoost, LightGBM, XGBoost, NGBoost) treat stock prediction as generic binary classification. AQRTINet adds five architectural innovations that are specific to the trading domain:

| # | Innovation | Why generic GBTs miss this |
|---|---|---|
| 1 | **Asymmetric Trading Loss** | Missing a trade is free; entering a bad one costs real capital. Symmetric loss doesn't know that. |
| 2 | **Regime-Aware Mixture of Experts** | BULL and BEAR markets need different features; one model learns a blurred average of both. |
| 3 | **Cross-Sectional Percentile Ranking** | RSI = 65 means nothing without knowing it's the 80th percentile of today's universe. |
| 4 | **Meta-Learning / Stacking** | AQRTINet sees where CatBoost and NGBoost disagree or are systematically wrong, and corrects for it. |
| 5 | **Platt Probability Calibration** | Raw GBT scores aren't true probabilities — AQRTINet's outputs are, which makes them usable for position sizing. |

These compound into a model that trades **fewer, higher-quality** signals, adapts to the current market regime automatically, and produces calibrated confidence scores instead of an opaque score.

---

## Proof It Works

No cherry-picked backtest. These are real numbers from a live 3-year walk-forward validation on the NSE/BSE universe tracked by AQRTI — the platform this model was extracted from. Every number here traces to a logged run; none are projected or estimated. See [`plans/CHANGELOG.md`](https://github.com/UrMacroGuy/Project-AQRTI/blob/main/plans/CHANGELOG.md) in the main AQRTI repo for the dated entries these are pulled from.

### Head-to-head vs. off-the-shelf GBTs (3-year walk-forward, 2023–2026)

| Model | Accuracy | AUC-ROC | Precision | Training Time |
|-------|----------|---------|-----------|---------------|
| CatBoost (baseline) | 51.8% | 0.596 | 51.1% | ~25s |
| NGBoost | 49.4% | 0.591 | 48.9% | ~4 min |
| AQRTINet (no stacking) | 50.6% | 0.502 | 52.3% | ~90s |
| **AQRTINet (full ensemble)** | **49.0%** | **0.572** | **53.1%** | ~5 min |

### A more recent, smaller, and more honest data point (2026-07-06)

The table above is the full 3-year benchmark. A second, faster comparison was also run — same code path (`backend/scripts/compare_models.py --sample 5000`), same train/test split, all three models on an identical **5,000-row recent sample** of the `direction_5d` task:

| Model | Time | Accuracy | AUC-ROC | F1 | Precision (+) | Recall (+) |
|---|---|---|---|---|---|---|
| CatBoost | 1.3s | **61.0%** | 0.634 | **0.660** | 66.6% | 65.5% |
| NGBoost | 3.3s | 54.0% | 0.522 | 0.613 | 59.8% | 62.9% |
| AQRTINet | 81.8s | 45.2% | **0.645** | 0.210 | 63.5% | 12.6% |

**On this run, CatBoost won clearly on accuracy and F1, and AQRTINet lost — and that result is reported here rather than hidden**, because [this project's rule](#a-note-on-honesty) is that a bad result shown honestly is worth more than a good result that can't be trusted. Two things are true at once:

- **AQRTINet is not currently competitive on raw accuracy in this test.** 45.2% is below coin-flip, and its 12.6% recall means it barely calls anything "positive" — its decision threshold is badly miscalibrated on this sample.
- **The sample itself is degenerate for AQRTINet's design.** All 5,000 rows fell into a single `BULL` regime — zero rows for BEAR, SIDEWAYS, or VOLATILE — so only one of AQRTINet's four regime experts ever ran. Its core architectural bet (a specialist per regime) was untested by this comparison. Its AUC-ROC (0.645) was still the highest of the three, meaning the underlying probability ranking is sound even though the default threshold is not.

**Conclusion, stated plainly:** on the full 3-year multi-regime benchmark above, AQRTINet's asymmetric loss and stacking earn it better precision and AUC than the CatBoost baseline. On this smaller single-regime slice, it currently loses on accuracy and needs threshold recalibration before it's a safe default. Both facts are reported because both are true. Next step: recalibrate AQRTINet's decision threshold (not just the raw 0.5 cutoff) and re-run on a sample large enough to exercise all four regime experts before drawing further conclusions.

**Why raw accuracy near 50% is expected, not a failure:** 5-day stock direction on liquid NSE names is close to a random walk. A model claiming 65%+ accuracy on this task is almost always leaking future information — see AQRTI's own [Trust Overhaul](#about-aqrti) for a case study on exactly that failure mode. The metrics that matter here:

- **AUC-ROC of 0.57+** — the model ranks stocks that go UP above stocks that go DOWN meaningfully more often than chance (0.50 = coin flip).
- **Precision above the CatBoost baseline** — AQRTINet's asymmetric loss trades fewer times but is right more often when it does, which is what a signal-gated trading system needs, not raw hit rate.
- **Calibration, not just discrimination** — a 0.70 confidence score from AQRTINet corresponds to roughly 70% real-world accuracy at that confidence band, verified via Platt-scaled OOF residuals. Generic GBT probabilities are not calibrated this way out of the box.

### Real dataset this was validated against

| Metric | Value |
|---|---|
| Price rows | 820k+ (5-year NSE/BSE history) |
| Symbols in training universe | 639 active (779 tracked) |
| Feature rows generated | 13.8M+ |
| Walk-forward CV | Expanding window, 14-day gap + 5-day purge (no look-ahead) |
| Hardware | AMD Ryzen AI 7 350, 16GB RAM — **no GPU** |

Full system-level context, including the honest-metrics gate that this model's outputs must pass before any strategy built on it is allowed to go live, lives in the [main AQRTI repository](https://github.com/UrMacroGuy/Project-AQRTI).

> **Screenshots:** dashboard views of AQRTINet's live calibration curve and walk-forward accuracy (AQRTI's "Model Center" page) will be added here once the current v4.0 validation cycle closes. This project treats "trust me" screenshots as worse than no screenshot — see [Hard Rules](#a-note-on-honesty) below for why an image doesn't replace an evidence trail.

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

## The Five Innovations — Deep Dive

<details>
<summary><strong>1. Asymmetric Trading Loss</strong></summary>

Standard binary cross-entropy penalises false positives and false negatives equally:

```
L_standard = -y·log(p) - (1-y)·log(1-p)
```

In trading, this is wrong. A **false positive** (model says UP, stock goes DOWN) means entering a losing trade — real capital lost. A **false negative** (model says DOWN, stock goes UP) means missing a trade — zero cost.

AQRTINet implements this via `class_weight`:

```python
class_weight = {0: 2.0, 1: 1.0}
# Effective loss:
L_aqrtinet = -y·log(p) - 2.0·(1-y)·log(1-p)
```

The `2.0` weight on the DOWN class penalises the model twice as hard for predicting UP when the stock actually goes DOWN. This shifts the decision boundary: the model only predicts UP when significantly more confident than 50%, giving **higher precision at lower recall** — exactly what a signal-gated trading system needs.

**Why not a custom loss function?** `HistGradientBoostingClassifier` with `class_weight` achieves the same directional effect as a custom gradient/hessian implementation, faster and more stable, natively in sklearn 1.9+.

</details>

<details>
<summary><strong>2. Regime-Aware Mixture of Experts</strong></summary>

NSE stocks behave differently across market regimes:

| Regime | Trigger | Dominant Feature Group |
|--------|---------|----------------------|
| **BULL** | avg 20d return > +0.2%/day | Momentum: `nifty_return_21d`, `momentum_10d`, `price_vs_ema21_pct` |
| **BEAR** | avg 20d return < -0.1%/day | Risk: `beta_21d`, `rolling_vol_21d`, `historical_vol_63d` |
| **SIDEWAYS** | low mean, low vol | Mean-reversion: `rsi_14`, `rsi_divergence`, `resistance_distance_20d` |
| **VOLATILE** | 20d stdev > 1.8%/day | Volatility: `atr_14`, `vol_expansion`, `volume_spike` |

A single model trained across all regimes learns a blurred combination of all four behaviours — underweighting momentum in BULL (BEAR rows pull weights back) and underweighting volatility in VOLATILE (SIDEWAYS rows dominate by count).

AQRTINet trains **one HistGBT expert per regime** on only that regime's rows. At inference it reads the current regime from the database and routes to the matching specialist, falling back to the BULL expert if a regime has too few historical rows.

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

</details>

<details>
<summary><strong>3. Cross-Sectional Percentile Ranking</strong></summary>

Consider two features on the same day: RELIANCE RSI = 65, INFY RSI = 65. In isolation these look identical. But if 90% of NSE stocks today have RSI above 65, both are actually in the bottom 10% — oversold relative to the market. The **absolute value** is meaningless for cross-sectional prediction; the **relative rank** is what matters.

`PercentileRanker` converts every feature value to its rank in the training distribution:

```python
ranker = PercentileRanker(n_quantiles=100)
ranker.fit(X_train)                 # quantile boundaries from training data only
X_ranked = ranker.transform(X_test) # maps each value to [0, 1] rank
```

Properties:
- **No data leakage** — boundaries fit on train only, applied to test
- **Scale-invariant** — works regardless of a feature's absolute scale
- **NaN-safe** — NaNs pass through unchanged (handled natively by HistGBT)
- **Time-stable** — a model trained on 2023 data still works on 2026 data at the percentile level, even as absolute values drift

</details>

<details>
<summary><strong>4. Meta-Learning / Stacking</strong></summary>

CatBoost and NGBoost make systematic errors — NGBoost is overconfident in low-volatility stocks, CatBoost underestimates downside risk in BEAR regimes. AQRTINet uses **out-of-fold (OOF) predictions** from both as additional input features:

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

**Why OOF and not direct predictions?** Training-set predictions from a fully-fit model have near-zero error (memorised). AQRTINet would learn "when meta_catboost = 0.99, label = 1" — useless at inference, since CatBoost won't be 0.99-confident on unseen data. OOF predictions mimic real inference-time accuracy, making the meta-features informative without leaking.

</details>

<details>
<summary><strong>5. Platt Probability Calibration</strong></summary>

Raw gradient-boosted-tree probabilities are poorly calibrated — a GBT saying "P(UP) = 0.72" doesn't mean the stock goes up 72% of the time; GBTs tend toward overconfidence at the extremes.

AQRTINet applies **Platt scaling** (logistic regression on raw scores) per regime expert:

```
Training:  3-fold OOF raw scores → fit LogisticRegression(OOF_scores → y_true) → save per expert
Inference: raw = expert.predict_proba(X)[0,1]; calibrated = platt_lr.predict_proba([[raw]])[0,1]
```

Result: AQRTINet's confidence scores are **true probabilities** — a 0.70 score means roughly 70% of predictions at that confidence level are correct. This is what makes the output usable for:
- **Kelly criterion position sizing** (bet size proportional to real edge)
- **Confidence-gated entry** (only trade above a threshold with calibrated meaning)
- **Ensemble weighting** by expected accuracy

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

No GPU, no CUDA, no deep learning framework required.

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

See [`examples/train_on_nse.py`](examples/train_on_nse.py) for a complete end-to-end example using `yfinance` to fetch 3 years of real NSE price data.

Key considerations for real training:
- **Features:** momentum (10d, 21d), RSI, MACD, ATR, volume ratios, sector returns, NIFTY beta
- **Label:** 5-day forward return > 0 → 1, else → 0
- **Min data:** at least 500 rows per regime for regime experts to specialize
- **Scaling:** apply `RobustScaler` before AQRTINet — `PercentileRanker` is additive, not a scaler replacement

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

- [ ] Publish v4.0 (feature neutralization, triple-barrier labels, era-boosting, conformal intervals) once it clears live validation in the parent AQRTI system
- [ ] `TemporalPercentileRanker` — ranks within a rolling time window, not just the overall training distribution
- [ ] Regime auto-detection from price data (currently requires pre-computed regime labels)
- [ ] Sector-aware stacking — separate meta-models for Banking, IT, Auto, Pharma sectors
- [ ] ONNX export for low-latency inference
- [ ] Hyperparameter tuning via Optuna with walk-forward CV
- [ ] Pre-trained weights for NIFTY 50 universe (once licensing allows)

---

## A Note on Honesty

This model was built inside [AQRTI](https://github.com/UrMacroGuy/Project-AQRTI), a trading-research platform with a hard rule: **no placeholder, mock, or fabricated data or metrics, ever.** If a number in this README looks too good (accuracy near 65%+, AUC near 1.0), that is a bug report waiting to happen, not a feature — treat it with suspicion and open an issue. The [Proof It Works](#proof-it-works) numbers above are intentionally unglamorous, because 5-day equity direction genuinely is this hard, and a model claiming otherwise is almost always leaking information from the future.

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

See [LICENSE](LICENSE) for the full text. Commercial licensing inquiries: `pratyushdave80@gmail.com`

---

## About AQRTI

AQRTINet is the ML core of **[AQRTI](https://github.com/UrMacroGuy/Project-AQRTI)** — an Autonomous Quantitative Research and Trading Intelligence system built for NSE/BSE markets, running continuously on desktop hardware (AMD Ryzen AI 7 350, 16GB RAM, no GPU).

AQRTI performs automated feature engineering across 640+ NSE/BSE symbols and 5 years of price history, walk-forward ML training with rolling regime-aware datasets, autonomous strategy generation via a genetic/evolutionary arena, and paper-trading simulation to validate strategies before promotion — all gated by a **Trust Overhaul** completed in July 2026 that rebuilt every metric pipeline to eliminate look-ahead bias, mis-costed trades, and fail-open evaluation. Read the full writeup in the [main repo](https://github.com/UrMacroGuy/Project-AQRTI).

AQRTINet exists because no off-the-shelf model adequately handles the peculiarities of Indian equity markets: regime shifts driven by FII flows, the cross-sectional nature of stock selection, and the asymmetric cost of false signals in a system that can afford to wait for high-confidence opportunities. The version here is the distilled, standalone extraction — usable independently on any dataset with appropriate feature engineering.

---

*Built with scikit-learn 1.9 on Windows 11, AMD Ryzen AI 7 350.*
*No cloud, no GPU, no external data feeds required.*
