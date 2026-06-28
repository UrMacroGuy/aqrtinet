"""
AQRTINet — Minimal NSE training example using yfinance.

Fetches 2 years of NIFTY50 stocks, computes basic features,
trains AQRTINet with regime routing, and evaluates precision/recall.

Requirements:
    pip install yfinance scikit-learn pandas numpy
    pip install aqrtinet  # or: python -m pip install -e ..
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import classification_report, roc_auc_score

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from aqrtinet import AQRTINet

# ── Config ──────────────────────────────────────────────────────────────
SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "SBIN.NS", "WIPRO.NS", "AXISBANK.NS",
]
YEARS = 2
LABEL_DAYS = 5    # predict 5-day return direction

# ── Download prices ──────────────────────────────────────────────────────
print(f"Downloading {len(SYMBOLS)} stocks ({YEARS}y)...")
raw = yf.download(SYMBOLS, period=f"{YEARS}y", auto_adjust=True, progress=False)
closes = raw["Close"].dropna(how="all")

# ── Feature engineering ──────────────────────────────────────────────────
def compute_features(closes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym in closes.columns:
        c = closes[sym].dropna()
        if len(c) < 60:
            continue
        df = pd.DataFrame(index=c.index)
        df["symbol"] = sym
        df["close"]  = c

        # Returns
        df["ret_1d"]  = c.pct_change(1)
        df["ret_5d"]  = c.pct_change(5)
        df["ret_21d"] = c.pct_change(21)

        # Momentum
        df["mom_10d"] = c.pct_change(10)

        # RSI-14
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

        # EMA deviation
        ema21 = c.ewm(span=21).mean()
        df["price_vs_ema21"] = (c - ema21) / ema21

        # Volume ratio
        if "Volume" in raw.columns.get_level_values(0):
            vol = raw["Volume"][sym].reindex(c.index).fillna(0)
            df["vol_ratio_20d"] = vol / vol.rolling(20).mean().replace(0, 1)
        else:
            df["vol_ratio_20d"] = 1.0

        # Volatility
        df["vol_10d"] = c.pct_change().rolling(10).std() * np.sqrt(252)

        # Label: 5-day forward return > 0
        df["label"] = (c.pct_change(LABEL_DAYS).shift(-LABEL_DAYS) > 0).astype(int)

        # Simple regime: BULL if 21d return > 1%, BEAR if < -1%, else SIDEWAYS
        def regime(r):
            if r > 0.01:   return "BULL"
            if r < -0.01:  return "BEAR"
            return "SIDEWAYS"
        df["regime"] = df["ret_21d"].apply(lambda x: regime(x) if pd.notna(x) else "SIDEWAYS")

        rows.append(df.dropna())

    return pd.concat(rows, ignore_index=True)

print("Computing features...")
data = compute_features(closes)
print(f"Dataset: {len(data)} rows, {data['symbol'].nunique()} stocks")

# ── Train / test split (chronological) ──────────────────────────────────
data = data.sort_values("Date" if "Date" in data.columns else data.index.name or "index")
split = int(len(data) * 0.8)
train = data.iloc[:split]
test  = data.iloc[split:]

feature_cols = ["ret_1d", "ret_5d", "ret_21d", "mom_10d", "rsi_14",
                "price_vs_ema21", "vol_ratio_20d", "vol_10d"]

X_train = train[feature_cols]
y_train = train["label"]
X_test  = test[feature_cols]
y_test  = test["label"]

# ── Train AQRTINet ───────────────────────────────────────────────────────
print("\nTraining AQRTINet...")
model = AQRTINet()

# Add regime column to X for routing (we pass it, model strips it during fit)
X_train_with_regime = X_train.copy()
X_train_with_regime["regime"] = train["regime"].values

model.fit(X_train_with_regime, y_train, regime_col="regime")
print(f"Experts trained: {model.experts_trained}")

# ── Evaluate ─────────────────────────────────────────────────────────────
# Route each test row to its regime expert
test_regimes = test["regime"].values
probas = np.zeros(len(X_test))

for i, (regime, (_, row)) in enumerate(zip(test_regimes, X_test.iterrows())):
    probas[i] = model.predict_proba(row.to_frame().T, regime=regime)[0]

preds = (probas >= 0.5).astype(int)

print(f"\n=== AQRTINet Results ===")
print(f"Test rows:  {len(y_test)}")
print(f"AUC-ROC:    {roc_auc_score(y_test, probas):.4f}")
print(classification_report(y_test, preds, target_names=["DOWN", "UP"]))

# Show precision at different confidence thresholds
print("Precision at confidence thresholds:")
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = probas >= thresh
    if mask.sum() > 0:
        prec = y_test[mask].mean()
        print(f"  >= {thresh:.0%}: {mask.sum():4d} signals, precision {prec:.1%}")

# Save
model.save("aqrtinet_nse_example.pkl")
print("\nModel saved to aqrtinet_nse_example.pkl")
