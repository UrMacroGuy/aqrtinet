"""
AQRTINet — Asymmetric Trading Loss

WHY ASYMMETRIC LOSS?
====================
Standard binary cross-entropy penalises false positives and false negatives equally:

    L(y, p) = -y·log(p) - (1-y)·log(1-p)

In stock trading, the costs are NOT equal:

    False Positive  (predicted UP, stock went DOWN) = real capital loss
    False Negative  (predicted DOWN, stock went UP) = missed gain, no loss

A system that can stay in cash has no cost for missing opportunities.
But every wrong entry costs brokerage + slippage + capital tied up in a loser.

THE ASYMMETRIC LOSS
===================
We want the model to be more cautious — only predict UP when it's confident.

    L(y, p) = -y·log(p) - α·(1-y)·log(1-p)

where α > 1 penalises false positives α× harder than false negatives.
AQRTINet uses α = 2.0.

IMPLEMENTATION
==============
scikit-learn's HistGradientBoostingClassifier does not support custom loss
functions directly, but class_weight={0: α, 1: 1.0} achieves the same effect:

    class_weight={0: 2.0, 1: 1.0}

This reweights the gradient contributions at each leaf:
  - Samples where y=0 (DOWN) get 2× gradient weight
  - Samples where y=1 (UP) get 1× gradient weight

The model learns a decision boundary that is more conservative —
it will only predict UP when the evidence is substantially stronger than
the standard 0.5 threshold would require.

PRACTICAL EFFECT
================
With α=2.0 and the standard 0.5 probability threshold, the effective
precision-recall trade-off shifts to roughly:

    Standard log-loss (α=1):  Precision ≈ Recall (balanced)
    Asymmetric (α=2):         Precision ↑, Recall ↓

Expected outcome: fewer trades, but higher win rate per trade.
This is the right trade-off when:
  - Position sizing is fixed (5% per trade)
  - Brokerage + slippage is a real cost
  - Cash is not a penalty (we don't lose money by not trading)

TUNING α
=========
α=1.0   → standard log-loss, equal FP/FN costs
α=1.5   → mild precision bias
α=2.0   → AQRTINet default (2× FP penalty)
α=3.0   → aggressive precision bias (very few signals, very high win rate)

Higher α → fewer trades, higher precision, lower recall.
The right α depends on your brokerage costs and capital deployment pressure.
For AQRTI's NSE paper trading (low capital, fixed 5% position size): α=2.0.
"""

ALPHA = 2.0
CLASS_WEIGHT = {0: ALPHA, 1: 1.0}
