# ◈ QuantEdge — ML-Powered Stock Signal Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-02569B?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.11%2B-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

> *"We don't predict prices. We predict probability of direction —
> because that's what's actually learnable from market data."*

---

## What This Demonstrates

This is a **production-quality** data science portfolio project built with
the rigor expected at quant shops and MNC technology teams:

- **Binary classification** on financial time series (direction: up/down)
- **19-feature engineering pipeline** spanning momentum, volatility, volume, and macro
- **LightGBM, XGBoost, Logistic Regression** with honest test-set-only metrics
- **LSTM for market regime detection** (Bull / Bear / Sideways) — NOT price prediction
- **GARCH(1,1) volatility forecasting** and VaR / CVaR risk metrics
- **Permutation feature importance** — model-agnostic, unbiased explainability
- **MLflow experiment tracking** across models and 10 tickers
- **Walk-forward backtesting** with Sharpe ratio, max drawdown, monthly P&L heatmap
- **Chronological train/test split** — zero data leakage guaranteed
- **Streamlit production dashboard** — 6 interactive tabs

---

## Core Philosophy

```
We do not predict stock prices. We predict the PROBABILITY of upward
movement — because that is what is actually learnable from market data.

  • CLASSIFY direction (up/down) instead of regressing price
  • FORECAST volatility with GARCH (volatility IS learnable)
  • DETECT market regimes with LSTM (sequences of states)
  • EXPLAIN predictions with feature importance (Permutation)
  • TRACK experiments with MLflow (production mindset)

A 58% directional accuracy vs 50% baseline is honest, meaningful,
and statistically significant. That is our edge.
```

---

## Why Not LSTM for Price Prediction?

This is the most important interview question you'll face on this project.
Here is the honest answer:

**Stock prices are near-random walks.** The Efficient Market Hypothesis
(even in its weak form) implies that past prices contain very little
predictable signal about future prices. When you train an LSTM to predict
tomorrow's price, it learns to copy yesterday's price (persistence) —
not because it understood market dynamics, but because that's the lowest
MSE strategy for a near-random series. On training data, loss looks low.
On test data, the model fails catastrophically because the specific patterns
it memorized never repeat in the same way.

**Classification is fundamentally more learnable.** Predicting *direction*
(up vs down) with features derived from momentum, volume, and volatility
captures genuine statistical regularities — mean reversion after RSI
extremes, volume-confirmed breakouts, volatility regime changes — that
have documented academic evidence behind them. A 58% directional accuracy
over 50% baseline represents a genuine, persistent edge. Showing this
understanding in an interview signals DS maturity that most candidates lack.

---

## Architecture

```
                         ┌─────────────────────────────────┐
                         │        Yahoo Finance API         │
                         │     10 stocks × 3 years OHLCV   │
                         └──────────────┬──────────────────┘
                                        │
                         ┌──────────────▼──────────────────┐
                         │      data/pipeline.py            │
                         │  19 features engineered          │
                         │  Chronological 80/20 split       │
                         │  Scaler fit on TRAIN only        │
                         └──┬──────────────────────────┬───┘
                            │                          │
           ┌────────────────▼────┐         ┌──────────▼────────────┐
           │  models/classifier  │         │  models/regime_lstm   │
           │  LightGBM ★         │         │  LSTM sequence classi-│
           │  XGBoost            │         │  fier: Bull/Bear/Side  │
           │  Logistic (baseline)│         │  window=30, 4 features │
           └────────────┬────────┘         └──────────┬────────────┘
                        │                             │
           ┌────────────▼────────┐                    │
           │ evaluation/metrics  │         ┌──────────▼────────────┐
           │  ROC, CM, Calib,    │         │  models/volatility    │
           │  Rolling accuracy   │         │  GARCH(1,1) AR(1)     │
           └────────────┬────────┘         │  VaR 95/99, CVaR      │
                        │                  └──────────┬────────────┘
           ┌────────────▼────────┐                    │
           │ backtesting/backtest│◄───────────────────┘
           │  Signal > threshold │
           │  Transaction costs  │
           │  Sharpe, drawdown   │
           │  Monthly heatmap    │
           └────────────┬────────┘
                        │
           ┌────────────▼────────────────────────────┐
           │        mlflow_tracking/experiments       │
           │  Log params, metrics, model artifacts    │
           └────────────┬────────────────────────────┘
                        │
           ┌────────────▼────────────────────────────┐
           │              app.py (Streamlit)          │
           │  Tab 1: Live Signal + Regime + Vol       │
           │  Tab 2: Model comparison + ROC + CM      │
           │  Tab 3: Feature importance + insights    │
           │  Tab 4: Equity curve + monthly heatmap   │
           │  Tab 5: VaR / CVaR / GARCH diagnostics  │
           │  Tab 6: MLflow experiment browser        │
           └─────────────────────────────────────────┘
```

---

## Key Results

| Metric                  | Value         |
|-------------------------|---------------|
| Best Model              | LightGBM      |
| Directional Accuracy    | ~57–59%       |
| vs Random Baseline      | +7–9 pp       |
| Backtest Sharpe Ratio   | ~0.5–1.2      |
| Stocks Covered          | 10 (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, META, NFLX, AMD) |
| Features                | 19            |
| Train/Test Split        | 80% / 20%     |
| Forecast Horizon        | 5 days        |

*Actual values vary by ticker and market period. All metrics on test set only.*

---

## How To Run

### 1. Clone the repository
```bash
git clone https://github.com/puravi-predicts/quantedge.git
cd quantedge
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

### 5. Optional: Launch MLflow UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 6. Usage
1. Select a ticker from the sidebar dropdown
2. Adjust the confidence threshold (default 0.60)
3. Click **▸ TRAIN MODELS**
4. Explore all 6 tabs

---

## Tech Stack

| Library        | Version  | Purpose                                      |
|----------------|----------|----------------------------------------------|
| `streamlit`    | ≥1.32    | Interactive dashboard UI                     |
| `yfinance`     | ≥0.2.36  | Market data download (OHLCV + VIX)           |
| `lightgbm`     | ≥4.3     | Gradient boosting classifier (primary model) |
| `xgboost`      | ≥2.0     | Gradient boosting (comparison)               |
| `scikit-learn` | ≥1.4     | Logistic baseline, permutation importance    |
| `tensorflow`   | ≥2.15    | LSTM regime detector                         |
| `arch`         | ≥6.3     | GARCH(1,1) volatility model                  |
| `statsmodels`  | ≥0.14    | Ljung-Box diagnostic test                    |
| `plotly`       | ≥5.19    | All interactive charts                       |
| `mlflow`       | ≥2.11    | Experiment tracking & model registry         |
| `scipy`        | ≥1.12    | Normal distribution for VaR                  |
| `pandas`       | ≥2.2     | Data manipulation                            |
| `numpy`        | =1.26.4  | Numerical operations                         |
| `loguru`       | ≥0.7     | Production-grade logging                     |

---

## Project Structure

```
quantedge/
├── app.py                        # Streamlit dashboard (6 tabs)
├── config.py                     # All constants and hyperparameters
├── requirements.txt              # Pinned dependencies
├── README.md                     # This file
│
├── data/
│   └── pipeline.py               # DataPipeline: download → features → split
│
├── models/
│   ├── classifier.py             # ModelTrainer: LightGBM, XGBoost, Logistic
│   ├── regime_lstm.py            # RegimeLSTM: Bull/Bear/Sideways detector
│   └── volatility.py            # GARCHVolatility: GARCH(1,1) + VaR/CVaR
│
├── backtesting/
│   └── backtest.py              # Backtester: walk-forward strategy simulation
│
├── evaluation/
│   └── metrics.py               # ROC, CM, calibration, rolling accuracy
│
└── mlflow_tracking/
    └── experiments.py           # ExperimentTracker: log all runs
```

---

## Non-Negotiable Design Principles

1. **No price prediction** — classification only
2. **No simulated data** — all data from `yfinance`
3. **Never shuffle time series** — chronological split always
4. **Fit scaler on TRAIN only** — zero data leakage
5. **All metrics on TEST SET only** — never training set
6. **Honest metrics** — 57% shown as 57%, never inflated
7. **GARCH on log returns × 100** — not raw prices
8. **LSTM only for regime classification** — not forecasting
9. **Every chart has title, axis labels, data source**

---

## Interview Talking Points

When a recruiter at **Deloitte, KPMG, JPMorgan, Accenture, or EY** asks:

**"Why not use LSTM to predict prices?"**
→ Explain the random walk argument. Show the LSTM is used for regime detection — a classification task with genuine signal. This demonstrates theoretical understanding, not just tool usage.

**"How did you prevent data leakage?"**
→ Chronological split, scaler fit on train only, no target peeking. Point to the code in `pipeline.py`.

**"How do you know the model actually works?"**
→ Test-set-only metrics. Backtest on held-out data. Statistical comparison against random baseline. Rolling accuracy stability chart.

**"How would you deploy this in production?"**
→ MLflow model registry, containerize with Docker, serve via FastAPI, automate retraining with Airflow. MLflow is already integrated here as a starting point.

---

## License

MIT — free to use, modify, and showcase in your portfolio.

---

*Built with the philosophy that honest, rigorous data science beats inflated accuracy metrics every time.*
