"""
QuantEdge Configuration
All constants and hyperparameters in one place.
"""

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "JPM", "META", "NFLX", "AMD"
]

FEATURE_COLS = [
    "log_return", "return_5d", "return_10d", "return_20d",
    "rsi_14", "rsi_7", "macd_line", "macd_signal",
    "macd_crossover", "momentum_10", "rolling_vol_10",
    "rolling_vol_20", "bb_width", "bb_position",
    "volume_zscore", "volume_ratio",
    "vix_level", "vix_change", "regime"
]

WINDOW_SIZE = 30           # LSTM sequence length
FORECAST_HORIZON = 5       # days ahead for classification
TRAIN_RATIO = 0.80
CONFIDENCE_THRESHOLD = 0.60
RANDOM_STATE = 42

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1
}

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0
}

DARK_THEME = {
    "bg": "#070d14",
    "surface": "#0a1525",
    "border": "#1e2d40",
    "accent": "#00d4ff",
    "green": "#22c55e",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "purple": "#a78bfa",
    "text": "#e2e8f0",
    "muted": "#556677",
    "font": "IBM Plex Mono"
}

PLOTLY_THEME = dict(
    paper_bgcolor="#0a1525",
    plot_bgcolor="#0a1525",
    font=dict(family="IBM Plex Mono", color="#8899aa", size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e2d40", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2d40", showgrid=True, zeroline=False),
)

CORE_PHILOSOPHY = """
We do not predict stock prices. We predict the PROBABILITY of upward movement —
because that is what is actually learnable from market data.

Price prediction with LSTM fails in practice because stock prices are near-random
walks. Our approach:

  • CLASSIFY direction (up/down) instead of regressing price
  • FORECAST volatility with GARCH (volatility IS learnable)
  • DETECT market regimes with LSTM (sequences of states)
  • EXPLAIN predictions with feature importance (Permutation Importance)
  • TRACK experiments with MLflow (production mindset)

A 58% directional accuracy vs 50% baseline is honest, meaningful, and
statistically significant. That is our edge.
"""
