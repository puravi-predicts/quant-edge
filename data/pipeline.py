"""
QuantEdge Data Pipeline
Downloads, engineers features, creates targets, splits, and scales.
All time-series safe: no data leakage, chronological split only.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from loguru import logger
import sys

sys.path.insert(0, "..")
from config import FEATURE_COLS, FORECAST_HORIZON, TRAIN_RATIO, RANDOM_STATE

logger.add("quantedge_pipeline.log", rotation="10 MB")


class DataPipeline:
    """
    Full data pipeline from raw OHLCV → model-ready arrays.
    
    Usage:
        pipeline = DataPipeline("AAPL", period="3y")
        X_train, X_test, y_train, y_test, feature_names, dates_test, scaler, df_full = pipeline.run()
    """

    def __init__(self, ticker: str, period: str = "3y"):
        self.ticker = ticker
        self.period = period
        self.scaler = StandardScaler()
        self.df_full = None

    
    # 1. DOWNLOAD
    
    def download(self) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance."""
        logger.info(f"Downloading {self.ticker} ({self.period})")
        
        df = yf.download(self.ticker, period=self.period, auto_adjust=True, progress=False)
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        logger.info(f"Downloaded {len(df)} rows for {self.ticker}")
        return df

    
    # 2. FEATURE ENGINEERING
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 18 features (regime added by LSTM later, set to 1 here as placeholder)."""
        df = df.copy()
        close = df["Close"]
        volume = df["Volume"]

        # ── Price Momentum 
        df["log_return"] = np.log(close / close.shift(1))
        df["return_5d"]  = np.log(close / close.shift(5))
        df["return_10d"] = np.log(close / close.shift(10))
        df["return_20d"] = np.log(close / close.shift(20))

        # ── RSI — Wilder's EWM method 
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)

        avg_gain_14 = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss_14 = loss.ewm(alpha=1/14, adjust=False).mean()
        rs_14 = avg_gain_14 / avg_loss_14
        df["rsi_14"] = 100 - (100 / (1 + rs_14))

        avg_gain_7 = gain.ewm(alpha=1/7, adjust=False).mean()
        avg_loss_7 = loss.ewm(alpha=1/7, adjust=False).mean()
        rs_7 = avg_gain_7 / avg_loss_7
        df["rsi_7"] = 100 - (100 / (1 + rs_7))

        # ── MACD 
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd_line"]   = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_crossover"] = (
            (df["macd_line"] > df["macd_signal"]) &
            (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
        ).astype(int)
        df["momentum_10"] = close / close.rolling(10).mean() - 1

        # ── Volatility 
        df["rolling_vol_10"] = df["log_return"].rolling(10).std() * np.sqrt(252)
        df["rolling_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

        # ── Bollinger Bands (20-period, 2 std) 
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_width"]    = (bb_upper - bb_lower) / bb_mid
        df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        df["bb_position"] = df["bb_position"].clip(0, 1)

        # ── Volume 
        vol_mean_20 = volume.rolling(20).mean()
        vol_std_20  = volume.rolling(20).std()
        df["volume_zscore"] = (volume - vol_mean_20) / vol_std_20
        df["volume_ratio"]  = volume / volume.rolling(10).mean()

        # ── Macro: VIX 
        try:
            vix = yf.download("^VIX", period=self.period, auto_adjust=True, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix["Close"].reindex(df.index, method="ffill")
            df["vix_level"]  = vix_close
            df["vix_change"] = vix_close.pct_change()
        except Exception as e:
            logger.warning(f"VIX download failed: {e}. Using zeros.")
            df["vix_level"]  = 20.0
            df["vix_change"] = 0.0

        # ── Regime placeholder (will be updated after LSTM) 
        # 0=Bearish, 1=Sideways, 2=Bullish
        roll_ret = close.pct_change(20)
        df["regime"] = 1  # default sideways
        df.loc[roll_ret < -0.05, "regime"] = 0
        df.loc[roll_ret > 0.05,  "regime"] = 2

        return df

    
    # 3. TARGET VARIABLE
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: 1 if price is higher FORECAST_HORIZON days from now, else 0.
        Drop last FORECAST_HORIZON rows (no future data available).
        """
        df = df.copy()
        future_close  = df["Close"].shift(-FORECAST_HORIZON)
        future_return = (future_close - df["Close"]) / df["Close"]
        df["target"]  = (future_return > 0).astype(int)

        # Drop rows where target is NaN (last FORECAST_HORIZON rows)
        df = df.iloc[:-FORECAST_HORIZON].copy()
        logger.info(f"Target created. Class balance: {df['target'].value_counts().to_dict()}")
        return df

    
    # 4. TRAIN / TEST SPLIT + SCALING

    def split_and_scale(self, df: pd.DataFrame):
        """
        Chronological train/test split.
        Scaler fit ONLY on train set to prevent data leakage.
        """
        df = df.dropna(subset=FEATURE_COLS + ["target"]).copy()

        split_idx = int(len(df) * TRAIN_RATIO)
        train = df.iloc[:split_idx]
        test  = df.iloc[split_idx:]

        feature_names = FEATURE_COLS

        X_train_raw = train[feature_names].values
        X_test_raw  = test[feature_names].values
        y_train     = train["target"].values
        y_test      = test["target"].values
        dates_test  = test.index

        # Fit scaler on train ONLY — critical for no leakage
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test  = self.scaler.transform(X_test_raw)

        logger.info(
            f"Split complete | Train: {len(train)} rows | Test: {len(test)} rows | "
            f"Test class balance: {pd.Series(y_test).value_counts().to_dict()}"
        )

        return (
            X_train, X_test,
            y_train, y_test,
            feature_names, dates_test,
            self.scaler, df
        )

    
    # 5. FULL PIPELINE IN ONE CALL
    
    def run(self):
        """
        Execute full pipeline:
          download → engineer_features → create_target → split_and_scale
        
        Returns:
            X_train, X_test, y_train, y_test,
            feature_names, dates_test, scaler, df_full
        """
        df = self.download()
        df = self.engineer_features(df)
        df = self.create_target(df)
        self.df_full = df

        result = self.split_and_scale(df)
        logger.success(f"Pipeline complete for {self.ticker}")
        return result
