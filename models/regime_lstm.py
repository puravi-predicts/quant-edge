"""
QuantEdge LSTM Regime Detector
Classifies market regime as BEARISH / SIDEWAYS / BULLISH.
LSTM used for regime DETECTION (sequence classification), NOT price forecasting.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "..")
from config import WINDOW_SIZE, TRAIN_RATIO, PLOTLY_THEME, DARK_THEME

try:
    import tensorflow as tf
    import keras
    from keras import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

REGIME_LABELS = {0: "BEARISH", 1: "SIDEWAYS", 2: "BULLISH"}
REGIME_COLORS = {
    0: DARK_THEME["red"],
    1: DARK_THEME["yellow"],
    2: DARK_THEME["green"],
}


class RegimeLSTM:
    """
    LSTM-based market regime classifier.
    
    Regime definition (based on 20-day rolling return):
        BEARISH  (0): roll_ret < -5%
        SIDEWAYS (1): -5% ≤ roll_ret ≤ +5%
        BULLISH  (2): roll_ret > +5%
    
    Features per timestep: [log_return, rolling_vol_20, rsi_14, vix_level]
    Sequence length: 30 days
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size    = window_size
        self.model          = None
        self.history        = None
        self.feature_cols   = ["log_return", "rolling_vol_20", "rsi_14", "vix_level"]
        self._fitted        = False

    
    # REGIME LABELS

    @staticmethod
    def compute_regime_labels(df: pd.DataFrame) -> pd.Series:
        """
        Compute regime code from 20-day rolling return.
        Returns pd.Series with index matching df.
        """
        roll_ret = df["Close"].pct_change(20)
        regime   = pd.Series(1, index=df.index)   # default sideways
        regime[roll_ret < -0.05]  = 0              # bearish
        regime[roll_ret >  0.05]  = 2              # bullish
        return regime

    @staticmethod
    def get_regime_label(code: int) -> str:
        return REGIME_LABELS.get(int(code), "SIDEWAYS")

    
    # SEQUENCE PREPARATION
    
    def prepare_sequences(self, df: pd.DataFrame):
        """
        Build (X, y) sequences for LSTM.
        X shape: (N, window_size, n_features)
        y shape: (N,) integer regime codes
        
        Returns: X, y, dates
        """
        # Ensure features exist
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing feature column: {col}")

        regime = self.compute_regime_labels(df)
        data   = df[self.feature_cols].copy()
        data["regime"] = regime

        # Drop NaN rows
        data = data.dropna()
        values  = data[self.feature_cols].values
        targets = data["regime"].values
        dates   = data.index

        # Normalize per-feature using training statistics (set outside)
        # We do a simple min-max here; proper scaling happens in the pipeline
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        values = scaler.fit_transform(values)

        X, y, seq_dates = [], [], []
        for i in range(self.window_size, len(values)):
            X.append(values[i - self.window_size: i])
            y.append(targets[i])
            seq_dates.append(dates[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        return X, y, pd.DatetimeIndex(seq_dates)

    
    # BUILD MODEL
    
    def build_model(self, n_features: int) -> "keras.Model":
        """
        LSTM architecture for sequence classification.
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not installed.")

        model = Sequential([
            Input(shape=(self.window_size, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(3, activation="softmax"),
        ])

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        self.model = model
        return model

    
    # TRAIN
    
    def train(self, df: pd.DataFrame):
        """
        Prepare sequences, build model, and train with early stopping.
        Chronological 80/20 split — no shuffling.
        
        Returns: Keras History object
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not installed.")

        X, y, dates = self.prepare_sequences(df)
        n_features  = X.shape[2]

        split = int(len(X) * TRAIN_RATIO)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.build_model(n_features)

        callbacks = [
            EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor="val_accuracy",
                verbose=0,
            ),
            ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                monitor="val_loss",
                verbose=0,
            ),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            shuffle=False,   # chronological order preserved
            verbose=0,
        )
        self._fitted = True
        return self.history

    
    # PREDICT REGIME

    def predict_regime(self, df: pd.DataFrame) -> tuple:
        """
        Predict regime for all dates with enough history.
        
        Returns:
            regime_series: pd.Series (index=dates, values=0/1/2)
            proba_array:   np.ndarray shape (N, 3) — [p_bear, p_side, p_bull]
        """
        if not self._fitted:
            # Fallback: use rolling return rule directly
            regime = self.compute_regime_labels(df)
            regime = regime.dropna()
            n = len(regime)
            proba = np.zeros((n, 3), dtype=np.float32)
            for i, r in enumerate(regime.values):
                proba[i, int(r)] = 1.0
            return regime, proba

        X, y, dates = self.prepare_sequences(df)
        proba = self.model.predict(X, verbose=0)
        codes = np.argmax(proba, axis=1)
        regime_series = pd.Series(codes, index=dates, name="regime_lstm")
        return regime_series, proba

    
    # TRAINING HISTORY PLOT
    
    def plot_training_history(self) -> go.Figure:
        """
        Plot training and validation accuracy/loss over epochs.
        """
        if self.history is None:
            fig = go.Figure()
            fig.update_layout(
                **PLOTLY_THEME,
                title=dict(text="LSTM Training — No History Available",
                           font=dict(color=DARK_THEME["text"])),
            )
            return fig

        h     = self.history.history
        epochs = list(range(1, len(h["accuracy"]) + 1))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Accuracy", "Loss"])

        # Accuracy
        fig.add_trace(go.Scatter(
            x=epochs, y=h["accuracy"],
            mode="lines", name="Train Acc",
            line=dict(color=DARK_THEME["accent"], width=2)),
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=epochs, y=h["val_accuracy"],
            mode="lines", name="Val Acc",
            line=dict(color=DARK_THEME["green"], width=2, dash="dash")),
            row=1, col=1,
        )

        # Loss
        fig.add_trace(go.Scatter(
            x=epochs, y=h["loss"],
            mode="lines", name="Train Loss",
            line=dict(color=DARK_THEME["yellow"], width=2)),
            row=1, col=2,
        )
        fig.add_trace(go.Scatter(
            x=epochs, y=h["val_loss"],
            mode="lines", name="Val Loss",
            line=dict(color=DARK_THEME["red"], width=2, dash="dash")),
            row=1, col=2,
        )

        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(
                text="LSTM Regime Detector — Training History",
                font=dict(color=DARK_THEME["text"])
            ),
            height=360,
            showlegend=True,
            legend=dict(bgcolor=DARK_THEME["surface"],
                        bordercolor=DARK_THEME["border"],
                        font=dict(color=DARK_THEME["text"])),
        )
        return fig

    
    # REGIME CHART
    
    def plot_regime_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        Visualize regime classification across the full price history.
        """
        regime_series, proba = self.predict_regime(df)
        close = df["Close"].reindex(regime_series.index)

        fig = make_subplots(rows=2, cols=1,
                            row_heights=[0.65, 0.35],
                            shared_xaxes=True,
                            vertical_spacing=0.05)

        # Price
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values,
            mode="lines", name="Close Price",
            line=dict(color=DARK_THEME["accent"], width=1.5)),
            row=1, col=1,
        )

        # Color bands by regime
        for code, color, label in [
            (0, "rgba(239,68,68,0.15)",  "BEARISH"),
            (1, "rgba(245,158,11,0.10)", "SIDEWAYS"),
            (2, "rgba(34,197,94,0.12)",  "BULLISH"),
        ]:
            mask = regime_series == code
            starts = mask[mask & ~mask.shift(1, fill_value=False)].index
            ends   = mask[mask & ~mask.shift(-1, fill_value=False)].index
            for s, e in zip(starts, ends):
                fig.add_vrect(x0=s, x1=e, fillcolor=color,
                              layer="below", line_width=0, row=1, col=1)

        # Regime codes
        fig.add_trace(go.Scatter(
            x=regime_series.index, y=regime_series.values,
            mode="lines", name="Regime Code",
            line=dict(color=DARK_THEME["purple"], width=1.5)),
            row=2, col=1,
        )
        fig.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=["BEARISH", "SIDEWAYS", "BULLISH"],
            row=2, col=1,
            tickfont=dict(color=DARK_THEME["text"]),
        )

        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(
                text="Market Regime Classification (LSTM) | Source: yfinance",
                font=dict(color=DARK_THEME["text"])
            ),
            height=500,
            showlegend=True,
            legend=dict(bgcolor=DARK_THEME["surface"],
                        bordercolor=DARK_THEME["border"],
                        font=dict(color=DARK_THEME["text"])),
        )
        return fig
