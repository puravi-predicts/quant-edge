"""
◈ QuantEdge — ML-Powered Stock Signal Engine
Streamlit dashboard: 6 tabs covering signals, model evaluation,
feature importance, backtesting, risk engine, and experiment tracking.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Path setup 
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    TICKERS, FEATURE_COLS, DARK_THEME, PLOTLY_THEME,
    CORE_PHILOSOPHY, CONFIDENCE_THRESHOLD
)

# ── Page config 
st.set_page_config(
    page_title="QUANTEDGE | ML Signal Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS 
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  /* Base */
  html, body, [class*="css"] {
      font-family: 'IBM Plex Mono', monospace !important;
      background-color: #070d14 !important;
      color: #e2e8f0 !important;
  }
  .main { background-color: #070d14; }
  section[data-testid="stSidebar"] {
      background-color: #0a1525 !important;
      border-right: 1px solid #1e2d40;
  }
  /* Metric cards */
  [data-testid="metric-container"] {
      background: #0a1525;
      border: 1px solid #1e2d40;
      border-radius: 8px;
      padding: 12px 16px;
  }
  [data-testid="stMetricLabel"]  { color: #556677 !important; font-size: 11px; }
  [data-testid="stMetricValue"]  { color: #e2e8f0 !important; font-size: 22px; }
  [data-testid="stMetricDelta"]  { font-size: 12px; }
  /* Buttons */
  .stButton > button {
      background: #00d4ff !important;
      color: #070d14 !important;
      font-family: 'IBM Plex Mono', monospace !important;
      font-weight: 600 !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 10px 24px !important;
      width: 100%;
  }
  .stButton > button:hover { background: #00b8d9 !important; }
  /* Tabs */
  [data-testid="stTabs"] button {
      font-family: 'IBM Plex Mono', monospace !important;
      color: #556677 !important;
      font-size: 12px !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
      color: #00d4ff !important;
      border-bottom: 2px solid #00d4ff !important;
  }
  /* Tables */
  .stDataFrame { background: #0a1525; }
  /* Selectbox, slider */
  [data-testid="stSelectbox"], [data-testid="stSlider"] {
      color: #e2e8f0;
  }
  /* Progress bar */
  .stProgress > div > div { background: #00d4ff !important; }
  /* Expander */
  .streamlit-expanderHeader { color: #00d4ff !important; }
  /* Info / warning boxes */
  .stAlert { background: #0a1525 !important; border-color: #1e2d40 !important; }
  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #070d14; }
  ::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 3px; }
  /* Headers */
  h1, h2, h3, h4 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)



# HELPERS

def card(label: str, value: str, delta: str = "", color: str = "#e2e8f0"):
    """Render a styled metric card."""
    delta_html = f'<div style="color:{color};font-size:12px;margin-top:2px">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div style="background:#0a1525;border:1px solid #1e2d40;border-radius:8px;
                padding:14px 18px;text-align:center;">
        <div style="color:#556677;font-size:10px;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:6px">{label}</div>
        <div style="color:#e2e8f0;font-size:22px;font-weight:600">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def signal_card(title: str, badge: str, badge_color: str, body_html: str):
    """Render a signal card with colored badge."""
    st.markdown(f"""
    <div style="background:#0a1525;border:1px solid #1e2d40;border-radius:8px;
                padding:16px 18px;height:100%;">
        <div style="color:#556677;font-size:10px;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:8px">{title}</div>
        <div style="display:inline-block;background:{badge_color}22;color:{badge_color};
                    border:1px solid {badge_color};border-radius:4px;padding:3px 10px;
                    font-size:13px;font-weight:600;margin-bottom:10px">{badge}</div>
        <div style="color:#8899aa;font-size:12px">{body_html}</div>
    </div>
    """, unsafe_allow_html=True)


def prob_bar(label: str, prob: float, color: str):
    """Render a probability progress bar."""
    pct = int(prob * 100)
    st.markdown(f"""
    <div style="margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;color:#8899aa;margin-bottom:2px">
        <span>{label}</span><span style="color:{color}">{pct}%</span>
      </div>
      <div style="background:#1e2d40;border-radius:3px;height:6px">
        <div style="background:{color};width:{pct}%;height:6px;border-radius:3px"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def disclaimer_box(text: str):
    st.markdown(f"""
    <div style="background:#0a1525;border:1px solid #f59e0b44;border-radius:8px;
                padding:12px 16px;color:#f59e0b;font-size:11px;margin-top:12px">
        {text}
    </div>
    """, unsafe_allow_html=True)


def philosophy_box():
    with st.expander("◈ CORE PHILOSOPHY — Why we don't predict prices", expanded=False):
        st.markdown(f"""
        <div style="background:#070d14;border-left:3px solid #00d4ff;
                    padding:16px 20px;border-radius:0 6px 6px 0;
                    color:#8899aa;font-size:12px;line-height:1.8">
            <pre style="background:transparent;color:#8899aa;
                        font-family:'IBM Plex Mono',monospace;
                        font-size:12px;white-space:pre-wrap">{CORE_PHILOSOPHY}</pre>
        </div>
        """, unsafe_allow_html=True)



# PIPELINE RUNNER (cached by ticker)

def run_full_pipeline(ticker: str, threshold: float, progress_bar, status_text):
    """Execute full QuantEdge pipeline and return all results."""
    from data.pipeline import DataPipeline
    from models.classifier import ModelTrainer
    from models.regime_lstm import RegimeLSTM
    from models.volatility import GARCHVolatility
    from backtesting.backtest import Backtester
    from mlflow_tracking.experiments import ExperimentTracker

    results = {}

    # 1. Data
    status_text.text("⬡ Downloading market data from Yahoo Finance...")
    progress_bar.progress(5)
    pipeline = DataPipeline(ticker, period="3y")
    X_train, X_test, y_train, y_test, feat_names, dates_test, scaler, df = pipeline.run()
    results["df"]          = df
    results["X_train"]     = X_train
    results["X_test"]      = X_test
    results["y_train"]     = y_train
    results["y_test"]      = y_test
    results["feat_names"]  = feat_names
    results["dates_test"]  = dates_test
    results["scaler"]      = scaler
    progress_bar.progress(15)

    # 2. Feature engineering confirmation
    status_text.text(f"⬡ Engineered {len(feat_names)} features — momentum, vol, volume, macro...")
    progress_bar.progress(20)

    # 3. Logistic
    status_text.text("⬡ Training Logistic Regression (baseline)...")
    trainer = ModelTrainer()
    from models.classifier import ModelTrainer as MT
    trainer = MT()
    trainer.train_logistic(X_train, y_train, X_test, y_test)
    progress_bar.progress(35)

    # 4. XGBoost
    status_text.text("⬡ Training XGBoost...")
    try:
        trainer.train_xgb(X_train, y_train, X_test, y_test)
    except Exception as e:
        st.warning(f"XGBoost skipped: {e}")
    progress_bar.progress(50)

    # 5. LightGBM
    status_text.text("⬡ Training LightGBM...")
    try:
        trainer.train_lgbm(X_train, y_train, X_test, y_test)
    except Exception as e:
        st.warning(f"LightGBM skipped: {e}")
    trainer.feature_names = feat_names
    results["trainer"] = trainer
    progress_bar.progress(62)

    # 6. GARCH
    status_text.text("⬡ Fitting GARCH(1,1) volatility model...")
    garch = GARCHVolatility()
    try:
        garch.fit(df["log_return"].dropna())
    except Exception as e:
        st.warning(f"GARCH fit warning: {e}")
    results["garch"] = garch
    progress_bar.progress(72)

    # 7. LSTM
    status_text.text("⬡ Training LSTM Regime Detector (Bull/Bear/Sideways)...")
    lstm = RegimeLSTM()
    try:
        lstm.train(df)
    except Exception as e:
        # LSTM failed (TF not available etc.) — use rule-based fallback
        pass
    regime_series, regime_proba = lstm.predict_regime(df)
    # Inject LSTM regime back into feature set
    if regime_series is not None and len(regime_series) > 0:
        df["regime"] = regime_series.reindex(df.index, method="ffill").fillna(1)
    results["lstm"]         = lstm
    results["regime"]       = regime_series
    results["regime_proba"] = regime_proba
    progress_bar.progress(84)

    # 8. Backtest
    status_text.text("⬡ Running walk-forward backtest...")
    best_name, best_model, best_result = trainer.get_best_model()
    prices_test = df.loc[dates_test, "Close"].values
    bt = Backtester(confidence_threshold=threshold)
    bt.run(dates_test, prices_test, best_result["y_pred"], best_result["y_proba"])
    bt_metrics = bt.compute_metrics()
    results["backtester"]  = bt
    results["bt_metrics"]  = bt_metrics
    results["best_name"]   = best_name
    results["best_result"] = best_result
    progress_bar.progress(93)

    # 9. Permutation importance
    status_text.text("⬡ Computing permutation feature importance...")
    try:
        fi_df = trainer.permutation_importance(X_test, y_test)
    except Exception:
        fi_df = pd.DataFrame({"feature": feat_names,
                               "importance_mean": np.zeros(len(feat_names)),
                               "importance_std":  np.zeros(len(feat_names))})
    results["fi_df"] = fi_df
    progress_bar.progress(98)

    # 10. MLflow
    status_text.text("⬡ Logging experiments to MLflow...")
    tracker = ExperimentTracker("quantedge")
    for name, res in trainer.results.items():
        tracker.log_run(
            ticker=ticker,
            model_name=name,
            params={},
            metrics={**res, "sharpe_ratio": bt_metrics["sharpe_ratio"],
                     "max_drawdown": bt_metrics["max_drawdown"]},
            model=res["model"],
            feature_importance_df=fi_df if name == best_name else None,
        )
    results["tracker"] = tracker
    progress_bar.progress(100)
    status_text.text("✓ Complete")
    return results



# SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px">
      <div style="color:#00d4ff;font-size:28px;font-weight:600;
                  letter-spacing:2px">◈ QUANTEDGE</div>
      <div style="color:#556677;font-size:10px;letter-spacing:3px;
                  margin-top:4px">ML SIGNAL ENGINE</div>
    </div>
    <hr style="border-color:#1e2d40;margin:0 0 20px">
    """, unsafe_allow_html=True)

    ticker = st.selectbox("SELECT TICKER", TICKERS, index=0)
    threshold = st.slider(
        "CONFIDENCE THRESHOLD",
        min_value=0.50, max_value=0.70,
        value=CONFIDENCE_THRESHOLD, step=0.01,
        help="Only trade when model confidence exceeds this level"
    )
    train_btn = st.button("▸  TRAIN MODELS", use_container_width=True)

    # Progress area (hidden until training)
    progress_ph = st.empty()
    status_ph   = st.empty()

    # ── Post-training sidebar summary 
    if "results" in st.session_state and st.session_state.get("trained_ticker") == ticker:
        R = st.session_state["results"]
        st.markdown("<hr style='border-color:#1e2d40;margin:16px 0'>", unsafe_allow_html=True)
        st.markdown("**LIVE SIGNAL**", unsafe_allow_html=False)

        best_name   = R["best_name"]
        best_result = R["best_result"]
        roc_auc     = best_result["roc_auc"]
        dir_acc     = best_result["directional_accuracy"]

        col1, col2 = st.columns(2)
        col1.metric("Best Model",  best_name)
        col2.metric("ROC-AUC",    f"{roc_auc:.3f}")
        st.metric("Directional Acc", f"{dir_acc*100:.1f}%",
                  delta=f"{(dir_acc - 0.5)*100:+.1f}pp vs random")

        # Current regime
        regime_series = R.get("regime")
        if regime_series is not None and len(regime_series) > 0:
            latest_regime = int(regime_series.iloc[-1]) if hasattr(regime_series, "iloc") else 1
            labels = {0: ("BEARISH", DARK_THEME["red"]),
                      1: ("SIDEWAYS", DARK_THEME["yellow"]),
                      2: ("BULLISH", DARK_THEME["green"])}
            rl, rc = labels[latest_regime]
            st.markdown(
                f'<div style="margin-top:6px">REGIME: '
                f'<span style="color:{rc};font-weight:600">{rl}</span></div>',
                unsafe_allow_html=True
            )

        # Vol regime
        garch = R.get("garch")
        if garch and garch.result is not None:
            try:
                vr_label, vr_color, vr_pct = garch.get_volatility_regime()
                st.markdown(
                    f'<div style="margin-top:4px">VOL: '
                    f'<span style="color:{vr_color};font-weight:600">{vr_label}</span></div>',
                    unsafe_allow_html=True
                )
            except Exception:
                pass



# TRAINING TRIGGER

if train_btn:
    with progress_ph:
        pb = st.progress(0)
    with status_ph:
        st_txt = st.empty()
    try:
        results = run_full_pipeline(ticker, threshold, pb, st_txt)
        st.session_state["results"]        = results
        st.session_state["trained_ticker"] = ticker
        st.session_state["threshold"]      = threshold
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        raise



# MAIN CONTENT

if "results" not in st.session_state or st.session_state.get("trained_ticker") != ticker:
    # Landing page
    st.markdown("""
    <div style="text-align:center;padding:80px 20px">
      <div style="color:#00d4ff;font-size:48px;font-weight:600;
                  letter-spacing:4px;margin-bottom:12px">◈ QUANTEDGE</div>
      <div style="color:#556677;font-size:14px;max-width:600px;
                  margin:0 auto;line-height:1.8">
        ML-Powered Stock Signal Engine<br>
        Select a ticker and click <b style="color:#00d4ff">▸ TRAIN MODELS</b> to begin.
      </div>
    </div>
    """, unsafe_allow_html=True)
    philosophy_box()
    st.stop()


# ── Load results from session 
R           = st.session_state["results"]
df          = R["df"]
X_test      = R["X_test"]
y_test      = R["y_test"]
dates_test  = R["dates_test"]
feat_names  = R["feat_names"]
trainer     = R["trainer"]
best_name   = R["best_name"]
best_result = R["best_result"]
garch       = R["garch"]
lstm        = R["lstm"]
backtester  = R["backtester"]
bt_metrics  = R["bt_metrics"]
fi_df       = R["fi_df"]
tracker     = R["tracker"]
regime_series  = R.get("regime")
regime_proba   = R.get("regime_proba")

# ── Tabs 
tabs = st.tabs([
    "◈  SIGNAL",
    "◈  MODEL PERFORMANCE",
    "◈  FEATURE IMPORTANCE",
    "◈  BACKTEST",
    "◈  RISK ENGINE",
    "◈  EXPERIMENTS",
])



# TAB 1 — SIGNAL DASHBOARD

with tabs[0]:
    st.markdown(f"### {ticker} — Live Signal Dashboard")

    # ── Row 1: Price stats from yfinance 
    try:
        info = yf.Ticker(ticker).fast_info
        curr_price = getattr(info, "last_price", df["Close"].iloc[-1])
        prev_close = getattr(info, "previous_close", df["Close"].iloc[-2])
        mkt_cap    = getattr(info, "market_cap", 0)
        daily_chg  = (curr_price - prev_close) / prev_close * 100 if prev_close else 0
        chg_str    = f"{daily_chg:+.2f}%"
        cap_str    = f"${mkt_cap/1e9:.1f}B" if mkt_cap and mkt_cap > 0 else "N/A"
    except Exception:
        curr_price = df["Close"].iloc[-1]
        chg_str    = f"{df['log_return'].iloc[-1]*100:+.2f}%"
        cap_str    = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("CURRENT PRICE",  f"${curr_price:.2f}")
    with c2: card("DAILY CHANGE",   chg_str,
                  color=DARK_THEME["green"] if "+" in chg_str else DARK_THEME["red"])
    with c3: card("MARKET CAP",     cap_str)
    with c4: card("DATA SOURCE",    "Yahoo Finance")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Signal cards 
    sc1, sc2, sc3 = st.columns(3)

    # ML Signal
    last_proba  = best_result["y_proba"][-1]
    last_pred   = int(best_result["y_pred"][-1])
    confidence  = last_proba[1] * 100
    sig_label   = "UP ▲" if last_pred == 1 else "DOWN ▼"
    sig_color   = DARK_THEME["green"] if last_pred == 1 else DARK_THEME["red"]
    with sc1:
        signal_card(
            "ML SIGNAL",
            sig_label,
            sig_color,
            f"Confidence: <b style='color:{sig_color}'>{confidence:.1f}%</b><br>"
            f"<span style='font-size:10px'>Based on {best_name} + {len(feat_names)} features</span>"
        )

    # Market Regime
    if regime_series is not None and len(regime_series) > 0:
        latest_r = int(regime_series.iloc[-1]) if hasattr(regime_series, "iloc") else 1
    else:
        latest_r = 1
    r_labels = {0: "BEARISH", 1: "SIDEWAYS", 2: "BULLISH"}
    r_colors = {0: DARK_THEME["red"], 1: DARK_THEME["yellow"], 2: DARK_THEME["green"]}
    r_label  = r_labels[latest_r]
    r_color  = r_colors[latest_r]

    if regime_proba is not None and len(regime_proba) > 0:
        last_rp = regime_proba[-1]
        p_bear, p_side, p_bull = float(last_rp[0]), float(last_rp[1]), float(last_rp[2])
    else:
        p_bear, p_side, p_bull = (0.15, 0.20, 0.65) if latest_r == 2 else \
                                   (0.65, 0.20, 0.15) if latest_r == 0 else \
                                   (0.15, 0.70, 0.15)

    with sc2:
        signal_card("MARKET REGIME (LSTM)", r_label, r_color, "")
        prob_bar("BULLISH",  p_bull, DARK_THEME["green"])
        prob_bar("SIDEWAYS", p_side, DARK_THEME["yellow"])
        prob_bar("BEARISH",  p_bear, DARK_THEME["red"])

    # Vol Regime
    vol_label, vol_color, vol_pct = "MEDIUM", DARK_THEME["yellow"], 50.0
    curr_vol = 20.0
    if garch and garch.result is not None:
        try:
            vol_label, vol_color, vol_pct = garch.get_volatility_regime()
            fc = garch.forecast(horizon=1)
            curr_vol = fc["current_vol"]
        except Exception:
            pass

    with sc3:
        signal_card(
            "VOLATILITY REGIME (GARCH)",
            vol_label,
            vol_color,
            f"Ann. Vol: <b style='color:{vol_color}'>{curr_vol:.1f}%</b><br>"
            f"Hist. Percentile: {vol_pct:.0f}th"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Candlestick chart 
    last_90 = df.tail(90)
    fig_candle = go.Figure()

    # Bollinger Bands
    bb_mid   = last_90["Close"].rolling(20).mean()
    bb_std   = last_90["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    fig_candle.add_trace(go.Scatter(
        x=last_90.index, y=bb_upper,
        mode="lines", name="BB Upper",
        line=dict(color="rgba(245,158,11,0.5)", width=1, dash="dash"),
    ))
    fig_candle.add_trace(go.Scatter(
        x=last_90.index, y=bb_lower,
        mode="lines", name="BB Lower",
        fill="tonexty", fillcolor="rgba(245,158,11,0.04)",
        line=dict(color="rgba(245,158,11,0.5)", width=1, dash="dash"),
    ))
    fig_candle.add_trace(go.Scatter(
        x=last_90.index, y=bb_mid,
        mode="lines", name="BB Mid",
        line=dict(color="rgba(245,158,11,0.3)", width=1),
    ))

    # Candlestick
    fig_candle.add_trace(go.Candlestick(
        x=last_90.index,
        open=last_90["Open"], high=last_90["High"],
        low=last_90["Low"],   close=last_90["Close"],
        name="OHLC",
        increasing_line_color=DARK_THEME["green"],
        decreasing_line_color=DARK_THEME["red"],
    ))

    # Train/test boundary
    split_date = dates_test[0] if len(dates_test) > 0 else last_90.index[-1]
    if split_date in last_90.index or (split_date >= last_90.index[0]):
        fig_candle.add_vline(
            x=str(split_date),
            line_color=DARK_THEME["accent"],
            line_dash="dash",
            annotation_text="Test Start",
            annotation_font_color=DARK_THEME["accent"],
        )

    fig_candle.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text=f"{ticker} Price — Last 90 Days with Bollinger Bands | Source: yfinance",
            font=dict(color=DARK_THEME["text"])
        ),
        xaxis_rangeslider_visible=False,
        height=420,
        showlegend=True,
        legend=dict(bgcolor=DARK_THEME["surface"], bordercolor=DARK_THEME["border"],
                    font=dict(color=DARK_THEME["text"])),
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── Feature snapshot 
    st.markdown("#### Today's Feature Snapshot")
    latest_row = df[feat_names].dropna().iloc[-1]

    def feature_color(feat, val):
        bullish = {
            "rsi_14": (val < 30, val > 70),  # oversold=bullish, overbought=bearish
            "log_return": (val > 0, val < 0),
            "macd_crossover": (val == 1, False),
            "bb_position": (val < 0.2, val > 0.8),
            "regime": (val == 2, val == 0),
        }
        if feat in bullish:
            bull, bear = bullish[feat]
            if bull: return DARK_THEME["green"]
            if bear: return DARK_THEME["red"]
        elif "return" in feat:
            return DARK_THEME["green"] if val > 0 else DARK_THEME["red"]
        return DARK_THEME["muted"]

    snap_rows = []
    for f in feat_names:
        v = latest_row[f]
        snap_rows.append({"Feature": f, "Value": f"{v:.4f}", "Signal": "↑" if feature_color(f, v) == DARK_THEME["green"] else ("↓" if feature_color(f, v) == DARK_THEME["red"] else "─")})
    snap_df = pd.DataFrame(snap_rows)
    st.dataframe(snap_df, use_container_width=True, hide_index=True)



# TAB 2 — MODEL PERFORMANCE

with tabs[1]:
    st.markdown("### Model Performance — Test Set Only")
    philosophy_box()

    # Comparison table
    comp_df = trainer.get_comparison_df()
    st.markdown("#### Model Comparison")
    
    # Highlight best
    def highlight_best(row):
        if "★" in str(row.get("Status", "")):
            return [f"background-color: {DARK_THEME['accent']}22; color: {DARK_THEME['accent']}"] * len(row)
        elif row.get("Model") == "Random Baseline":
            return [f"color: {DARK_THEME['muted']}"] * len(row)
        return [""] * len(row)

    styled = comp_df.style.apply(highlight_best, axis=1).format({
        "Accuracy":        "{:.1%}",
        "ROC-AUC":         "{:.3f}",
        "F1":              "{:.3f}",
        "Directional Acc": "{:.1%}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Edge metric
    best_acc = best_result["directional_accuracy"]
    edge_pp  = (best_acc - 0.5) * 100
    st.markdown(f"""
    <div style="background:#0a1525;border:1px solid #1e2d40;border-radius:8px;
                padding:16px 20px;text-align:center;margin:12px 0">
        <div style="color:#556677;font-size:10px;letter-spacing:1px">STATISTICAL EDGE OVER RANDOM</div>
        <div style="color:#00d4ff;font-size:36px;font-weight:600">{edge_pp:+.1f} pp</div>
        <div style="color:#8899aa;font-size:11px">
            {best_name}: {best_acc*100:.1f}% directional accuracy vs 50.0% baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts row
    from evaluation.metrics import (
        plot_roc_curves, plot_confusion_matrix,
        plot_calibration_curve, rolling_directional_accuracy
    )

    col_a, col_b = st.columns(2)

    with col_a:
        roc_fig = plot_roc_curves({
            name: {"y_proba": r["y_proba"], "y_test": r["y_test"]}
            for name, r in trainer.results.items()
        })
        st.plotly_chart(roc_fig, use_container_width=True)

    with col_b:
        # Compute confusion matrix cleanly
        from sklearn.metrics import confusion_matrix
        import plotly.graph_objects as go
        
        cm = confusion_matrix(y_test, best_result["y_pred"])
        tn, fp, fn, tp = cm.ravel()

        labels = [
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"]
        ]

        # Generate base heatmap
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted DOWN", "Predicted UP"],
            y=["Actual DOWN", "Actual UP"],
            text=labels,
            texttemplate="%{text}",
            colorscale=[
                [0, DARK_THEME["surface"]],
                [1, DARK_THEME["accent"]]
            ],
            showscale=False,
            textfont=dict(color=DARK_THEME["text"], size=16, family="IBM Plex Mono"),
        ))

        # Apply basic layout properties manually to avoid Python 3.12 unpack collision
        cm_fig.update_layout(
            paper_bgcolor=PLOTLY_THEME.get("paper_bgcolor", "#0a1525"),
            plot_bgcolor=PLOTLY_THEME.get("plot_bgcolor", "#0a1525"),
            font=PLOTLY_THEME.get("font"),
            margin=PLOTLY_THEME.get("margin"),
            title=dict(
                text=f"Confusion Matrix — {best_name} (Test Set)",
                font=dict(color=DARK_THEME["text"])
            ),
            height=380,
        )

        # Safely assign axis layouts
        cm_fig.update_xaxes(
            gridcolor=PLOTLY_THEME.get("xaxis", {}).get("gridcolor", "#1e2d40"),
            showgrid=PLOTLY_THEME.get("xaxis", {}).get("showgrid", True),
            zeroline=False,
            title="Predicted Label",
            tickfont=dict(color=DARK_THEME["text"]),
        )

        cm_fig.update_yaxes(
            gridcolor=PLOTLY_THEME.get("yaxis", {}).get("gridcolor", "#1e2d40"),
            showgrid=PLOTLY_THEME.get("yaxis", {}).get("showgrid", True),
            zeroline=False,
            title="True Label",
            tickfont=dict(color=DARK_THEME["text"]),
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cal_fig = plot_calibration_curve(y_test, best_result["y_proba"], best_name)
        st.plotly_chart(cal_fig, use_container_width=True)

    with col_d:
        roll_fig = rolling_directional_accuracy(
            y_test, best_result["y_pred"], dates_test, window=30
        )
        st.plotly_chart(roll_fig, use_container_width=True)



# TAB 3 — FEATURE IMPORTANCE

with tabs[2]:
    st.markdown("### What Drives the Predictions?")
    st.markdown(
        "<div style='color:#556677;font-size:12px;margin-bottom:16px'>"
        "Permutation importance on test set — the only unbiased measure of feature value"
        "</div>",
        unsafe_allow_html=True
    )

    # Color by category
    def feat_color(name):
        vol_feats    = ["rolling_vol_10", "rolling_vol_20", "bb_width", "bb_position"]
        vol_feats2   = ["garch", "vol"]
        volume_feats = ["volume_zscore", "volume_ratio"]
        macro_feats  = ["vix_level", "vix_change", "regime"]
        if name in vol_feats or any(v in name for v in vol_feats2):
            return DARK_THEME["yellow"]
        if name in volume_feats:
            return DARK_THEME["green"]
        if name in macro_feats:
            return DARK_THEME["purple"]
        return DARK_THEME["accent"]  # momentum

    top15 = fi_df.head(15)
    colors = [feat_color(f) for f in top15["feature"]]

    fig_fi = go.Figure(go.Bar(
        x=top15["importance_mean"],
        y=top15["feature"],
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            array=top15["importance_std"],
            visible=True,
            color=DARK_THEME["muted"],
        ),
    ))
    # 1. Apply the global theme configurations safely first
    fig_fi.update_layout(**PLOTLY_THEME)

    # 2. Layer your explicit axis, title, and height overrides on top
    fig_fi.update_layout(
        title=dict(
            text=f"Top 15 Feature Importance (Permutation, {best_name}) | Source: yfinance",
            font=dict(color=DARK_THEME["text"])
        ),
        xaxis_title="Importance (ROC-AUC decrease on permutation)",
        yaxis=dict(
            **PLOTLY_THEME.get("yaxis", {}), # Safely unpack theme's yaxis configurations
            autorange="reversed",            # Keeps the most important features at the top
            tickfont=dict(color=DARK_THEME["text"])
        ),
        height=500,
        showlegend=False,
    )

    # Legend
    legend_html = "".join([
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'background:{c};border-radius:2px;margin-right:4px"></span>'
        f'<span style="color:#8899aa;font-size:11px;margin-right:16px">{l}</span>'
        for c, l in [
            (DARK_THEME["accent"],  "Momentum"),
            (DARK_THEME["yellow"],  "Volatility"),
            (DARK_THEME["green"],   "Volume"),
            (DARK_THEME["purple"],  "Macro"),
        ]
    ])
    st.markdown(legend_html, unsafe_allow_html=True)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Auto-insight
    if len(fi_df) > 0:
        top_feat  = fi_df.iloc[0]["feature"]
        top_score = fi_df.iloc[0]["importance_mean"]
        interp_map = {
            "rsi": "overbought/oversold conditions have short-term mean reversion signal",
            "volume_zscore": "unusual volume precedes directional moves",
            "vix_level": "macro fear is the dominant driver — this is a macro-regime driven stock",
            "bb_position": "price position within Bollinger Bands has strong predictive value",
            "log_return": "recent momentum (1-day return) is highly predictive",
            "macd": "trend-following (MACD) signals are influential",
        }
        interp = next(
            (v for k, v in interp_map.items() if k in top_feat),
            "this feature captures the most signal in current market conditions"
        )
        st.markdown(
            f"<div style='background:#0a1525;border-left:3px solid #00d4ff;"
            f"padding:12px 16px;border-radius:0 6px 6px 0;font-size:12px;"
            f"color:#8899aa;margin-top:8px'>"
            f"<b style='color:#e2e8f0'>Top predictor:</b> {top_feat} "
            f"(importance: {top_score:.4f})<br>"
            f"This suggests {interp}."
            f"</div>",
            unsafe_allow_html=True
        )



# TAB 4 — BACKTEST

with tabs[3]:
    st.markdown("### Walk-Forward Backtest")
    st.markdown(
        "<div style='color:#556677;font-size:11px;margin-bottom:16px'>"
        "Model trained on 80% of data. Backtest runs on held-out 20% test set only."
        "</div>",
        unsafe_allow_html=True
    )

    # Metrics
    m = bt_metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: card("STRATEGY RETURN", f"{m['total_return_strategy']*100:.1f}%",
                  delta=f"BnH: {m['total_return_bnh']*100:.1f}%",
                  color=DARK_THEME["green"] if m["total_return_strategy"] > m["total_return_bnh"] else DARK_THEME["red"])
    with c2: card("SHARPE RATIO",    f"{m['sharpe_ratio']:.2f}")
    with c3: card("MAX DRAWDOWN",    f"{m['max_drawdown']*100:.1f}%",
                  color=DARK_THEME["red"])
    with c4: card("WIN RATE",        f"{m['win_rate']*100:.1f}%")
    with c5: card("# TRADES",        str(m["n_trades"]))

    # Equity curve
    st.plotly_chart(backtester.plot_equity_curve(), use_container_width=True)

    # Drawdown
    st.plotly_chart(backtester.plot_drawdown(), use_container_width=True)

    # Monthly heatmap
    st.plotly_chart(backtester.plot_monthly_returns(), use_container_width=True)

    # Trade log
    st.markdown("#### Last 20 Trades")
    trade_log = backtester.get_trade_log(last_n=20)
    st.dataframe(trade_log, use_container_width=True)

    disclaimer_box(
        "⚠️ This backtest is for educational purposes only. "
        "Past performance does not guarantee future results. "
        "No look-ahead bias: model trained only on data prior to the test period."
    )



# TAB 5 — RISK ENGINE

with tabs[4]:
    st.markdown("### Risk Engine — GARCH(1,1) VaR & CVaR")

    position_size = st.number_input(
        "Position Size ($)",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
    )

    if garch and garch.result is not None:
        try:
            var = garch.compute_var(position_size)

            # VaR metrics
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                card("VaR 95% (Daily $)", f"-${var['var_95']:,.0f}",
                     delta=f"{var['var_95_pct']:.2f}% of position",
                     color=DARK_THEME["yellow"])
            with rc2:
                card("VaR 99% (Daily $)", f"-${var['var_99']:,.0f}",
                     delta=f"{var['var_99_pct']:.2f}% of position",
                     color=DARK_THEME["red"])
            with rc3:
                card("CVaR 95% (Exp. Shortfall)", f"-${var['cvar_95']:,.0f}",
                     delta=f"{var['cvar_95_pct']:.2f}% of position",
                     color=DARK_THEME["red"])

            st.markdown(
                f"<div style='color:#8899aa;font-size:12px;margin:12px 0'>"
                f"On a typical bad day (1-in-20 chance), you could lose "
                f"<b style='color:{DARK_THEME['yellow']}'>${var['var_95']:,.0f}</b> "
                f"on a ${position_size:,.0f} position. "
                f"In extreme scenarios (1-in-100), loss could reach "
                f"<b style='color:{DARK_THEME['red']}'>${var['var_99']:,.0f}</b>."
                f"</div>",
                unsafe_allow_html=True
            )

            # Forecast chart
            st.plotly_chart(garch.plot_vol_forecast(), use_container_width=True)

            # Return distribution
            st.plotly_chart(garch.plot_return_distribution(position_size), use_container_width=True)

            # Diagnostics
            with st.expander("GARCH Diagnostics", expanded=False):
                diag = garch.diagnostic_plots()
                st.plotly_chart(diag["fig_residuals"], use_container_width=True)
                st.plotly_chart(diag["fig_qq"],        use_container_width=True)

                lb_p = diag["lb_pvalue"]
                if lb_p > 0.05:
                    st.success(f"✅ Ljung-Box p = {lb_p:.4f} — GARCH fit is good. Autocorrelation captured.")
                else:
                    st.warning(f"⚠️ Ljung-Box p = {lb_p:.4f} — Residual autocorrelation remains. Consider EGARCH.")

                st.write(f"AIC: {diag['aic']:.2f}   |   BIC: {diag['bic']:.2f}")

        except Exception as e:
            st.warning(f"Risk engine error: {e}")
    else:
        st.info("GARCH model not available. Ensure arch library is installed.")



# TAB 6 — EXPERIMENT TRACKER

with tabs[5]:
    st.markdown("### MLflow Experiment Log")
    st.markdown(
        "<div style='color:#556677;font-size:12px;margin-bottom:16px'>"
        "All model runs tracked automatically. Every experiment is reproducible."
        "</div>",
        unsafe_allow_html=True
    )

    with st.expander("Why MLflow?", expanded=False):
        st.markdown("""
        <div style='color:#8899aa;font-size:12px;line-height:1.8'>
        In production data science teams, every experiment is tracked.<br>
        MLflow enables <b style='color:#e2e8f0'>reproducibility</b>,
        <b style='color:#e2e8f0'>model versioning</b>, and
        <b style='color:#e2e8f0'>team collaboration</b> —
        standard practice at MNCs like JPMorgan, Deloitte, and Accenture.<br><br>
        Every QuantEdge run logs: hyperparameters, all 6 metrics,
        the fitted model artifact, and feature importance CSV.
        </div>
        """, unsafe_allow_html=True)

    runs_df = tracker.get_all_runs()

    if runs_df.empty:
        st.info("No runs logged yet. Train a model to populate the experiment log.")
    else:
        # Filter controls
        f1, f2 = st.columns(2)
        if "Ticker" in runs_df.columns:
            tickers_avail = ["All"] + sorted(runs_df["Ticker"].dropna().unique().tolist())
            sel_ticker = f1.selectbox("Filter by Ticker", tickers_avail)
            if sel_ticker != "All":
                runs_df = runs_df[runs_df["Ticker"] == sel_ticker]

        if "Model" in runs_df.columns:
            models_avail = ["All"] + sorted(runs_df["Model"].dropna().unique().tolist())
            sel_model = f2.selectbox("Filter by Model", models_avail)
            if sel_model != "All":
                runs_df = runs_df[runs_df["Model"] == sel_model]

        # Highlight best run
        def hl_best(row):
            if "ROC-AUC" in row and runs_df["ROC-AUC"].max() == row["ROC-AUC"]:
                return [f"background-color:{DARK_THEME['green']}22;color:{DARK_THEME['green']}"] * len(row)
            return [""] * len(row)

        styled_runs = runs_df.style.apply(hl_best, axis=1)
        st.dataframe(styled_runs, use_container_width=True, hide_index=True)

        best_run = tracker.get_best_run()
        if best_run:
            st.markdown(
                f"<div style='color:#8899aa;font-size:11px;margin-top:8px'>"
                f"★ Best run: <b style='color:{DARK_THEME['green']}'>{best_run.get('run_name','')}</b> "
                f"— ROC-AUC {best_run.get('roc_auc',0):.3f}"
                f"</div>",
                unsafe_allow_html=True
            )
