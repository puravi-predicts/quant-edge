"""
QuantEdge Walk-Forward Backtester
Strategy: go LONG when classifier confidence >= threshold, else CASH.
No lookahead bias: model trained only on data prior to test period.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "..")
from config import CONFIDENCE_THRESHOLD, PLOTLY_THEME, DARK_THEME

DISCLAIMER = (
    "⚠️ This backtest is for educational purposes only. "
    "Past performance does not guarantee future results. "
    "No look-ahead bias: the model was trained only on data prior to the test period."
)


class Backtester:
    """
    Walk-forward backtest of the ML signal strategy.
    
    Strategy logic:
        BUY  (signal=1) when y_pred==1 AND y_proba[:,1] >= threshold
        CASH (signal=0) otherwise
    
    Transaction cost applied on every position change.
    
    Usage:
        bt = Backtester(confidence_threshold=0.60, transaction_cost=0.001)
        results_df = bt.run(dates, prices, y_pred, y_proba)
        metrics = bt.compute_metrics()
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        transaction_cost: float = 0.001,
    ):
        self.threshold        = confidence_threshold
        self.transaction_cost = transaction_cost
        self.results_df       = None
        self._metrics         = None

    
    # RUN

    def run(
        self,
        dates:   pd.DatetimeIndex,
        prices:  np.ndarray,
        y_pred:  np.ndarray,
        y_proba: np.ndarray,
    ) -> pd.DataFrame:
        """
        Execute the backtest.
        
        Args:
            dates:   DatetimeIndex for the test period
            prices:  Close prices aligned with dates
            y_pred:  Binary predictions (0/1)
            y_proba: Probability array shape (N, 2)
        
        Returns:
            DataFrame with all daily P&L columns
        """
        prices = np.array(prices)
        n = len(dates)

        # ── Signal generation 
        confidence  = y_proba[:, 1]
        signal      = np.where((y_pred == 1) & (confidence >= self.threshold), 1, 0)

        # ── Daily returns 
        daily_return = np.zeros(n)
        daily_return[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        # ── Strategy P&L 
        strategy_return = np.zeros(n)
        position = 0

        for i in range(1, n):
            new_position = signal[i - 1]   # lag 1: act on previous signal
            cost = 0.0
            if new_position != position:
                cost = self.transaction_cost
            strategy_return[i] = new_position * daily_return[i] - cost
            position = new_position

        # ── Cumulative returns 
        strat_cum = np.cumprod(1 + strategy_return)
        bnh_cum   = np.cumprod(1 + daily_return)
        # Start from 1.0
        strat_cum = strat_cum / strat_cum[0]
        bnh_cum   = bnh_cum   / bnh_cum[0]

        # ── Position changes (trades) 
        signal_lag   = np.concatenate([[0], signal[:-1]])
        position_chg = np.abs(np.diff(np.concatenate([[0], signal_lag])))

        self.results_df = pd.DataFrame({
            "date":              dates,
            "price":             prices,
            "daily_return":      daily_return,
            "signal":            signal_lag,            # lagged to avoid lookahead
            "confidence":        confidence,
            "strategy_return":   strategy_return,
            "strategy_cum":      strat_cum,
            "bnh_cum":           bnh_cum,
            "position_change":   position_chg,
        }).set_index("date")

        self._metrics = None  # reset cache
        return self.results_df


    # METRICS
    
    def compute_metrics(self) -> dict:
        """Compute all backtest metrics. Cached after first call."""
        if self._metrics is not None:
            return self._metrics

        df = self.results_df
        sr = df["strategy_return"]
        sc = df["strategy_cum"]
        bc = df["bnh_cum"]

        # Returns
        total_return_strategy = float(sc.iloc[-1] - 1)
        total_return_bnh      = float(bc.iloc[-1] - 1)

        # Sharpe (annualized, assuming 252 trading days)
        sharpe = float(
            (sr.mean() * 252) / (sr.std() * np.sqrt(252))
            if sr.std() > 0 else 0.0
        )

        # Max drawdown
        rolling_max = sc.cummax()
        drawdown    = sc / rolling_max - 1
        max_dd      = float(drawdown.min())

        # Trade statistics
        n_trades = int(df["position_change"].sum())
        trade_returns = sr[df["position_change"] > 0]
        win_rate = float(
            (trade_returns > 0).mean() if len(trade_returns) > 0 else 0.0
        )

        # Average holding period (days between consecutive trade signals)
        changes = df[df["position_change"] > 0].index
        if len(changes) > 1:
            holding_periods = pd.Series(changes).diff().dt.days.dropna()
            avg_hold = float(holding_periods.mean())
        else:
            avg_hold = 0.0

        self._metrics = {
            "total_return_strategy": total_return_strategy,
            "total_return_bnh":      total_return_bnh,
            "sharpe_ratio":          sharpe,
            "max_drawdown":          max_dd,
            "win_rate":              win_rate,
            "n_trades":              n_trades,
            "avg_holding_days":      avg_hold,
            "excess_return":         total_return_strategy - total_return_bnh,
        }
        return self._metrics

    
    # EQUITY CURVE
    
    def plot_equity_curve(self) -> go.Figure:
        """
        Strategy vs Buy-and-Hold cumulative returns with shaded difference.
        """
        df = self.results_df

        fig = go.Figure()

        # Shaded area between curves
        fig.add_trace(go.Scatter(
            x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
            y=pd.concat([df["strategy_cum"], df["bnh_cum"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,212,255,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ))

        # Buy-and-Hold
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bnh_cum"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=DARK_THEME["muted"], width=1.5),
        ))

        # Strategy
        fig.add_trace(go.Scatter(
            x=df.index, y=df["strategy_cum"],
            mode="lines",
            name="ML Strategy",
            line=dict(color=DARK_THEME["accent"], width=2),
        ))

        m = self.compute_metrics()
        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(
                text=(
                    f"Equity Curve | Strategy: {m['total_return_strategy']*100:.1f}% "
                    f"vs B&H: {m['total_return_bnh']*100:.1f}% | Source: yfinance"
                ),
                font=dict(color=DARK_THEME["text"])
            ),
            xaxis_title="Date",
            yaxis_title="Portfolio Value (normalized to 1.0)",
            legend=dict(bgcolor=DARK_THEME["surface"],
                        bordercolor=DARK_THEME["border"],
                        font=dict(color=DARK_THEME["text"])),
            height=420,
        )
        return fig

    
    # DRAWDOWN CHART
    
    def plot_drawdown(self) -> go.Figure:
        """
        Strategy drawdown over the test period.
        """
        df = self.results_df
        rolling_max = df["strategy_cum"].cummax()
        drawdown    = (df["strategy_cum"] / rolling_max - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=drawdown.values,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)",
            line=dict(color=DARK_THEME["red"], width=1.5),
        ))

        m = self.compute_metrics()
        fig.add_hline(
            y=m["max_drawdown"] * 100,
            line_color=DARK_THEME["red"],
            line_dash="dash",
            annotation_text=f"Max DD: {m['max_drawdown']*100:.1f}%",
            annotation_font_color=DARK_THEME["red"],
        )

        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Strategy Drawdown | Source: yfinance",
                       font=dict(color=DARK_THEME["text"])),
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=280,
        )
        return fig

    
    # MONTHLY RETURNS HEATMAP
    
    def plot_monthly_returns(self) -> go.Figure:
        """
        Year × Month heatmap of strategy monthly returns.
        Visually impressive for portfolio / interview context.
        """
        df = self.results_df.copy()
        df["month"] = df.index.month
        df["year"]  = df.index.year

        monthly = (
            df.groupby(["year", "month"])["strategy_return"]
            .apply(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )
        pivot = monthly.pivot(index="year", columns="month", values="strategy_return") * 100

        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        z     = pivot.values
        years = [str(y) for y in pivot.index]
        text  = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=years,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            showscale=True,
            textfont=dict(size=10, family="IBM Plex Mono",
                          color=DARK_THEME["text"]),
            colorbar=dict(
                tickfont=dict(color=DARK_THEME["text"]),
                title=dict(text="%", font=dict(color=DARK_THEME["text"])),
            ),
        ))

        fig.update_layout(**PLOTLY_THEME)
        fig.update_layout(
            title=dict(
                text=f"Monthly Returns Heatmap — ML Strategy | Source: yfinance",
                font=dict(color=DARK_THEME["text"])
            ),
            xaxis_title="Month",
            yaxis_title="Year",
            xaxis=dict(
                **PLOTLY_THEME.get("xaxis", {}),
                tickfont=dict(color=DARK_THEME["text"])
            ),
            yaxis=dict(
                **PLOTLY_THEME.get("yaxis", {}),
                tickfont=dict(color=DARK_THEME["text"])
            ),
            height=max(300, 60 * len(years) + 100),
        )
        return fig

    
    # TRADE LOG
    
    def get_trade_log(self, last_n: int = 20) -> pd.DataFrame:
        """
        Return the last N trade entries (rows where position changed).
        """
        df = self.results_df.copy()
        trades = df[df["position_change"] > 0].copy()
        trades = trades[["signal", "confidence", "strategy_return"]].copy()
        trades.columns = ["Signal", "Confidence", "Return"]
        trades["Signal"]     = trades["Signal"].map({1: "BUY ▲", 0: "CASH ─"})
        trades["Confidence"] = (trades["Confidence"] * 100).round(1).astype(str) + "%"
        trades["Return"]     = (trades["Return"] * 100).round(2).astype(str) + "%"
        return trades.tail(last_n)
