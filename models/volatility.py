"""
QuantEdge GARCH Volatility Model
Fits GARCH(1,1) on log returns * 100.
Forecasts volatility and computes VaR / CVaR.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "..")
from config import PLOTLY_THEME, DARK_THEME


class GARCHVolatility:
    """
    GARCH(1,1) volatility model with AR(1) mean.
    
    Usage:
        garch = GARCHVolatility()
        garch.fit(log_returns)
        forecast = garch.forecast(horizon=14)
        risk = garch.compute_var(position_value=100_000)
    """

    def __init__(self):
        self.model    = None
        self.result   = None
        self.returns  = None
        self.returns_scaled = None

    
    # FIT
    
    def fit(self, returns: pd.Series):
        """
        Fit GARCH(1,1) with AR(1) mean to log returns.
        Internally scales returns by 100 for numerical stability.
        
        Args:
            returns: pd.Series of log returns (NOT prices)
        """
        self.returns = returns.dropna()
        self.returns_scaled = self.returns * 100

        self.model = arch_model(
            self.returns_scaled,
            vol="Garch",
            p=1, q=1,
            mean="AR", lags=1,
            dist="normal"
        )
        self.result = self.model.fit(disp="off", show_warning=False)
        return self

    
    # FORECAST
    
    def forecast(self, horizon: int = 14) -> dict:
        """
        Forecast conditional variance for the next `horizon` days.
        
        Returns:
            dict with daily_sigma_pct, annualized_vol_pct, current_vol, vol_percentile
        """
        fc = self.result.forecast(horizon=horizon, reindex=False)
        variance = fc.variance.values[-1]          # shape (horizon,)
        daily_sigma = np.sqrt(variance) / 100       # back to original scale
        annualized  = daily_sigma * np.sqrt(252) * 100

        # Historical conditional vol series for percentile
        cond_vol_hist = self.result.conditional_volatility / 100 * np.sqrt(252) * 100
        current_vol   = annualized[0]
        vol_pct       = float(np.mean(cond_vol_hist <= current_vol) * 100)

        return {
            "daily_sigma_pct":    daily_sigma * 100,      # %
            "annualized_vol_pct": annualized,              # %
            "current_vol":        current_vol,             # % annualized
            "vol_percentile":     vol_pct,
            "horizon":            horizon,
            "cond_vol_history":   cond_vol_hist,
        }

    
    # VALUE AT RISK
    
    def compute_var(self, position_value: float) -> dict:
        """
        Parametric VaR and CVaR based on GARCH daily sigma.
        
        Args:
            position_value: USD notional of the position
        
        Returns:
            dict with var_95, var_99, cvar_95 in $ and %
        """
        fc = self.forecast(horizon=1)
        daily_sigma = fc["daily_sigma_pct"][0] / 100   # decimal

        var_95 = abs(norm.ppf(0.05) * daily_sigma * position_value)
        var_99 = abs(norm.ppf(0.01) * daily_sigma * position_value)

        # CVaR (Expected Shortfall at 95%)
        cvar_95 = (norm.pdf(norm.ppf(0.05)) / 0.05) * daily_sigma * position_value

        return {
            "var_95":     var_95,
            "var_99":     var_99,
            "cvar_95":    cvar_95,
            "var_95_pct": var_95 / position_value * 100,
            "var_99_pct": var_99 / position_value * 100,
            "cvar_95_pct": cvar_95 / position_value * 100,
            "daily_sigma_pct": daily_sigma * 100,
            "position_value": position_value,
        }

    
    # VOLATILITY REGIME
    
    def get_volatility_regime(self) -> tuple:
        """
        Classify current vol into LOW / MEDIUM / HIGH regime.
        
        Returns:
            (regime_label: str, color: str, percentile: float)
        """
        fc = self.forecast(horizon=1)
        pct = fc["vol_percentile"]

        if pct < 25:
            return ("LOW — Calm Market",      DARK_THEME["green"],  pct)
        elif pct < 75:
            return ("MEDIUM — Normal Conditions", DARK_THEME["yellow"], pct)
        else:
            return ("HIGH — Elevated Risk",   DARK_THEME["red"],    pct)

    
    # DIAGNOSTIC PLOTS
    
    def diagnostic_plots(self) -> dict:
        """
        Returns a dict of Plotly figures for GARCH diagnostics:
          - residuals: standardized residuals over time
          - qq: QQ plot
          - vol_forecast: 14-day forward variance forecast
          - return_dist: return distribution with VaR lines
        """
        std_resid = self.result.std_resid
        dates     = self.returns.index[-len(std_resid):]

        # ── Standardized residuals 
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=dates, y=std_resid,
            mode="lines",
            name="Std Residuals",
            line=dict(color=DARK_THEME["accent"], width=1),
        ))
        fig_resid.add_hline(y=0,  line_color=DARK_THEME["muted"], line_dash="dash")
        fig_resid.add_hline(y=2,  line_color=DARK_THEME["yellow"], line_dash="dot", annotation_text="+2σ")
        fig_resid.add_hline(y=-2, line_color=DARK_THEME["yellow"], line_dash="dot", annotation_text="-2σ")
        fig_resid.update_layout(
            **PLOTLY_THEME,
            title=dict(text="GARCH Standardized Residuals", font=dict(color=DARK_THEME["text"])),
            xaxis_title="Date", yaxis_title="Std. Residual",
            height=300,
        )

        # ── QQ Plot 
        sorted_resid = np.sort(std_resid)
        theoretical  = norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical, y=sorted_resid,
            mode="markers",
            name="Residuals",
            marker=dict(color=DARK_THEME["accent"], size=3),
        ))
        fig_qq.add_trace(go.Scatter(
            x=[-4, 4], y=[-4, 4],
            mode="lines",
            name="Normal",
            line=dict(color=DARK_THEME["muted"], dash="dash"),
        ))
        fig_qq.update_layout(
            **PLOTLY_THEME,
            title=dict(text="QQ Plot — GARCH Residuals", font=dict(color=DARK_THEME["text"])),
            xaxis_title="Theoretical Quantile",
            yaxis_title="Sample Quantile",
            height=300,
        )

        # ── Ljung-Box test 
        lb_result = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].values[0])

        # ── AIC / BIC 
        aic = self.result.aic
        bic = self.result.bic

        return {
            "fig_residuals": fig_resid,
            "fig_qq":        fig_qq,
            "lb_pvalue":     lb_pvalue,
            "aic":           aic,
            "bic":           bic,
        }

    
    # VOLATILITY FORECAST CHART
    
    def plot_vol_forecast(self) -> go.Figure:
        """
        14-day forward annualized volatility forecast with shaded uncertainty band.
        """
        fc = self.forecast(horizon=14)
        days = np.arange(1, 15)
        vol  = fc["annualized_vol_pct"]
        upper = vol * 1.2
        lower = vol * 0.8

        fig = go.Figure()

        # Shaded uncertainty band
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0,212,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±20% Band",
            showlegend=True,
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=days, y=vol,
            mode="lines+markers",
            name="Forecasted Ann. Vol (%)",
            line=dict(color=DARK_THEME["accent"], width=2),
            marker=dict(size=5),
        ))

        # Current vol reference
        fig.add_hline(
            y=fc["current_vol"],
            line_color=DARK_THEME["yellow"],
            line_dash="dash",
            annotation_text=f"Current: {fc['current_vol']:.1f}%",
            annotation_font_color=DARK_THEME["yellow"],
        )

        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(
                text="14-Day Volatility Forecast (GARCH 1,1) | Source: yfinance",
                font=dict(color=DARK_THEME["text"])
            ),
            xaxis_title="Days Ahead",
            yaxis_title="Annualized Volatility (%)",
            height=380,
        )
        return fig

    
    # RETURN DISTRIBUTION WITH VaR LINES
    
    def plot_return_distribution(self, position_value: float = 100_000) -> go.Figure:
        """
        Histogram of daily returns with VaR 95%, VaR 99%, and CVaR shaded region.
        """
        var = self.compute_var(position_value)
        ret_pct = self.returns.values * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=ret_pct,
            nbinsx=80,
            name="Daily Returns",
            marker_color=DARK_THEME["accent"],
            opacity=0.6,
        ))

        # VaR 95%
        fig.add_vline(
            x=-var["var_95_pct"],
            line_color=DARK_THEME["yellow"],
            line_dash="dash",
            annotation_text=f"VaR 95% = {var['var_95_pct']:.2f}%",
            annotation_font_color=DARK_THEME["yellow"],
        )

        # VaR 99%
        fig.add_vline(
            x=-var["var_99_pct"],
            line_color=DARK_THEME["red"],
            line_dash="dash",
            annotation_text=f"VaR 99% = {var['var_99_pct']:.2f}%",
            annotation_font_color=DARK_THEME["red"],
        )

        fig.update_layout(
            **PLOTLY_THEME,
            title=dict(
                text="Daily Return Distribution with VaR Lines | Source: yfinance",
                font=dict(color=DARK_THEME["text"])
            ),
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=380,
        )
        return fig
