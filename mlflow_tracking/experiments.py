"""
QuantEdge MLflow Experiment Tracker
Logs every model run with params, metrics, and artifacts.
Production-mindset: every experiment is reproducible and versioned.
"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "..")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """
    Wraps MLflow to log all QuantEdge model runs.
    
    Usage:
        tracker = ExperimentTracker("quantedge")
        tracker.log_run(ticker, model_name, params, metrics, model, fi_df)
        df = tracker.get_all_runs()
    """

    def __init__(self, experiment_name: str = "quantedge"):
        self.experiment_name = experiment_name
        self._runs_cache: list = []

        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
        else:
            print("[MLflow] Not installed — runs will be logged in-memory only.")

    
    # LOG A RUN
    
    def log_run(
        self,
        ticker:                str,
        model_name:            str,
        params:                dict,
        metrics:               dict,
        model,
        feature_importance_df: pd.DataFrame = None,
    ) -> str:
        """
        Log a single model run to MLflow.
        
        Returns:
            run_id string (or "local" if MLflow unavailable)
        """
        run_record = {
            "run_name":            f"{ticker}_{model_name}",
            "ticker":              ticker,
            "model":               model_name,
            "accuracy":            round(metrics.get("accuracy", 0), 4),
            "roc_auc":             round(metrics.get("roc_auc", 0), 4),
            "f1_score":            round(metrics.get("f1", 0), 4),
            "directional_accuracy":round(metrics.get("directional_accuracy", 0), 4),
            "sharpe_ratio":        round(metrics.get("sharpe_ratio", 0), 4),
            "max_drawdown":        round(metrics.get("max_drawdown", 0), 4),
            "timestamp":           pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "params":              str(params),
        }
        self._runs_cache.append(run_record)

        if not MLFLOW_AVAILABLE:
            return "local"

        run_name = f"{ticker}_{model_name}"
        with mlflow.start_run(run_name=run_name):
            # ── Parameters 
            mlflow.log_param("ticker",     ticker)
            mlflow.log_param("model",      model_name)
            for k, v in params.items():
                try:
                    mlflow.log_param(str(k), str(v)[:250])
                except Exception:
                    pass

            # ── Metrics 
            mlflow.log_metric("accuracy",            metrics.get("accuracy", 0))
            mlflow.log_metric("roc_auc",             metrics.get("roc_auc", 0))
            mlflow.log_metric("f1_score",            metrics.get("f1", 0))
            mlflow.log_metric("directional_accuracy",metrics.get("directional_accuracy", 0))
            mlflow.log_metric("sharpe_ratio",        metrics.get("sharpe_ratio", 0))
            mlflow.log_metric("max_drawdown",        metrics.get("max_drawdown", 0))

            # ── Model 
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                pass

            # ── Feature importance CSV 
            if feature_importance_df is not None:
                try:
                    tmp = "/tmp/feature_importance.csv"
                    feature_importance_df.to_csv(tmp, index=False)
                    mlflow.log_artifact(tmp, "feature_importance")
                except Exception:
                    pass

            run_id = mlflow.active_run().info.run_id

        return run_id

    
    # GET ALL RUNS
    
    def get_all_runs(self) -> pd.DataFrame:
        """
        Return a DataFrame of all logged runs.
        Tries MLflow first; falls back to in-memory cache.
        """
        if MLFLOW_AVAILABLE:
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is not None:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["metrics.roc_auc DESC"],
                    )
                    if not runs.empty:
                        # Rename columns for display
                        col_map = {
                            "tags.mlflow.runName":         "Run",
                            "params.ticker":               "Ticker",
                            "params.model":                "Model",
                            "metrics.accuracy":            "Accuracy",
                            "metrics.roc_auc":             "ROC-AUC",
                            "metrics.directional_accuracy":"Directional Acc",
                            "metrics.sharpe_ratio":        "Sharpe",
                            "metrics.max_drawdown":        "Max Drawdown",
                            "start_time":                  "Timestamp",
                        }
                        available = {k: v for k, v in col_map.items() if k in runs.columns}
                        df = runs[list(available.keys())].rename(columns=available)
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
                        return df
            except Exception as e:
                pass

        # Fallback to in-memory cache
        if self._runs_cache:
            df = pd.DataFrame(self._runs_cache)
            display_cols = [
                "run_name", "ticker", "model", "accuracy", "roc_auc",
                "directional_accuracy", "sharpe_ratio", "max_drawdown", "timestamp"
            ]
            rename = {
                "run_name": "Run", "ticker": "Ticker", "model": "Model",
                "accuracy": "Accuracy", "roc_auc": "ROC-AUC",
                "directional_accuracy": "Directional Acc",
                "sharpe_ratio": "Sharpe", "max_drawdown": "Max Drawdown",
                "timestamp": "Timestamp",
            }
            return df[[c for c in display_cols if c in df.columns]].rename(columns=rename)

        return pd.DataFrame(columns=[
            "Run", "Ticker", "Model", "Accuracy", "ROC-AUC",
            "Directional Acc", "Sharpe", "Max Drawdown", "Timestamp"
        ])

    
    # BEST RUN
    
    def get_best_run(self, metric: str = "roc_auc") -> dict:
        """
        Return the run dict with the highest value of `metric`.
        """
        if not self._runs_cache:
            return {}
        best = max(self._runs_cache, key=lambda r: r.get(metric, 0))
        return best
