"""
QuantEdge Model Trainer
Trains LightGBM, XGBoost, and Logistic Regression classifiers.
All metrics computed on test set only. No data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance as sk_perm_importance
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "..")
from config import LIGHTGBM_PARAMS, XGBOOST_PARAMS, RANDOM_STATE
from evaluation.metrics import directional_accuracy

try:
    from lightgbm import LGBMClassifier, early_stopping as lgbm_early_stopping
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class ModelTrainer:
    """
    Trains and evaluates all three classifiers on the same train/test split.
    
    Usage:
        trainer = ModelTrainer()
        results = trainer.train_all(X_train, X_test, y_train, y_test, feature_names)
        best_name, best_model, best_result = trainer.get_best_model()
    """

    def __init__(self):
        self.results       = {}
        self.feature_names = []

    
    # HELPERS
    
    def _compute_metrics(
        self,
        model_name: str,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Compute all metrics on the test set."""
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        return {
            "model_name":            model_name,
            "model":                 model,
            "accuracy":              float(accuracy_score(y_test, y_pred)),
            "precision":             float(prec),
            "recall":                float(rec),
            "f1":                    float(f1),
            "roc_auc":               float(roc_auc_score(y_test, y_proba[:, 1])),
            "directional_accuracy":  float(directional_accuracy(y_test, y_pred)),
            "y_pred":                y_pred,
            "y_proba":               y_proba,
            "y_test":                y_test,
            "confusion_matrix":      confusion_matrix(y_test, y_pred),
        }

    def _val_split(self, X_train: np.ndarray, y_train: np.ndarray):
        """Chronological 85/15 split of training data for early stopping."""
        split = int(len(X_train) * 0.85)
        return (
            X_train[:split], X_train[split:],
            y_train[:split], y_train[split:],
        )

    
    # LIGHTGBM
    
    def train_lgbm(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> dict:
        if not LGBM_AVAILABLE:
            raise RuntimeError("lightgbm not installed.")

        X_tr, X_val, y_tr, y_val = self._val_split(X_train, y_train)

        model = LGBMClassifier(**LIGHTGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm_early_stopping(50, verbose=False)],
        )
        result = self._compute_metrics("LightGBM", model, X_test, y_test)
        self.results["LightGBM"] = result
        return result

    
    # XGBOOST
    
    def train_xgb(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> dict:
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed.")

        X_tr, X_val, y_tr, y_val = self._val_split(X_train, y_train)

        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        result = self._compute_metrics("XGBoost", model, X_test, y_test)
        self.results["XGBoost"] = result
        return result

    
    # LOGISTIC REGRESSION (baseline)
    
    def train_logistic(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
    ) -> dict:
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        result = self._compute_metrics("Logistic", model, X_test, y_test)
        self.results["Logistic"] = result
        return result

    
    # TRAIN ALL
    
    def train_all(
        self,
        X_train:       np.ndarray,
        X_test:        np.ndarray,
        y_train:       np.ndarray,
        y_test:        np.ndarray,
        feature_names: list,
    ) -> dict:
        """
        Train all three models and return results dict.
        
        Returns:
            {model_name: result_dict}
        """
        self.feature_names = feature_names

        self.train_logistic(X_train, y_train, X_test, y_test)

        if XGB_AVAILABLE:
            self.train_xgb(X_train, y_train, X_test, y_test)

        if LGBM_AVAILABLE:
            self.train_lgbm(X_train, y_train, X_test, y_test)

        return self.results

    
    # BEST MODEL
    
    def get_best_model(self, metric: str = "roc_auc") -> tuple:
        """
        Return (name, model, result_dict) for the highest-scoring model.
        """
        if not self.results:
            raise RuntimeError("No models trained yet.")

        best_name = max(self.results, key=lambda k: self.results[k][metric])
        best_result = self.results[best_name]
        return best_name, best_result["model"], best_result

    
    # PERMUTATION FEATURE IMPORTANCE
    
    def permutation_importance(
        self,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        model=None,
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """
        Compute permutation importance on test set.
        Works uniformly for all sklearn-compatible models.
        
        Returns:
            DataFrame with columns: feature, importance_mean, importance_std
            Sorted descending by importance_mean.
        """
        if model is None:
            _, model, _ = self.get_best_model()

        result = sk_perm_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
        )

        df = pd.DataFrame({
            "feature":          self.feature_names,
            "importance_mean":  result.importances_mean,
            "importance_std":   result.importances_std,
        })
        df.sort_values("importance_mean", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    
    # COMPARISON DATAFRAME
    
    def get_comparison_df(self) -> pd.DataFrame:
        """
        Returns a clean comparison DataFrame for display in the dashboard.
        Includes a random-baseline row at 50%.
        """
        rows = [{
            "Model":               "Random Baseline",
            "Accuracy":            0.50,
            "ROC-AUC":             0.50,
            "F1":                  0.50,
            "Directional Acc":     0.50,
            "Status":              "baseline",
        }]

        best_name, _, _ = self.get_best_model() if self.results else (None, None, None)

        for name, r in self.results.items():
            rows.append({
                "Model":               name,
                "Accuracy":            round(r["accuracy"], 4),
                "ROC-AUC":             round(r["roc_auc"], 4),
                "F1":                  round(r["f1"], 4),
                "Directional Acc":     round(r["directional_accuracy"], 4),
                "Status":              "★ best" if name == best_name else "",
            })

        df = pd.DataFrame(rows)
        df.sort_values("ROC-AUC", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
