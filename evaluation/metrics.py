"""
QuantEdge Evaluation Metrics
Standalone functions for all model evaluation and plotting.
All metrics computed on test set only.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    confusion_matrix, precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import sys

sys.path.insert(0, "..")
from config import PLOTLY_THEME, DARK_THEME



# DIRECTIONAL ACCURACY

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of correct directional calls (up/down).
    Identical to accuracy for binary classification, kept for semantic clarity.
    """
    return float(accuracy_score(y_true, y_pred))



# STYLED CLASSIFICATION REPORT

def classification_report_styled(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with per-class and overall metrics.
    """
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    overall_acc = accuracy_score(y_true, y_pred)

    rows = []
    for i, cls in enumerate(["DOWN (0)", "UP (1)"]):
        rows.append({
            "Class": cls,
            "Precision": round(prec[i], 4),
            "Recall": round(rec[i], 4),
            "F1-Score": round(f1[i], 4),
            "Support": int(support[i]),
        })
    rows.append({
        "Class": "── Overall ──",
        "Precision": round(overall_acc, 4),
        "Recall": round(overall_acc, 4),
        "F1-Score": round(2 * prec.mean() * rec.mean() / (prec.mean() + rec.mean()), 4),
        "Support": len(y_true),
    })
    rows.append({
        "Class": "ROC-AUC",
        "Precision": round(roc_auc, 4),
        "Recall": "",
        "F1-Score": "",
        "Support": "",
    })
    return pd.DataFrame(rows)



# ROC CURVES — ALL 3 MODELS

def plot_roc_curves(results_dict: dict) -> go.Figure:
    """
    Overlay ROC curves for all models on one dark-theme chart.
    
    Args:
        results_dict: {model_name: {"y_proba": array, "y_test": array}}
    """
    colors = {
        "LightGBM": DARK_THEME["accent"],
        "XGBoost":  DARK_THEME["purple"],
        "Logistic": DARK_THEME["yellow"],
    }

    fig = go.Figure()

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Baseline (AUC = 0.50)",
        line=dict(color=DARK_THEME["muted"], dash="dash", width=1),
    ))

    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"][:, 1])
        roc_auc = auc(fpr, tpr)
        short = name.replace("Regression", "Reg.")
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{short} (AUC = {roc_auc:.3f})",
            line=dict(color=colors.get(name, "#ffffff"), width=2),
        ))

    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text="ROC Curves — Test Set", font=dict(color=DARK_THEME["text"])),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(
            bgcolor=DARK_THEME["surface"],
            bordercolor=DARK_THEME["border"],
            font=dict(color=DARK_THEME["text"]),
        ),
        height=420,
    )
    return fig



# CONFUSION MATRIX

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Annotated confusion matrix heatmap with TP/TN/FP/FN labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = [
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"]
    ]

    fig = go.Figure(data=go.Heatmap(
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

    # 1. Apply the base global theme first
    fig.update_layout(**PLOTLY_THEME)

    # 2. Overwrite the specific layout values safely
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix — {model_name} (Test Set)",
            font=dict(color=DARK_THEME["text"])
        ),
        xaxis=dict(
            **PLOTLY_THEME.get("xaxis", {}),  # Safely bring in theme's xaxis settings
            title="Predicted Label",
            tickfont=dict(color=DARK_THEME["text"]),
        ),
        yaxis=dict(
            **PLOTLY_THEME.get("yaxis", {}),  # Safely bring in theme's yaxis settings
            title="True Label",
            tickfont=dict(color=DARK_THEME["text"]),
        ),
        height=380,
    )
    
    return fig


# CALIBRATION CURVE

def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Reliability diagram: predicted probability vs actual fraction positive.
    A perfectly calibrated model lies on the diagonal.
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=10)

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color=DARK_THEME["muted"], dash="dash", width=1),
    ))

    # Model calibration
    fig.add_trace(go.Scatter(
        x=mean_pred, y=fraction_pos,
        mode="lines+markers",
        name=f"{model_name}",
        line=dict(color=DARK_THEME["accent"], width=2),
        marker=dict(size=7, color=DARK_THEME["accent"]),
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text=f"Calibration Curve — Are Probabilities Reliable?",
            font=dict(color=DARK_THEME["text"])
        ),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives (Actual)",
        legend=dict(
            bgcolor=DARK_THEME["surface"],
            bordercolor=DARK_THEME["border"],
            font=dict(color=DARK_THEME["text"]),
        ),
        height=380,
    )
    return fig



# PRECISION-RECALL CURVE

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model"
) -> go.Figure:
    """
    Precision-Recall curve with average precision score.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
    ap = average_precision_score(y_true, y_proba[:, 1])
    baseline = y_true.mean()

    fig = go.Figure()

    # Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode="lines",
        name=f"Random Baseline (AP = {baseline:.2f})",
        line=dict(color=DARK_THEME["muted"], dash="dash", width=1),
    ))

    fig.add_trace(go.Scatter(
        x=rec, y=prec,
        mode="lines",
        name=f"{model_name} (AP = {ap:.3f})",
        line=dict(color=DARK_THEME["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text=f"Precision-Recall Curve — {model_name}",
            font=dict(color=DARK_THEME["text"])
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(
            bgcolor=DARK_THEME["surface"],
            bordercolor=DARK_THEME["border"],
            font=dict(color=DARK_THEME["text"]),
        ),
        height=380,
    )
    return fig



# ROLLING DIRECTIONAL ACCURACY

def rolling_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int = 30
) -> go.Figure:
    """
    30-day rolling directional accuracy over the test period.
    Shows whether model performance is stable or degrading over time.
    """
    correct = (y_true == y_pred).astype(float)
    s = pd.Series(correct, index=dates)
    rolling_acc = s.rolling(window).mean() * 100

    fig = go.Figure()

    # 50% baseline
    fig.add_hline(
        y=50,
        line_color=DARK_THEME["muted"],
        line_dash="dash",
        annotation_text="Random Baseline 50%",
        annotation_font_color=DARK_THEME["muted"],
    )

    fig.add_trace(go.Scatter(
        x=rolling_acc.index,
        y=rolling_acc.values,
        mode="lines",
        name=f"{window}-Day Rolling Accuracy",
        line=dict(color=DARK_THEME["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
    ))

    # 1. Apply the base global theme first
    fig.update_layout(**PLOTLY_THEME)

    # 2. Overwrite and add your custom layout values safely
    fig.update_layout(
        title=dict(
            text=f"{window}-Day Rolling Directional Accuracy (Test Period)",
            font=dict(color=DARK_THEME["text"])
        ),
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        yaxis=dict(
            **PLOTLY_THEME.get("yaxis", {}), # Safely unpack original yaxis theme settings
            range=[30, 80]                    # Explicitly layer your custom range on top
        ),
        legend=dict(
            bgcolor=DARK_THEME["surface"],
            bordercolor=DARK_THEME["border"],
            font=dict(color=DARK_THEME["text"]),
        ),
        height=380,
    )
    
    return fig