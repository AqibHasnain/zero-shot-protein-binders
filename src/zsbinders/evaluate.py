"""Scoring metrics shared by the zero-shot runs and the supervised baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_CLASS = "binder"


def binarize(labels: pd.Series) -> np.ndarray:
    return (labels == POSITIVE_CLASS).astype(int).to_numpy()


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on these labels.

    Optimistic by construction: it is tuned on the same data it is scored on, and exists to
    separate ranking quality from the calibration of the fixed 0.5 cut.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[f1[:-1].argmax()])


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    return {
        "n": len(y_true),
        "positive_rate": float(y_true.mean()),
        "majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
    }


def evaluate_scored(scored_df: pd.DataFrame, label: str = "zero-shot") -> dict[str, dict[str, float]]:
    """Metrics at the fixed 0.5 cut and at the oracle-best F1 threshold."""
    usable = scored_df.dropna(subset=["p_binder"])
    dropped = len(scored_df) - len(usable)
    if dropped:
        print(f"[warn] {dropped} designs failed to score and are excluded (see the 'error' column)")

    y_true = binarize(usable["y_true"])
    scores = usable["p_binder"].to_numpy(dtype=float)
    tuned = best_f1_threshold(y_true, scores)
    return {
        f"{label} @0.5": metrics(y_true, scores, 0.5),
        f"{label} @best-f1": metrics(y_true, scores, tuned),
    }


def report(scored_df: pd.DataFrame, threshold: float = 0.5) -> str:
    """Per-class report, confusion matrix, and per-target accuracy as printable text."""
    usable = scored_df.dropna(subset=["p_binder"])
    y_true = binarize(usable["y_true"])
    y_pred = (usable["p_binder"].to_numpy(dtype=float) >= threshold).astype(int)

    per_target = (
        usable.assign(correct=(y_true == y_pred))
        .groupby("target")["correct"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    lines = [
        f"threshold: {threshold:.3f}",
        "confusion matrix [rows=true, cols=pred], order=[non_binder, binder]:",
        str(confusion_matrix(y_true, y_pred)),
        "",
        classification_report(y_true, y_pred, target_names=["non_binder", "binder"], zero_division=0),
        "per-target accuracy:",
        per_target.to_string(),
    ]
    return "\n".join(lines)


def metrics_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.rename_axis("model")
