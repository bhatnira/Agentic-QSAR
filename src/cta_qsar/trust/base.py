"""Trust plugin base classes."""

from __future__ import annotations

import abc
from typing import Any, Protocol, runtime_checkable

import numpy as np


class TrustEvaluator(abc.ABC):
    """Base class for trust plugins."""

    name: str = ""

    @abc.abstractmethod
    def evaluate(
        self,
        *,
        model: Any,
        task_type: str,
        representation: Any,
        X: Any,
        y: Any,
        splits: list[tuple[Any, Any]],
        smiles: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one trust facet and return metric dicts."""


@runtime_checkable
class UncertaintyMethod(Protocol):
    name: str

    def scores(
        self, *, model: Any, X: Any, task_type: str, n_samples: int | None = None
    ) -> Any:
        """Per-sample uncertainty scores (higher = more uncertain)."""


@runtime_checkable
class ExplainabilityMethod(Protocol):
    name: str

    def explain(
        self, *, model: Any, X: Any, feature_names: list[str], task_type: str
    ) -> dict[str, Any]:
        """Feature importance mapping."""


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic),
    }


def classification_metrics(y_true: Any, y_pred: Any, y_proba: Any | None = None) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "macro"
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if n_classes == 2 else "macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, average=avg, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            from sklearn.metrics import average_precision_score

            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["pr_auc"] = float("nan")
        try:
            from sklearn.calibration import brier_score_loss

            metrics["brier"] = float(brier_score_loss(y_true, y_proba))
        except Exception:  # noqa: BLE001
            metrics["brier"] = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / max(tp + fn, 1))
        metrics["specificity"] = float(tn / max(tn + fp, 1))
    return metrics


def aggregate_folds(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean + std per metric across folds."""
    keys = set()
    for m in fold_metrics:
        keys.update(m.keys())
    agg: dict[str, Any] = {}
    for key in sorted(keys):
        values = [m[key] for m in fold_metrics if key in m and m[key] is not None]
        values = [v for v in values if not (isinstance(v, float) and v != v)]
        if not values:
            continue
        agg[key] = {"mean": float(sum(values) / len(values)), "std": float(_std(values))}
    return agg


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return float((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)


def primary_metric(task_type: str) -> str:
    if task_type in ("binary", "multiclass"):
        return "roc_auc" if task_type == "binary" else "mcc"
    return "rmse"