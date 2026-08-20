"""Predictive-performance trust plugin."""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.trust.base import (
    TrustEvaluator,
    aggregate_folds,
    classification_metrics,
    primary_metric,
    regression_metrics,
)


class PredictivePerformance(TrustEvaluator):
    name = "predictive"

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
        y = np.asarray(y).ravel()
        has_proba = hasattr(model, "predict_proba")
        fold_metrics: list[dict[str, Any]] = []
        for train_idx, test_idx in splits:
            if len(test_idx) == 0:
                continue
            if _is_graph_input(X):
                X_tr = [X[i] for i in train_idx]
                X_te = [X[i] for i in test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if _is_graph_input(X):
                fitted = model.fit(X_tr, y_tr)
                pred = fitted.predict(X_te)
            else:
                fitted = model.fit(X_tr, y_tr)
                pred = fitted.predict(X_te)
            if task_type == "regression":
                m = regression_metrics(y_te, pred)
            else:
                proba = None
                if has_proba and task_type == "binary":
                    proba = fitted.predict_proba(X_te)
                    if proba.ndim > 1:
                        proba = proba[:, 1]
                elif has_proba and task_type == "multiclass":
                    proba = fitted.predict_proba(X_te)
                m = classification_metrics(y_te, pred, proba)
            fold_metrics.append(m)
        if not fold_metrics:
            return {"error": "no folds evaluated"}
        aggregated = aggregate_folds(fold_metrics)
        primary = primary_metric(task_type)
        return {
            **aggregated,
            "primary_metric": primary,
            "n_folds": len(fold_metrics),
            "fold_metrics": fold_metrics,
        }


def _is_graph_input(X: Any) -> bool:
    return isinstance(X, list) and len(X) > 0 and hasattr(X[0], "to_torch")

PLUGINS = [PredictivePerformance]
