"""Generalization trust plugin.

Measures performance under scaffold/cluster (series-exclusive) folds, which is
the strongest QSAR generalization evidence the system collects.  Requires
scaffold groups; implemented as a fold-metric consumer of the experiment's
generalization splits.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.trust.base import (
    TrustEvaluator,
    aggregate_folds,
    classification_metrics,
    regression_metrics,
)


class Generalization(TrustEvaluator):
    name = "generalization"

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
        if not splits:
            return {"evaluated": False, "reason": "no generalization splits provided"}
        fold_metrics: list[dict[str, Any]] = []
        is_graph = isinstance(X, list) and len(X) > 0 and hasattr(X[0], "to_torch")
        for train_idx, test_idx in splits:
            if is_graph:
                X_tr = [X[i] for i in train_idx]
                X_te = [X[i] for i in test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            fitted = model.fit(X_tr, y_tr)
            pred = fitted.predict(X_te)
            if task_type == "regression":
                m = regression_metrics(y_te, pred)
            else:
                m = classification_metrics(y_te, pred)
            fold_metrics.append(m)
        return {
            **aggregate_folds(fold_metrics),
            "n_folds": len(fold_metrics),
            "fold_metrics": fold_metrics,
            "primary": "rmse" if task_type == "regression" else "balanced_accuracy",
        }

PLUGINS = [Generalization]
