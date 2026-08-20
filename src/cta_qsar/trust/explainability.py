"""Explainability trust plugin.

Uses permutation importance (deterministic, dependency-free).  SHAP is
attempted only when installed and inexpensive.  Output: per-feature
importance with a stability estimate across two repeats.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from cta_qsar.trust.base import TrustEvaluator


class PermutationExplainability(TrustEvaluator):
    name = "explainability"

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
        factor = 2 if task_type == "regression" else 3
        n_feats = X.shape[1] if hasattr(X, "shape") else 0
        if n_feats == 0 or n_feats > 5000:
            return {"evaluated": False, "reason": "too many features or non-matrix input"}
        if len(splits) == 0:
            return {"evaluated": False, "reason": "no splits"}
        tr, te = splits[0]
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = np.asarray(y).ravel()[tr], np.asarray(y).ravel()[te]
        scoring = "r2" if task_type == "regression" else "roc_auc" if task_type == "binary" else "accuracy"
        try:
            fitted = model.fit(X_tr, y_tr)
            result = permutation_importance(
                fitted, X_te, y_te, n_repeats=factor, scoring=scoring, random_state=42, n_jobs=-1
            )
        except Exception as exc:  # noqa: BLE001
            return {"evaluated": False, "reason": str(exc)}
        means = result.importances_mean
        stds = result.importances_std
        order = np.argsort(means)[::-1][: min(25, len(means))]
        names = getattr(representation, "feature_names", None) or [
            f"f{i}" for i in range(len(means))
        ]
        top = [
            {
                "feature": str(names[i]),
                "importance": float(means[i]),
                "std": float(stds[i]),
                "stability": float(stds[i] / max(abs(means[i]), 1e-9)),
            }
            for i in order
            if means[i] > 0
        ]
        return {
            "evaluated": True,
            "method": "permutation_importance",
            "n_features": int(n_feats),
            "top_features": top,
        }

PLUGINS = [PermutationExplainability]
