"""Robustness trust plugin: seed and split sensitivity."""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.trust.base import TrustEvaluator, classification_metrics, regression_metrics


def _score(model: Any, task_type: str, X_tr: Any, y_tr: Any, X_te: Any, y_te: Any) -> float:
    fitted = model.fit(X_tr, y_tr)
    pred = fitted.predict(X_te)
    if task_type == "regression":
        m = regression_metrics(y_te, pred)
        return -float(m["rmse"])  # higher is better
    m = classification_metrics(y_te, pred)
    return float(m["balanced_accuracy"])


class SeedSensitivity(TrustEvaluator):
    name = "robustness"

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
        """Train on the first split's train fold with a few seeds, report score spread."""
        if not splits:
            return {"evaluated": False, "reason": "no splits"}
        is_graph = isinstance(X, list) and len(X) > 0 and hasattr(X[0], "to_torch")
        train_idx, test_idx = splits[0][0], splits[0][1]
        if is_graph:
            X_tr = [X[i] for i in train_idx]
            X_te = [X[i] for i in test_idx]
        else:
            X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = np.asarray(y).ravel()[train_idx], np.asarray(y).ravel()[test_idx]
        seeds = (42, 1337, 2024)
        scores: list[float] = []
        base_model = model
        for seed in seeds:
            candidate = _clone_with_seed(base_model, seed)
            if candidate is None:
                continue
            try:
                scored = _score(candidate, task_type, X_tr, y_tr, X_te, y_te)
                scores.append(scored)
            except Exception:  # noqa: BLE001
                continue
        if not scores:
            return {"evaluated": False, "reason": "model not reseedable or failed"}
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return {
            "evaluated": True,
            "n_seeds": len(scores),
            "score_mean": mean,
            "score_std": std,
            "coefficient_of_variation": float(std / max(abs(mean), 1e-9)),
            "scores": scores,
            "interpretation": (
                "stable" if std <= max(0.05, abs(mean) * 0.1) else "unstable"
            ),
        }


def _clone_with_seed(model: Any, seed: int) -> Any:
    """Rebuild the estimator with a different random_state where supported."""
    try:
        params = model.get_params(deep=False)
    except Exception:  # noqa: BLE001
        return None
    candidate_params = dict(params)
    found = False
    for key in ("random_state", "random_seed", "seed"):
        if key in candidate_params:
            candidate_params[key] = seed
            found = True
    if not found:
        return None
    try:
        return type(model)(**candidate_params)
    except Exception:  # noqa: BLE001
        return None

PLUGINS = [SeedSensitivity]
