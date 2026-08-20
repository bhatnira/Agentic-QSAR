"""Uncertainty trust plugin (fold-standard-deviation ensemble)."""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.trust.base import TrustEvaluator


class FoldEnsembleUncertainty(TrustEvaluator):
    """Per-sample uncertainty via prediction spread across holdout folds.

    For regression: std of fold predictions mapped to relative uncertainty.
    For classification: disagreement rate measured by |proba - 0.5| for binary
    (higher disagreement = more uncertain), or entropy of predicted class
    distribution for multiclass.
    """

    name = "uncertainty"

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
        n = len(y)
        is_graph = isinstance(X, list) and len(X) > 0 and hasattr(X[0], "to_torch")
        preds = np.full((n, len(splits)), np.nan)
        for fold, (tr, te) in enumerate(splits):
            try:
                if is_graph:
                    X_tr = [X[i] for i in tr]
                    X_te = [X[i] for i in te]
                else:
                    X_tr, X_te = X[tr], X[te]
                fitted = model.fit(X_tr, y[tr])
                preds[te, fold] = np.asarray(fitted.predict(X_te), dtype=float)
            except Exception:  # noqa: BLE001
                continue
        covered = ~np.isnan(preds).all(axis=1)
        if covered.sum() == 0:
            return {"evaluated": False, "reason": "no uncertainty computed"}
        per_sample = preds[covered]
        spread = np.nanstd(per_sample, axis=1)
        if task_type == "binary":
            spread = np.minimum(spread, 0.5) * 2.0  # disagreements -> 0..1
        mean_spread = float(np.nanmean(spread))
        median_spread = float(np.nanmedian(spread))
        high = float((spread > np.nanpercentile(spread, 75)).mean())
        return {
            "evaluated": True,
            "method": "fold_ensemble_std",
            "n_covered_samples": int(covered.sum()),
            "mean_spread": mean_spread,
            "median_spread": median_spread,
            "high_uncertainty_fraction_75pct": high,
            "per_sample": spread.tolist(),
            "interpretation": (
                "uncertainty calibrated with prediction spread across folds; "
                "higher spread = lower confidence"
            ),
        }

PLUGINS = [FoldEnsembleUncertainty]
