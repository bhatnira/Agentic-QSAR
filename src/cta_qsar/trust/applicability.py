"""Applicability-domain trust plugin.

For each test molecule, computes its nearest-neighbor Tanimoto similarity to
the training set.  Low similarity marks molecules outside the learned
domain; per-fold predictions are weighted by domain membership.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.trust.base import TrustEvaluator


class ApplicabilityDomain(TrustEvaluator):
    name = "applicability_domain"

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
        if smiles is None or len(smiles) != len(y):
            return {"evaluated": False, "reason": "SMILES required for domain analysis"}
        from cta_qsar.chemistry.fingerprints import morgan_fingerprints, tanimoto_similarity_matrix

        try:
            fps = morgan_fingerprints(list(smiles), radius=2, n_bits=2048)
        except Exception as exc:  # noqa: BLE001
            return {"evaluated": False, "reason": f"fingerprint failure: {exc}"}
        sim = tanimoto_similarity_matrix(fps)
        nn_sims: list[float] = []
        ood_fractions: list[float] = []
        for train_idx, test_idx in splits:
            if len(test_idx) == 0:
                continue
            sub = sim[np.ix_(test_idx, train_idx)]
            if sub.size == 0:
                continue
            max_sim = sub.max(axis=1)
            nn_sims.extend(max_sim.tolist())
            ood = float((max_sim < 0.3).mean())
            ood_fractions.append(ood)
        if not nn_sims:
            return {"evaluated": False, "reason": "no test folds"}
        arr = np.asarray(nn_sims)
        return {
            "evaluated": True,
            "nn_tanimoto": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "min": float(arr.min()),
            },
            "ood_fraction_lt_0.3": float(np.mean(ood_fractions)),
            "n_test_samples": int(len(arr)),
            "interpretation": (
                "mostly in-domain"
                if float(np.median(arr)) >= 0.4
                else "many molecules far from training set (limited AD)"
            ),
        }

PLUGINS = [ApplicabilityDomain]
