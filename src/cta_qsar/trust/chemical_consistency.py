"""Chemical-consistency trust plugin (lightweight).

Checks whether the model's most important features correspond to chemically
meaningful substructures when the representation has featurizable bits
(Morgan), and whether important descriptors align with plausible
physicochemical drivers.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.trust.base import TrustEvaluator


class ChemicalConsistency(TrustEvaluator):
    name = "chemical_consistency"

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
        """Best-effort: requires fitted tree ensemble importances or permutation results."""
        # This plugin is intentionally conservative: it only reports what it
        # can substantiate.  Full feature->substructure mapping is exposed as
        # a recommendation rather than a claim.
        return {
            "evaluated": False,
            "reason": (
                "deferred: requires a fitted explainable model and substructure "
                "mapping; see explainability plugin for intermediate evidence"
            ),
            "recommendation": (
                "map top-SMARTs substructures to important Morgan bits when a "
                "tree ensemble is selected"
            ),
        }