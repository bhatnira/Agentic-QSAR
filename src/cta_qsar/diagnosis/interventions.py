"""Intervention proposals and ranking by expected value / cost."""

from __future__ import annotations

from typing import Any

from cta_qsar.core.interfaces import FailureDiagnosis, ProposedIntervention

_BASELINE_ACTIONS: dict[str, dict[str, Any]] = {
    "alternative_representation": {
        "description": "Try a different feature space (other fingerprint family, descriptors, or graph).",
    },
    "graph_model": {
        "description": "Use a graph convolutional network that learns representations from structure.",
    },
    "stronger_regularization": {
        "description": "Reduce model capacity / increase regularization strength.",
    },
    "class_weighted": {
        "description": "Balance classes via class_weight/resampling for imbalanced endpoints.",
    },
    "ensemble": {
        "description": "Average predictions across seeds/folds to stabilize performance.",
    },
    "scaffold_aware_selection": {
        "description": "Select and report models under scaffold-holdout generalization.",
    },
    "feature_selection": {
        "description": "Prune noisy descriptors to reduce overfitting.",
    },
    "more_data_like": {
        "description": "Add structurally similar training molecules to expand the applicability domain.",
    },
    "uncertainty_reporting": {
        "description": "Report per-molecule confidence and flag OOD predictions.",
    },
}

_FAILURE_TO_ACTIONS: dict[str, list[str]] = {
    "chemical_series_dependence": [
        "alternative_representation", "graph_model", "scaffold_aware_selection", "ensemble",
    ],
    "overfitting": [
        "stronger_regularization", "feature_selection", "simpler_model", "ensemble",
    ],
    "class_imbalance": ["class_weighted", "scaffold_aware_selection"],
    "limited_applicability_domain": [
        "uncertainty_reporting", "more_data_like", "alternative_representation",
    ],
    "unstable_representation": [
        "alternative_representation", "ensemble", "simpler_model",
    ],
    "insufficient_data": ["simpler_model", "feature_selection"],
    "label_noise": ["more_data_like", "uncertainty_reporting"],
}

_COSTS: dict[str, float] = {
    "alternative_representation": 2.0,
    "graph_model": 8.0,
    "stronger_regularization": 1.0,
    "class_weighted": 1.0,
    "ensemble": 2.0,
    "scaffold_aware_selection": 1.5,
    "feature_selection": 1.0,
    "more_data_like": 1.0,
    "uncertainty_reporting": 0.5,
    "simpler_model": 0.5,
}


def propose(diagnosis: FailureDiagnosis, context: dict[str, Any] | None = None) -> list[ProposedIntervention]:
    """Map a diagnosis to ranked interventions.

    Ranking: (expected_improvement + trust_gain) / compute_cost.
    """
    action_names = _FAILURE_TO_ACTIONS.get(diagnosis.failure_type, ["alternative_representation"])
    interventions: list[ProposedIntervention] = []
    specificity = diagnosis.confidence
    for index, action in enumerate(action_names):
        template = _BASELINE_ACTIONS.get(action, {"description": action})
        improvement = max(0.1, specificity * (0.6 - index * 0.1))
        trust_gain = 0.3 if action in ("scaffold_aware_selection", "uncertainty_reporting") else 0.1
        cost = _COSTS.get(action, 1.0)
        interventions.append(
            ProposedIntervention(
                name=action,
                description=template["description"],
                expected_improvement=improvement,
                expected_trust_gain=trust_gain,
                compute_cost=cost,
                actions={"representation_hint": _refine(action, context or {})},
                rationale=f"from diagnosis {diagnosis.failure_type} (conf {diagnosis.confidence:.2f})",
            )
        )
    interventions.sort(
        key=lambda i: (i.expected_improvement + i.expected_trust_gain) / max(i.compute_cost, 1e-6),
        reverse=True,
    )
    return interventions


def _refine(action: str, context: dict[str, Any]) -> str:
    if action != "alternative_representation":
        return ""
    experiments = context.get("experiments", [])
    tried = {
        e.get("representation") if isinstance(e, dict) else getattr(e, "representation", None)
        for e in experiments
    }
    for candidate in ("mordred", "graph", "foundation_embeddings", "rdkit_descriptors", "torsion"):
        if candidate not in tried:
            return candidate
    return ""


class InterventionProposerPlugin:
    """Registry-facing wrapper around the intervention rules."""

    name = "intervention_proposer"

    def propose(self, diagnosis: FailureDiagnosis, context: dict[str, Any]) -> list[ProposedIntervention]:
        return propose(diagnosis, context)


PLUGINS = [InterventionProposerPlugin]
