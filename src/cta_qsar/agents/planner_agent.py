"""Planner agent: candidate generation + LLM strategy selection."""

from __future__ import annotations

from typing import Any

from cta_qsar.core.interfaces import ExperimentCandidate, QSARCase
from cta_qsar.core.logging import get_logger
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.experiments.planner import (
    explain_decisions,
    generate_candidates,
)
from cta_qsar.llm.base import ReasoningModel

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, registry: PluginRegistry, llm: ReasoningModel | None = None) -> None:
        self.registry = registry
        self.llm = llm

    def plan(
        self,
        *,
        case: QSARCase,
        enabled_representations: list[str] | None,
        enabled_models: list[str] | None,
        validated_splits: list[str],
        budget: BudgetState,
        history: list[dict[str, Any]],
        dataset_props: dict[str, Any],
        n_samples: int,
        hardware_tier: str,
        kg_context: str = "",
        evidence_facts: list[Any] | None = None,
        policy_weights: dict[str, float] | None = None,
    ) -> tuple[list[ExperimentCandidate], list[ExperimentCandidate]]:
        """Return (ranked_candidates, rejected_candidates).

        ``kg_context`` is a rendered knowledge-graph digest (see
        knowledge.explain.render_evidence_board) grounded into the LLM prompt;
        ``evidence_facts`` are the raw Facts used for heuristic re-ranking;
        ``policy_weights`` are the self-improving-planner multipliers used in
        utility ranking (defaults to 1.0 when the policy is frozen).
        """
        candidates = generate_candidates(
            registry=self.registry,
            case=case,
            enabled_representations=enabled_representations,
            enabled_models=enabled_models,
            validated_splits=validated_splits,
            budget=budget,
            history=history,
            task_type=case.task_type,
            n_samples=n_samples,
            dataset_props=dataset_props,
            hardware_tier=hardware_tier,
            evidence=evidence_facts or [],
            policy_weights=policy_weights,
        )
        usage = explain_decisions(candidates)
        logger.info("Candidate planning:\n%s", usage)
        if kg_context:
            logger.info("Knowledge-graph context:\n%s", kg_context)

        llm_refinement: list[ExperimentCandidate] = []
        if self.llm is not None:
            try:
                selection = self.llm.select_strategy(
                    {
                        "case": case.model_dump(),
                        "ranked_candidates": [c.model_dump() for c in candidates[:8]],
                        "history": history[-5:],
                        "budget": budget.to_dict(),
                        "evidence_context": kg_context,
                    }
                )
                llm_refinement = _llm_plan_to_candidates(
                    selection.experiment_plan, candidates
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM strategy selection failed (%s); using heuristic ranking", exc)
        return candidates, llm_refinement


def _llm_plan_to_candidates(
    plan: list[dict[str, Any]], candidates: list[ExperimentCandidate]
) -> list[ExperimentCandidate]:
    """Map LLM-selected plan entries back onto heuristic candidates."""
    by_key: dict[str, ExperimentCandidate] = {}
    for candidate in candidates:
        by_key[f"{candidate.representation}|{candidate.model}|{candidate.validation}"] = candidate
    picked: list[ExperimentCandidate] = []
    for entry in plan:
        key = f"{entry.get('representation')}|{entry.get('model')}|{entry.get('validation')}"
        if key in by_key:
            candidate = by_key[key]
            candidate.llm_rationale = entry.get("reason", "")
            picked.append(candidate)
    return picked