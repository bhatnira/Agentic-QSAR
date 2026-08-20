"""Deterministic heuristic LLM: a full ReasoningModel that requires no API.

Used for tests, CPU/dev profiles, and as a graceful fallback when no API key
is configured.  It produces defensible rule-based decisions from the same
structured evidence the real LLMs consume.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.llm.base import (
    CaseClassification,
    DiagnosisOutput,
    InterventionOutput,
    ReasoningModel,
    StopDecision,
    StrategySelection,
)
from cta_qsar.llm.providers import ProviderSpec, register_provider


def build(
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> HeuristicModel:
    """Factory used by the provider registry; extra kwargs pass through."""
    return HeuristicModel(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)


register_provider(
    ProviderSpec(
        name="mock",
        build=build,
        requires_env=(),
        description="Deterministic heuristic model; no network or API key required",
        aliases=("heuristic",),
    )
)


class HeuristicModel(ReasoningModel):
    provider_name = "mock"

    def classify_case(self, case_context: dict[str, Any]) -> CaseClassification:
        task_type = case_context.get("endpoint", {}).get("task_type", "regression")
        risks = list(case_context.get("quality_report", {}).get("risks", []))
        if not risks:
            issues = case_context.get("quality_report", {})
            if issues.get("conflicting_labels", {}).get("n_conflicting_groups", 0):
                risks.append("conflicting labels on duplicate molecules")
            if issues.get("duplicate_molecules", {}).get("n_duplicates", 0):
                risks.append("duplicate molecules present")
            if issues.get("class_balance", {}).get("imbalance_ratio", 1) > 5:
                risks.append("strong class imbalance")
        return CaseClassification(
            problem_statement=case_context.get("case", {}).get("problem_statement", ""),
            task_type=task_type,
            risks=risks,
            validation_strategy=["random", "scaffold"],
            representation_strategy=["morgan", "rdkit_descriptors"],
            model_strategy=["ridge", "random_forest", "xgboost"],
            reasoning="heuristic profile: use cheap baselines, then scaffold-based generalization evidence",
        )

    def select_strategy(self, strategy_context: dict[str, Any]) -> StrategySelection:
        ranked = strategy_context.get("ranked_candidates", [])
        plan = []
        for item in ranked[:3]:
            plan.append(
                {
                    "representation": item.get("representation"),
                    "model": item.get("model"),
                    "validation": item.get("validation"),
                    "hyperparameter_budget": item.get("hyperparameter_budget", 1),
                    "reason": item.get("reason", ""),
                }
            )
        return StrategySelection(
            experiment_plan=plan,
            rationale="heuristic: utility-sorted candidates (scientific value per compute)",
            evidence_cited=["ranked_candidates"],
        )

    def diagnose(self, experiment_context: dict[str, Any]) -> list[DiagnosisOutput]:
        diagnoses = experiment_context.get("deterministic_diagnoses", [])
        return [
            DiagnosisOutput(
                failure_type=d.get("failure_type", "unknown"),
                evidence=d.get("evidence", {}),
                hypothesis=d.get("hypothesis", ""),
                confidence=d.get("confidence", 0.0),
                recommended_actions=d.get("recommended_actions", []),
            )
            for d in diagnoses
        ]

    def propose_intervention(self, diagnosis_context: dict[str, Any]) -> InterventionOutput:
        ranked = diagnosis_context.get("ranked_interventions", [])
        if not ranked:
            return InterventionOutput(action="none", description="no interventions proposed")
        best = ranked[0]
        return InterventionOutput(
            action=best.get("name", "alternative_representation"),
            description=best.get("description", ""),
            expected_improvement=best.get("expected_improvement", 0.1),
            expected_trust_gain=best.get("expected_trust_gain", 0.1),
            compute_cost=best.get("compute_cost", 1.0),
            rationale=best.get("rationale", ""),
        )

    def decide_stop(self, budget_context: dict[str, Any]) -> StopDecision:
        remaining = budget_context.get("experiments_remaining", 0)
        no_improvement_rounds = budget_context.get("no_improvement_rounds", 0)
        if remaining <= 0:
            return StopDecision(should_stop=True, reason="experiment budget exhausted")
        if no_improvement_rounds >= 2:
            return StopDecision(
                should_stop=True,
                reason="no meaningful improvement observed in recent experiments",
            )
        return StopDecision(should_stop=False, reason="continue experimentation")

    def summarize(self, report_context: dict[str, Any]) -> str:
        n_experiments = len(report_context.get("experiments", []))
        best = report_context.get("best_experiment", {})
        summary = (
            f"The agent completed {n_experiments} experiments. "
            f"Best observed strategy under the specified budget: "
            f"{best.get('representation', 'n/a')} + {best.get('model', 'n/a')} "
            f"with {best.get('split', 'n/a')} validation."
        )
        return summary


MockLLM = HeuristicModel
"""Alias used by tests and a fallback path."""