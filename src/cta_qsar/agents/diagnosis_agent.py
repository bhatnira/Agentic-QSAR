"""Diagnosis agent: deterministic rules + LLM refinement + interventions."""

from __future__ import annotations

from typing import Any

from cta_qsar.core.interfaces import (
    ExperimentRecord,
    FailureDiagnosis,
    ProposedIntervention,
)
from cta_qsar.core.logging import get_logger
from cta_qsar.diagnosis.failure import diagnose
from cta_qsar.diagnosis.interventions import propose
from cta_qsar.llm.base import ReasoningModel

logger = get_logger(__name__)


class DiagnosisAgent:
    def __init__(self, llm: ReasoningModel | None = None) -> None:
        self.llm = llm

    def run(
        self, experiment: ExperimentRecord, context: dict[str, Any]
    ) -> tuple[list[FailureDiagnosis], list[ProposedIntervention]]:
        """Deterministic diagnosis + interventions; LLM improves if available."""
        diagnoses = diagnose(experiment, experiment.trust)
        interventions = self._interventions_for(diagnoses, context)

        if self.llm is not None and diagnoses:
            try:
                llm_diagnoses = self.llm.diagnose(
                    {
                        "experiment": experiment.model_dump(),
                        "deterministic_diagnoses": [d.model_dump() for d in diagnoses],
                    }
                )
                if llm_diagnoses:
                    merged: list[FailureDiagnosis] = []
                    for deterministic in diagnoses:
                        match = next(
                            (
                                d
                                for d in llm_diagnoses
                                if d.failure_type == deterministic.failure_type
                            ),
                            None,
                        )
                        merged.append(
                            FailureDiagnosis(
                                failure_type=deterministic.failure_type,
                                evidence=deterministic.evidence,
                                hypothesis=match.hypothesis if match else deterministic.hypothesis,
                                confidence=match.confidence if match else deterministic.confidence,
                                recommended_actions=(
                                    match.recommended_actions
                                    if match
                                    else deterministic.recommended_actions
                                ),
                            )
                        )
                    diagnoses = merged
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM diagnosis refinement failed (%s)", exc)

        if self.llm is not None and diagnoses:
            try:
                proposal = self.llm.propose_intervention(
                    {
                        "diagnosis": diagnoses[0].model_dump(),
                        "ranked_interventions": [
                            i.model_dump() for i in interventions[:5]
                        ],
                    }
                )
                if proposal.action and proposal.action != "none":
                    interventions.insert(
                        0,
                        ProposedIntervention(
                            name=proposal.action,
                            description=proposal.description,
                            expected_improvement=proposal.expected_improvement,
                            expected_trust_gain=proposal.expected_trust_gain,
                            compute_cost=proposal.compute_cost,
                            rationale=proposal.rationale or "LLM proposal",
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM intervention proposal failed (%s)", exc)
        return diagnoses, interventions

    @staticmethod
    def _interventions_for(
        diagnoses: list[FailureDiagnosis], context: dict[str, Any]
    ) -> list[ProposedIntervention]:
        all_interventions: list[ProposedIntervention] = []
        for diagnosis in diagnoses:
            all_interventions.extend(propose(diagnosis, context))
        seen: set[str] = set()
        unique: list[ProposedIntervention] = []
        for intervention in all_interventions:
            if intervention.name in seen:
                continue
            seen.add(intervention.name)
            unique.append(intervention)
        unique.sort(
            key=lambda i: (i.expected_improvement + i.expected_trust_gain)
            / max(i.compute_cost, 1e-6),
            reverse=True,
        )
        return unique