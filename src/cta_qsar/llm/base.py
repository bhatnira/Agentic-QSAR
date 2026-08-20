"""ReasoningModel: the abstract LLM contract.

The scientist agent reasons through structured scientific information only;
all numeric/chemical computation is done by deterministic Python tools that
are passed to the LLM as evidence.  The LLM selects strategies, diagnoses
failures, proposes interventions, and summarizes --- it never calculates.
"""

from __future__ import annotations

import abc
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    content: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    raw: str = ""
    provider: str = ""
    model: str = ""


class CaseClassification(BaseModel):
    problem_statement: str = ""
    task_type: str = ""
    risks: list[str] = Field(default_factory=list)
    validation_strategy: list[str] = Field(default_factory=list)
    representation_strategy: list[str] = Field(default_factory=list)
    model_strategy: list[str] = Field(default_factory=list)
    reasoning: str = ""


class StrategySelection(BaseModel):
    experiment_plan: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    evidence_cited: list[str] = Field(default_factory=list)


class DiagnosisOutput(BaseModel):
    failure_type: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str = ""
    confidence: float = 0.0
    recommended_actions: list[str] = Field(default_factory=list)


class InterventionOutput(BaseModel):
    action: str = ""
    description: str = ""
    expected_improvement: float = 0.0
    expected_trust_gain: float = 0.0
    compute_cost: float = 1.0
    rationale: str = ""


class StopDecision(BaseModel):
    should_stop: bool = False
    reason: str = ""
    next_candidate: dict[str, Any] | None = None


class ReasoningModel(abc.ABC):
    """Interface for LLM-backed scientific reasoning."""

    provider_name: str = "abstract"

    def __init__(self, model: str = "", temperature: float = 0.1, max_tokens: int = 4096) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- high-level capabilities ------------------------------------------
    @abc.abstractmethod
    def classify_case(self, case_context: dict[str, Any]) -> CaseClassification:
        """Understand the QSAR problem from the profiled dataset."""

    @abc.abstractmethod
    def select_strategy(self, strategy_context: dict[str, Any]) -> StrategySelection:
        """Choose the (rep, model, validation) strategy given candidates."""

    @abc.abstractmethod
    def diagnose(self, experiment_context: dict[str, Any]) -> list[DiagnosisOutput]:
        """Interpret trust evidence into failure hypotheses."""

    @abc.abstractmethod
    def propose_intervention(self, diagnosis_context: dict[str, Any]) -> InterventionOutput:
        """Choose the highest-value intervention for a diagnosis."""

    @abc.abstractmethod
    def decide_stop(self, budget_context: dict[str, Any]) -> StopDecision:
        """Decide whether further experimentation is worthwhile."""

    @abc.abstractmethod
    def summarize(self, report_context: dict[str, Any]) -> str:
        """Produce the scientific narrative for the final report."""

    # -- low-level prompt plumbing ----------------------------------------
    def _render_prompt(self, template: str, context: dict[str, Any]) -> str:
        payload = yaml.safe_dump(context, sort_keys=False, default_flow_style=False)
        return template.replace("{{CONTEXT}}", payload)

    def _call_json(self, prompt: str, schema: type[BaseModel] | None = None) -> LLMResponse:
        raise NotImplementedError


PROMPTS = {
    "case": """You are the CTA-QSAR scientist. A QSAR dataset has been profiled.
Use ONLY the structured evidence below. Cite the evidence you use. Do not
invent numbers, chemistry, or references.

Return a JSON object with keys: problem_statement, task_type, risks,
validation_strategy, representation_strategy, model_strategy, reasoning.

{{CONTEXT}}
""",
    "strategy": """You are the CTA-QSAR experiment planner. Choose the single
next experiment with the highest scientific value per unit compute. Consider
the experiment history and candidate ranking evidence below. Fabricated
results are forbidden; you only rank planned experiments.

Return JSON: {"experiment_plan": [{"representation","model","validation",
"hyperparameter_budget","reason"}], "rationale": "...", "evidence_cited": [...]}.

{{CONTEXT}}
""",
    "diagnosis": """You are the CTA-QSAR failure analyst. Diagnose the failure
of the experiment from its trust report. Use only the evidence shown. Do not
invent chemistry.

Return a JSON list of diagnoses: [{"failure_type","evidence","hypothesis",
"confidence", "recommended_actions"}].

{{CONTEXT}}
""",
    "intervention": """You are the CTA-QSAR self-correction specialist.
Given a failure diagnosis, propose ONE intervention with the highest
expected improvement + trust gain per unit compute. Do not invent results.

Return JSON: {"action","description","expected_improvement",
"expected_trust_gain","compute_cost","rationale"}.

{{CONTEXT}}
""",
    "stop": """You are the CTA-QSAR budget controller. Decide whether another
experiment is worthwhile given the remaining budget and observed
improvements. Be conservative with compute.

Return JSON: {"should_stop": bool, "reason": "...", "next_candidate": {...}|null}.

{{CONTEXT}}
""",
    "report": """You are the CTA-QSAR scientific writer. Write the executive
summary for the final report (2-4 paragraphs). Use ONLY the results shown.
Never claim optimality; use 'best observed strategy under the specified
budget'. Do not invent references.

{{CONTEXT}}
""",
}


def prompt_for(kind: str) -> str:
    return PROMPTS[kind]