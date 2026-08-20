"""Core plugin interfaces.

Every extensible capability of CTA-QSAR is a plugin implementing one of these
abstract protocols.  The orchestration engine only depends on these interfaces,
so new capabilities can be added without touching the LangGraph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Amounts & plans (shared value objects)
# --------------------------------------------------------------------------


class SplitPlan(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    recommended: bool = False


@dataclass
class CostEstimate:
    runtime_seconds: float
    memory_gb: float
    gpu_required: bool = False

    @property
    def normalized_cost(self) -> float:
        """CPU-seconds normalized to 1 CPU core second."""
        cores = 1.0
        return self.runtime_seconds / max(cores, 0.001) + self.memory_gb * 10.0


# --------------------------------------------------------------------------
# Column-type plugins
# --------------------------------------------------------------------------


@runtime_checkable
class EndpointPlugin(Protocol):
    """Understands a class of QSAR endpoints."""

    name: str
    task_type: str

    def applicable(self, column: pd.Series) -> bool: ...

    def detect(self, column: pd.Series, column_name: str) -> dict[str, Any]: ...


@runtime_checkable
class PreprocessingPlugin(Protocol):
    """Transforms a raw dataset before feature extraction."""

    name: str

    def fit(self, df: pd.DataFrame) -> PreprocessingPlugin: ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


# --------------------------------------------------------------------------
# Representation plugins
# --------------------------------------------------------------------------


@runtime_checkable
class RepresentationPlugin(Protocol):
    """Converts standardized molecules into a feature space."""

    name: str
    version: str

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        """Return (applicable, reason)."""

    def estimate_cost(self, n_molecules: int) -> CostEstimate: ...

    def fit(self, smiles: list[str]) -> RepresentationPlugin: ...

    def transform(self, smiles: list[str]) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# Model plugins
# --------------------------------------------------------------------------


@runtime_checkable
class ModelPlugin(Protocol):
    """A predictive model that scikit-learn can consume or that is self-contained."""

    name: str
    supports: tuple[str, ...]  # task types: regression, binary, multiclass, multitask

    def applicability(self, task_type: str, representation_name: str) -> tuple[bool, str]: ...

    def estimate_cost(
        self, n_samples: int, n_features: int, representation_name: str
    ) -> CostEstimate: ...

    def build_estimator(
        self, task_type: str, n_classes: int | None = None, **hyperparams: Any
    ) -> Any:
        """Return a scikit-learn compatible estimator."""

    def hyperparameter_space(self) -> dict[str, list[Any] | tuple[Any, Any]]:
        """Suggested hyperparameter candidates for cheap tuning."""


# --------------------------------------------------------------------------
# Validation plugins
# --------------------------------------------------------------------------


@runtime_checkable
class ValidationPlugin(Protocol):
    """Produces CV folds for a dataset."""

    name: str

    def applicability(self, dataset_props: dict[str, Any]) -> tuple[bool, str]: ...

    def split(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        task_type: str,
        n_splits: int = 5,
        n_repeats: int = 2,
        random_seed: int = 42,
        test_fraction: float = 0.2,
    ) -> SplitPlan: ...


# --------------------------------------------------------------------------
# Trust plugins
# --------------------------------------------------------------------------


@runtime_checkable
class TrustPlugin(Protocol):
    name: str

    def evaluate(
        self,
        *,
        model: Any,
        task_type: str,
        representation: Any,
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        smiles: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a dict of metric-name -> value."""


@runtime_checkable
class UncertaintyPlugin(Protocol):
    name: str

    def evaluate(self, *, model: Any, X: np.ndarray, task_type: str) -> np.ndarray:
        """Return per-sample uncertainty scores."""


@runtime_checkable
class ExplainabilityPlugin(Protocol):
    name: str

    def explain(
        self, *, model: Any, X: np.ndarray, feature_names: list[str], task_type: str
    ) -> dict[str, Any]:
        """Return {feature_name: importance}."""


# --------------------------------------------------------------------------
# Diagnosis & correction plugins
# --------------------------------------------------------------------------


class FailureDiagnosis(BaseModel):
    failure_type: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str
    confidence: float
    recommended_actions: list[str] = Field(default_factory=list)


@runtime_checkable
class DiagnosisPlugin(Protocol):
    name: str

    def diagnose(self, trust_report: dict[str, Any], experiment: ExperimentRecord) -> (
        list[FailureDiagnosis]
    ): ...


class ProposedIntervention(BaseModel):
    name: str
    description: str
    expected_improvement: float = 0.0  # 0..1 heuristic scale
    expected_trust_gain: float = 0.0  # 0..1 heuristic scale
    compute_cost: float = 1.0  # normalized CPU-seconds
    actions: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


@runtime_checkable
class InterventionPlugin(Protocol):
    name: str

    def propose(
        self, diagnosis: FailureDiagnosis, context: dict[str, Any]
    ) -> list[ProposedIntervention]: ...


# --------------------------------------------------------------------------
# Experimental records
# --------------------------------------------------------------------------


class ExperimentRecord(BaseModel):
    """Complete scientific record of one executed experiment."""

    id: str
    dataset_hash: str
    preprocessing_version: str
    representation: str
    model: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    split: str
    random_seed: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    trust: dict[str, Any] = Field(default_factory=dict)
    runtime_seconds: float = 0.0
    memory_gb: float = 0.0
    llm_decision: str = ""
    rationale: str = ""
    result: str = "completed"  # completed | failed | skipped
    failure_diagnosis: list[dict[str, Any]] = Field(default_factory=list)
    intervention: list[dict[str, Any]] = Field(default_factory=list)
    tags: dict[str, Any] = Field(default_factory=dict)

    @property
    def signature(self) -> str:
        """Uniqueness key used to avoid repeating identical experiments."""
        import hashlib
        import json

        payload = json.dumps(
            {
                "dataset": self.dataset_hash,
                "preprocessing": self.preprocessing_version,
                "rep": self.representation,
                "model": self.model,
                "hp": self.hyperparameters,
                "split": self.split,
                "seed": self.random_seed,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Case & strategy objects
# --------------------------------------------------------------------------


class QSARCase(BaseModel):
    """Structured understanding of the QSAR problem."""

    problem_statement: str = ""
    dataset_size: int = 0
    n_unique_molecules: int = 0
    task_type: str = ""
    endpoint_name: str = ""
    endpoint_confidence: float = 0.0
    endpoint_reasoning: str = ""
    data_issues: list[str] = Field(default_factory=list)
    chemistry_notes: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_validation: list[str] = Field(default_factory=list)
    recommended_representations: list[str] = Field(default_factory=list)
    recommended_models: list[str] = Field(default_factory=list)
    llm_provider: str = ""
    llm_model: str = ""


class ExperimentCandidate(BaseModel):
    """A planned, not-yet-executed experiment."""

    rank: int = 0
    representation: str = ""
    model: str = ""
    validation: str = ""
    hyperparameter_budget: int = 1
    estimated_runtime_seconds: float = 0.0
    estimated_memory_gb: float = 0.0
    expected_improvement: float = 0.0
    expected_information_gain: float = 0.0
    expected_trustworthiness_gain: float = 0.0
    compute_cost: float = 1.0
    utility: float = 0.0
    reason: str = ""
    llm_rationale: str = ""

    def compute_utility(self) -> float:
        denom = max(self.compute_cost, 1e-6)
        value = (
            self.expected_improvement
            + self.expected_information_gain
            + self.expected_trustworthiness_gain
        )
        self.utility = value / denom
        return self.utility