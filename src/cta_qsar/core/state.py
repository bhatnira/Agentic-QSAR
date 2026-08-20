"""LangGraph state definition for the QSAR agent workflow.

The state is a TypedDict; heavy objects (DataFrames, fitted models) are kept on
the state so the cyclic graph can pass them between nodes.  ``experiments``,
``diagnoses`` and ``notes`` grow monotonically; because nodes run strictly
sequentially in this workflow, nodes return the accumulated list on each visit
(plain last-value channels; Annotated reducers double-write on langgraph 1.2.x).
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from cta_qsar.core.interfaces import (
    ExperimentCandidate,
    ExperimentRecord,
    FailureDiagnosis,
    QSARCase,
)

StepDecision = dict[str, Any]


class QSARState(TypedDict, total=False):
    # -- inputs
    data_path: str
    smiles_column: str
    target_column: str
    config: Any
    budget_minutes: int
    max_experiments: int

    # -- loaded data
    raw_df: pd.DataFrame
    profile: dict[str, Any]
    endpoint: dict[str, Any]
    standardized_df: pd.DataFrame
    preprocessing_version: str
    standardization_log: list[dict[str, Any]]
    quality_report: dict[str, Any]
    chemical_space: dict[str, Any]

    # -- case reasoning
    case: QSARCase
    validated_splits: list[str]

    # -- planning
    candidates: list[ExperimentCandidate]
    selected_candidate: ExperimentCandidate
    rejected_candidates: list[ExperimentCandidate]
    knowledge_context: dict[str, Any]

    # -- execution
    experiments: list[ExperimentRecord]
    current_experiment: ExperimentRecord
    failed_experiment: ExperimentRecord

    # -- trust & diagnosis
    trust_reports: list[dict[str, Any]]
    diagnoses: list[FailureDiagnosis]

    # -- budget
    started_at: float
    experiments_remaining: int
    stop_reasons: list[str]

    # -- reporting
    final_report: dict[str, Any]
    run_id: str
    output_dir: str

    # -- routing bookkeeping
    plan_round: int
    notes: list[str]
    error: str | None