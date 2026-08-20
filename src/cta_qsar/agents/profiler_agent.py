"""Profiler agent: dataset profiling + endpoint detection + case reasoning."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cta_qsar.chemistry.chemical_space import summarize_chemical_space
from cta_qsar.chemistry.validation import quality_report
from cta_qsar.core.interfaces import QSARCase
from cta_qsar.core.logging import get_logger
from cta_qsar.endpoints.detector import EndpointDetector
from cta_qsar.llm.base import ReasoningModel

logger = get_logger(__name__)


def profile_dataset(
    df: pd.DataFrame,
    *,
    smiles_column: str,
    target_column: str | None,
    llm: ReasoningModel | None = None,
    detected_endpoint: dict[str, Any] | None = None,
    temporal_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Profile a dataframe into structured scientific context."""
    has_temporal = bool(temporal_columns)
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "n_numeric_columns": int(df.select_dtypes("number").shape[1]),
        "n_categorical_columns": int(df.select_dtypes(include="object").shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_value_summary": {
            col: int(n)
            for col, n in (df.isna().sum()).items()
            if n > 0
        },
        "has_temporal_column": has_temporal,
        "temporal_columns": temporal_columns or [],
    }
    if target_column and target_column in df.columns and detected_endpoint:
        profile["endpoint"] = detected_endpoint
        profile["task_type"] = detected_endpoint.get("task_type", "regression")
        qa = quality_report(
            df,
            smiles_column=smiles_column,
            target_column=target_column,
            task_type=profile["task_type"],
            endpoint=detected_endpoint,
        )
        profile["quality_report"] = qa
    try:
        profile["chemical_space"] = summarize_chemical_space(df, smiles_column)
    except Exception as exc:  # noqa: BLE001
        profile["chemical_space"] = {"error": str(exc)}
    if llm is not None:
        try:
            classification = llm.classify_case(
                {
                    "profile": profile,
                    "case": profile.get("case", {}),
                }
            )
            profile["llm_case_classification"] = classification.model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM case classification failed (%s); continuing heuristically", exc)
            profile["llm_case_classification"] = {"error": str(exc)}
    return profile


class ProfilerAgent:
    """Exposes the profiling workflow to the orchestrator."""

    def __init__(self, llm: ReasoningModel | None = None) -> None:
        self.llm = llm
        self.detector = EndpointDetector()

    def run(self, df: pd.DataFrame, smiles_column: str, target_column: str | None) -> dict[str, Any]:
        return profile_dataset(
            df,
            smiles_column=smiles_column,
            target_column=target_column,
            llm=self.llm,
        )


def build_case(profile: dict[str, Any], run_config: dict[str, Any]) -> QSARCase:
    """Assemble the structured QSARCase from a profile."""
    endpoint = profile.get("endpoint", {})
    quality = profile.get("quality_report", {})
    case = QSARCase(
        dataset_size=profile.get("n_rows", 0),
        n_unique_molecules=profile.get("chemical_space", {}).get("n_unique_molecules", 0),
        task_type=endpoint.get("task_type", "regression"),
        endpoint_name=endpoint.get("endpoint_name", ""),
        endpoint_confidence=endpoint.get("confidence", 0.0),
        endpoint_reasoning=endpoint.get("reasoning", ""),
        llm_provider=run_config.get("provider", "mock"),
        llm_model=run_config.get("model", "heuristic"),
    )
    risks: list[str] = []
    if quality.get("conflicting_labels", {}).get("n_conflicting_groups", 0):
        risks.append("conflicting labels across duplicate molecules")
    if quality.get("duplicate_molecules", {}).get("n_duplicates", 0):
        risks.append("duplicate molecules present")
    if quality.get("class_balance", {}).get("imbalance_ratio", 1) > 5:
        risks.append("strong class imbalance")
    if quality.get("outliers", {}).get("n_extreme", 0):
        risks.append("extreme target values flagged (not removed)")
    if endpoint.get("ambiguous"):
        risks.append("endpoint type uncertain; ask user or use conservative assumptions")
    if profile.get("n_rows", 0) < 200:
        risks.append("small dataset; prefer simple models and strict validation")
    case.risks = risks
    case.data_issues = []
    if quality.get("invalid_smiles", 0):
        case.data_issues.append(f"{quality['invalid_smiles']} invalid SMILES rejected")
    if not endpoint.get("column_found", True):
        case.data_issues.append("target column missing")
    return case