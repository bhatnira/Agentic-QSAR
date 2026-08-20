"""Endpoint detection.

Detects whether a target column is regression / binary / multiclass /
multitask, and infers units/transformations from evidence --- never from the
column name alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from cta_qsar.core.interfaces import EndpointPlugin


class EndpointDetection(BaseModel):
    task_type: str
    endpoint_name: str
    units_if_known: str | None = None
    transformation_status: str = "untouched"
    confidence: float = 0.5
    reasoning: str = ""
    column_found: bool = True
    aggregation_note: str | None = None
    ambiguous: bool = False
    ask_for: list[str] = Field(default_factory=list)

    n_classes: int = 0
    class_labels: list[Any] = Field(default_factory=list)
    n_targets: int = 1
    target_columns: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


_CLASSIC_NEGLOGS = ("pic50", "pec50", "ppa", "plogp", "plogd")
_CLASSIC_LINEAR = (
    "ic50", "ec50", "gi50", "ki", "kd", "kcat", "logp", "logs", "logd",
    "mw", "tpsa", "ic 50", "ec 50",
)


def _looks_log_transformed(name_lower: str, values: pd.Series) -> bool:
    """Evidence-based check; names hint but are never conclusive."""
    neg = any(k in name_lower for k in _CLASSIC_NEGLOGS)
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 5:
        return neg
    positive = np.abs(vals[vals != 0])
    if len(positive) < 5:
        return neg
    span = float(np.log10(positive).max() - np.log10(positive).min())
    return neg or span > 2.5


def _all_columns_present(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns)


def _detect_multitask_columns(
    df: pd.DataFrame, columns: list[str]
) -> EndpointDetection | None:
    """Multiple aligned target columns -> multitask endpoint (uniform task)."""
    tasks: list[str] = []
    for col in columns:
        vals = df[col].dropna()
        numeric = pd.to_numeric(vals, errors="coerce")
        n_parsed = int(numeric.notna().sum())
        n_unique = int(numeric.nunique()) if len(numeric) else 0
        if n_parsed == 0:
            labels = sorted(pd.unique(vals).tolist())
            n_unique = len(labels)
        task = (
            "unknown"
            if n_unique == 0
            else "binary"
            if n_unique == 2
            else "multiclass"
            if n_unique <= 12
            else "regression"
        )
        tasks.append(task)
    uniq = {t for t in tasks if t != "unknown"}
    if not uniq or len(uniq) > 1:
        return None
    base = next(iter(uniq))
    if base == "multiclass":
        return None
    return EndpointDetection(
        task_type=f"multitask_{base}",
        endpoint_name="+".join(columns),
        confidence=0.6,
        reasoning=(
            f"{len(columns)} aligned target columns detected as uniform "
            f"{base} tasks; treated as multitask {base}."
        ),
        n_targets=len(columns),
        target_columns=list(columns),
    )


def _has_multitask_shaped_targets(df: pd.DataFrame, target_column: str) -> bool:
    values = df[target_column].dropna()
    if len(values) == 0:
        return False
    shaped_ok = values.apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).mean() > 0.8
    return bool(shaped_ok)


def _guess_units(name_lower: str) -> str:
    for token, unit in (
        ("pic50", "μM^-1 (pIC50)"), ("pec50", "μM^-1 (pEC50)"),
        ("ic50", "μM"), ("ec50", "μM"), ("gi50", "μM"),
        ("hiv", "pM"), ("ki", "μM"), ("kd", "μM"),
    ):
        if token in name_lower:
            return unit
    return "unknown"


def build_endpoint_plugins() -> list[EndpointPlugin]:
    """Return the default endpoint plugins."""
    from cta_qsar.endpoints.classification import BinaryEndpointPlugin, MulticlassEndpointPlugin
    from cta_qsar.endpoints.multitask import MultitaskEndpointPlugin
    from cta_qsar.endpoints.regression import RegressionEndpointPlugin

    return [
        MultitaskEndpointPlugin(),
        RegressionEndpointPlugin(),
        BinaryEndpointPlugin(),
        MulticlassEndpointPlugin(),
    ]


def detect_endpoint(
    df: pd.DataFrame,
    target_column: str,
    *,
    known_task_type: str | None = None,
    target_columns: list[str] | None = None,
    user_units: str | None = None,
    plugins: list[EndpointPlugin] | None = None,
) -> EndpointDetection:
    """Classify the endpoint for a target column."""
    if target_column not in df.columns:
        return EndpointDetection(
            task_type="unknown",
            endpoint_name=target_column,
            confidence=0.0,
            reasoning=f"Target column {target_column!r} not found in dataset.",
            column_found=False,
            ambiguous=True,
        )

    columns = target_columns or [target_column]
    if len(columns) > 1 and _all_columns_present(df, columns):
        detection = _detect_multitask_columns(df, columns)
        if detection is not None:
            return detection

    column = df[target_column]
    name = str(target_column)

    if _has_multitask_shaped_targets(df, target_column):
        n_targets = int(len(column.dropna().iloc[0])) if column.notna().any() else 0
        return EndpointDetection(
            task_type="multitask_regression",
            endpoint_name=name,
            confidence=0.6,
            reasoning="Target cells contain list-like values; treated as multitask.",
            n_targets=n_targets,
            target_columns=target_columns or [target_column],
        )

    if known_task_type is not None:
        return _classify_values(column, name, known_task_type, user_units)

    for plugin in plugins or []:
        if plugin.applicable(column):
            info = plugin.detect(column, name)
            return EndpointDetection(
                task_type=plugin.task_type,
                endpoint_name=name,
                units_if_known=info.get("units"),
                transformation_status=info.get("transformation", "untouched"),
                confidence=float(info.get("confidence", 0.5)),
                reasoning=str(info.get("reasoning", "")),
                n_classes=int(info.get("n_classes", 0)),
                class_labels=info.get("labels", []),
            )

    return _classify_values(column, name, None, user_units)


def _classify_values(
    column: pd.Series,
    name: str,
    known_task_type: str | None,
    user_units: str | None,
) -> EndpointDetection:
    name_lower = name.lower()
    vals = pd.to_numeric(column, errors="coerce")
    n_parsed = int(vals.notna().sum())
    n_total = int(column.notna().sum())

    affinity_like = bool(user_units) or any(k in name_lower for k in _CLASSIC_LINEAR)
    if affinity_like:
        units = user_units or _guess_units(name_lower)
        transformed = _looks_log_transformed(name_lower, column)
        task = known_task_type or "regression"
        return EndpointDetection(
            task_type=task,
            endpoint_name=name,
            units_if_known=units,
            transformation_status="log-transformed" if transformed else "untouched",
            confidence=0.55,
            reasoning=(
                f"Affinity/physchem-like endpoint {name!r}; numerical parsing rate "
                f"{n_parsed}/{n_total}."
            ),
        )

    numeric = vals.dropna()
    n_unique = numeric.nunique()
    if n_parsed == 0 or len(numeric) == 0:
        labels = sorted(pd.unique(column.dropna()).tolist())
        n_classes = len(labels)
        task = "binary" if n_classes == 2 else ("multiclass" if 2 < n_classes <= 12 else "unknown")
        return EndpointDetection(
            task_type=task,
            endpoint_name=name,
            confidence=0.6 if n_classes in (2, 3, 4, 5) else 0.4,
            reasoning=(
                f"Non-numeric values with {n_classes} unique labels; inferred {task}."
                if n_classes > 0
                else "Target column is empty or fully missing."
            ),
            n_classes=n_classes,
            class_labels=labels,
            ambiguous=n_classes == 0 or n_classes > 12,
            ask_for=["task type confirmation"] if n_classes > 12 or n_classes == 0 else [],
        )

    if n_unique <= 12:
        task = "multiclass" if n_unique > 2 else "binary"
        return EndpointDetection(
            task_type=task,
            endpoint_name=name,
            confidence=0.7 if n_unique <= 6 else 0.5,
            reasoning=(
                f"Column parses as numbers with only {n_unique} unique values; "
                f"treated as {task} classification via label encoding."
            ),
            n_classes=int(n_unique),
            class_labels=sorted(numeric.unique().tolist()),
        )

    transformed = _looks_log_transformed(name_lower, column)
    return EndpointDetection(
        task_type=known_task_type or "regression",
        endpoint_name=name,
        units_if_known=user_units,
        transformation_status="log-transformed" if transformed else "untouched",
        confidence=0.65,
        reasoning=(
            f"Continuous numeric endpoint with {len(numeric)} values and "
            f"{n_unique} unique points; inferred regression."
        ),
    )


class EndpointDetector:
    """Facade over :func:`detect_endpoint` with plugin support."""

    def __init__(self, plugins: list[EndpointPlugin] | None = None) -> None:
        self.plugins = plugins if plugins is not None else build_endpoint_plugins()

    def detect(
        self,
        df: pd.DataFrame,
        target_column: str,
        *,
        known_task_type: str | None = None,
        target_columns: list[str] | None = None,
        user_units: str | None = None,
    ) -> EndpointDetection:
        return detect_endpoint(
            df,
            target_column,
            known_task_type=known_task_type,
            target_columns=target_columns,
            user_units=user_units,
            plugins=self.plugins,
        )