"""Multitask endpoint plugin (list-valued target cells)."""

from __future__ import annotations

from typing import Any

import pandas as pd


class MultitaskEndpointPlugin:
    name = "multitask"
    task_type = "multitask_regression"

    def applicable(self, column: pd.Series) -> bool:
        values = column.dropna()
        if len(values) == 0:
            return False
        return values.apply(lambda v: isinstance(v, (list, tuple))).mean() > 0.5

    def detect(self, column: pd.Series, column_name: str) -> dict[str, Any]:
        values = column.dropna()
        n_targets = len(values.iloc[0]) if values.notna().any() else 0
        numeric = values.apply(
            lambda v: all(isinstance(x, (int, float)) for x in v) if isinstance(v, (list, tuple)) else False
        ).mean()
        task = "multitask_regression" if numeric > 0.8 else "multitask_classification"
        return {
            "n_targets": int(n_targets),
            "task": task,
            "confidence": 0.6,
            "reasoning": (
                f"List-valued target cells with ~{int(numeric * 100)}% numeric entries; "
                f"inferred {task}."
            ),
        }