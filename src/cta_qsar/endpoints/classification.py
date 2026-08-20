"""Binary and multiclass classification endpoint plugins."""

from __future__ import annotations

from typing import Any

import pandas as pd


class _CountsMixin:
    @staticmethod
    def _counts(column: pd.Series) -> dict[str, Any]:
        counts = column.value_counts()
        total = len(counts)
        return {
            "n_classes": int(total),
            "labels": [str(x) for x in counts.index.tolist()],
            "confidence": min(0.75, 0.55 + 0.05 * total),
        }


class BinaryEndpointPlugin(_CountsMixin):
    name = "binary"
    task_type = "binary"

    def applicable(self, column: pd.Series) -> bool:
        c = self._counts(column)
        return c["n_classes"] == 2

    def detect(self, column: pd.Series, column_name: str) -> dict[str, Any]:
        c = self._counts(column)
        imbalanced = max(column.value_counts()) / max(column.value_counts().min(), 1) >= 10
        return {
            **c,
            "transformation": "label_encoded" if imbalanced else "none",
            "weights": {"balanced": imbalanced},
            "reasoning": (
                f"Two exclusive labels for {column_name!r}; binary classification "
                f"({'imbalanced' if imbalanced else 'roughly balanced'})."
            ),
        }


class MulticlassEndpointPlugin(_CountsMixin):
    name = "multiclass"
    task_type = "multiclass"

    def applicable(self, column: pd.Series) -> bool:
        c = self._counts(column)
        return 3 <= c["n_classes"] <= 12

    def detect(self, column: pd.Series, column_name: str) -> dict[str, Any]:
        c = self._counts(column)
        return {
            **c,
            "transformation": "ordinal_or_onehot",
            "reasoning": (
                f"{c['n_classes']} discrete categories in {column_name!r}; "
                "multiclass classification with ordinal encoding."
            ),
        }

PLUGINS = [BinaryEndpointPlugin, MulticlassEndpointPlugin]
