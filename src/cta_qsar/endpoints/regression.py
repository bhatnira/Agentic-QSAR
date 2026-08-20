"""Regression endpoint plugin."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cta_qsar.endpoints.detector import _CLASSIC_LINEAR, _guess_units, _looks_log_transformed


class RegressionEndpointPlugin:
    name: str = "regression"
    task_type: str = "regression"

    def applicable(self, column: pd.Series) -> bool:
        vals = pd.to_numeric(column, errors="coerce").dropna()
        if len(vals) < 20:
            return False
        return vals.nunique() > 12

    def detect(self, column: pd.Series, column_name: str) -> dict[str, Any]:
        name_lower = str(column_name).lower()
        affinity = any(k in name_lower for k in _CLASSIC_LINEAR)
        transformed = _looks_log_transformed(name_lower, column)
        return {
            "units": _guess_units(name_lower) if affinity else None,
            "transformation": "log-transformed" if transformed else "untouched",
            "confidence": 0.7,
            "reasoning": (
                f"Continuous endpoint {column_name!r} with many unique numeric values; "
                "regression assumed."
            ),
        }

class RegressionPlugin(RegressionEndpointPlugin):
    """Alias with stable name used by the registry."""

    name = "regression_plugin"


PLUGINS = [RegressionEndpointPlugin]
