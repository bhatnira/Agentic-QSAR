"""Temporal split plugin (only when a temporal column exists)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cta_qsar.core.interfaces import SplitPlan
from cta_qsar.validation.base import _base_split_plan


class TemporalSplit:
    name = "temporal"

    def applicability(self, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        if dataset_props.get("has_temporal_column"):
            return True, "chronological splits test prospective performance"
        return False, "no temporal metadata in the dataset"

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
    ) -> SplitPlan:
        return _base_split_plan(
            "temporal",
            n_splits=n_splits,
            n_repeats=1,
            test_fraction=test_fraction,
            random_seed=random_seed,
            description="Chronological train/test folds (no future leakage)",
        )

PLUGINS = [TemporalSplit]
