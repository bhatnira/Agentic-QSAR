"""Random split plugin."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cta_qsar.core.interfaces import SplitPlan
from cta_qsar.validation.base import _base_split_plan, make_cv_folds


class RandomSplit:
    name = "random"

    def applicability(self, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "generic random split; cheap baseline"

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
            "random",
            n_splits=n_splits,
            n_repeats=n_repeats,
            test_fraction=test_fraction,
            random_seed=random_seed,
            description="Shuffled random train/test splits (repeat = seed variation)",
        )

    def folds(
        self, n: int, y: pd.Series, *, n_splits: int, n_repeats: int, random_seed: int, test_fraction: float
    ) -> list[tuple[Any, Any]]:
        return make_cv_folds(
            n, y, strategy="random", n_splits=n_splits, n_repeats=n_repeats,
            random_seed=random_seed, test_fraction=test_fraction,
        )

PLUGINS = [RandomSplit]
