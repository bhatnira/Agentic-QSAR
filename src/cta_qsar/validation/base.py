"""Validation plugin base and shared split utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit

from cta_qsar.core.interfaces import SplitPlan

TaskType = str


def _base_split_plan(
    name: str,
    *,
    n_splits: int,
    n_repeats: int,
    test_fraction: float,
    random_seed: int,
    description: str,
) -> SplitPlan:
    return SplitPlan(
        name=name,
        params={
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "test_fraction": test_fraction,
            "random_seed": random_seed,
        },
        description=description,
    )


def make_cv_folds(
    n: int,
    y: pd.Series | np.ndarray,
    *,
    strategy: str,
    n_splits: int,
    n_repeats: int,
    random_seed: int,
    test_fraction: float,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) fold pairs."""
    y = np.asarray(y)
    indices = np.arange(n)

    if strategy == "random":
        folds = []
        for rep in range(n_repeats):
            splitter = ShuffleSplit(
                n_splits=n_splits, test_size=test_fraction, random_state=random_seed + rep * 1000
            )
            folds.extend(splitter.split(indices))
        return folds

    if strategy == "stratified":
        folds = []
        for rep in range(n_repeats):
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=1, random_state=random_seed + rep * 1000
            )
            folds.extend(splitter.split(indices, y))
        return folds

    if strategy == "repeated_cv":
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed
        )
        return list(splitter.split(indices))

    if strategy == "repeated_stratified_cv":
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed
        )
        return list(splitter.split(indices, y))

    if strategy in ("scaffold", "cluster"):
        if groups is None:
            raise ValueError(f"{strategy} split requires molecule groups")
        return _group_folds(groups, n_splits, n_repeats, random_seed)

    if strategy == "temporal":
        if groups is None:
            raise ValueError("temporal split requires a temporal column")
        return _temporal_folds(groups, n_splits)

    raise ValueError(f"unknown split strategy: {strategy}")


def _group_folds(
    groups: np.ndarray, n_splits: int, n_repeats: int, random_seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group/leave-cluster-out folds (scaffold groups or cluster ids).

    Molecules sharing a group stay together; groups are split into n_splits
    folds with random shuffling per repeat, ensuring chemical series
    generalization is tested.
    """

    unique_groups = np.unique(groups)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(random_seed + rep * 1000)
        group_ids = rng.permutation(unique_groups)
        fold_id = np.searchsorted(np.sort(group_ids), groups) % n_splits
        for k in range(n_splits):
            test = np.where(fold_id == k)[0]
            train = np.where(fold_id != k)[0]
            if len(test) > 0:
                folds.append((train, test))
    return folds


def _temporal_folds(values: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(values, kind="stable")
    n = len(order)
    folds = []
    for k in range(1, n_splits):
        cutoff = int(n * k / n_splits)
        train = order[:cutoff]
        test = order[cutoff:int(n * (k + 1) / n_splits)] if k < n_splits - 1 else order[cutoff:]
        if len(test) > 0:
            folds.append((train, test))
    return folds