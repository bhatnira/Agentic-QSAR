"""Unit tests: validation split plugins."""

from __future__ import annotations

import numpy as np

from cta_qsar.validation.base import make_cv_folds
from cta_qsar.validation.cluster_split import ClusterSplit, cluster_groups
from cta_qsar.validation.random_split import RandomSplit
from cta_qsar.validation.scaffold_split import ScaffoldSplit, scaffold_id
from cta_qsar.validation.stratified import StratifiedSplit
from cta_qsar.validation.temporal_split import TemporalSplit
from tests.fixtures import datasets

N = 200


def _y() -> np.ndarray:
    return np.arange(N) % 2


def test_random_split_folds_cover_all_samples() -> None:
    folds = make_cv_folds(
        N, _y(), strategy="random", n_splits=5, n_repeats=1,
        random_seed=42, test_fraction=0.2,
    )
    assert len(folds) == 5
    for train, test in folds:
        assert len(train) + len(test) == N
        assert len(test) == int(0.2 * N)


def test_scaffold_split_keeps_scaffolds_together() -> None:
    df = datasets.make_scaffold_heavy(n=N)
    smiles = df["SMILES"].tolist()
    groups = np.asarray([scaffold_id(s) or f"none-{i}" for i, s in enumerate(smiles)])
    folds = make_cv_folds(
        N, _y(), strategy="scaffold", n_splits=3, n_repeats=1,
        random_seed=42, test_fraction=0.2, groups=groups,
    )
    assert len(folds) == 3
    for train, test in folds:
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        assert train_groups.isdisjoint(test_groups)


def test_scaffold_id_is_deterministic() -> None:
    assert scaffold_id("c1ccccc1C") == scaffold_id("Cc1ccccc1")
    assert scaffold_id("c1ccccc1") != scaffold_id("c1ccccn1")
    assert scaffold_id("garbage") is None


def test_stratified_folds_preserve_class_balance() -> None:
    y = _y()
    folds = make_cv_folds(
        N, y, strategy="stratified", n_splits=5, n_repeats=1,
        random_seed=42, test_fraction=0.2,
    )
    for _, test in folds:
        frac = y[test].mean()
        assert 0.35 < frac < 0.65


def test_temporal_folds_are_chronological() -> None:
    time = np.arange(N)
    folds = make_cv_folds(
        N, _y(), strategy="temporal", n_splits=4, n_repeats=1,
        random_seed=42, test_fraction=0.2, groups=time,
    )
    for train, test in folds:
        assert time[train].max() <= time[test].min()


def test_cluster_butina_reuses_groups() -> None:
    df = datasets.make_scaffold_heavy(n=90)
    groups = cluster_groups(df["SMILES"].tolist())
    assert len(set(groups.tolist())) < len(groups)


def test_plugin_applicability_gates() -> None:
    props_reg = {"task_type": "regression", "n_rows": 200, "has_temporal_column": False}
    props_bin = {"task_type": "binary", "n_rows": 200, "has_temporal_column": False}
    props_temp = {"task_type": "regression", "n_rows": 200, "has_temporal_column": True}
    assert RandomSplit().applicability(props_reg) == (
        True,
        "generic random split; cheap baseline",
    )
    assert StratifiedSplit().applicability(props_reg)[0] is False  # regression
    assert StratifiedSplit().applicability(props_bin)[0] is True  # classification
    assert ScaffoldSplit().applicability(props_reg)[0] is True  # always applicable
    assert ClusterSplit().applicability(props_reg)[0] is True
    assert TemporalSplit().applicability(props_temp)[0] is True
    assert TemporalSplit().applicability(props_reg)[0] is False


def test_random_split_plugin_split_method() -> None:
    df = datasets.make_regression(n=100)
    plan = RandomSplit().split(
        df, df["pIC50"], task_type="regression", n_splits=5, random_seed=42
    )
    assert plan.name == "random"
    assert plan.params["n_splits"] == 5