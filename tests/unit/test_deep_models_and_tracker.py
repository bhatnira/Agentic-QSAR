"""Unit tests: tracker, GAT, MPNN, and GNN experiment integration."""

from __future__ import annotations

import numpy as np
import pytest

from cta_qsar.core.interfaces import ExperimentCandidate, ExperimentRecord
from cta_qsar.core.registry import get_registry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.experiments.runner import ExperimentRunner
from cta_qsar.experiments.tracker import ExperimentTracker

try:
    import torch  # noqa: F401

    TORCH_OK = True
except ImportError:
    TORCH_OK = False


@pytest.fixture(scope="module")
def registry():
    registry = get_registry()
    registry.auto_discover()
    return registry


# -- deep model plugins ----------------------------------------------------


@pytest.mark.skipif(not TORCH_OK, reason="torch not installed")
@pytest.mark.parametrize("model_name", ["gcn", "gat", "mpnn"])
def test_graph_gnn_plugins_available(registry, model_name) -> None:
    plugin = registry.get_or_none("model", model_name)
    assert plugin is not None
    applicable, _ = plugin.applicability("regression", "graph")
    assert applicable
    not_applicable, _ = plugin.applicability("regression", "morgan")
    assert not not_applicable


@pytest.mark.skipif(not TORCH_OK, reason="torch not installed")
@pytest.mark.parametrize("model_name", ["gcn", "gat", "mpnn"])
def test_gnn_estimator_fit_predict(registry, model_name) -> None:
    from cta_qsar.representations.graph.featurizer import featurize

    smiles = ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1", "C1CCCCC1"] * 3
    X = featurize(smiles)
    y = np.arange(len(smiles), dtype=float)
    plugin = registry.get_or_none("model", model_name)
    estimator = plugin.build_estimator("regression", epochs=2, batch_size=4)
    estimator.fit(X, y)
    pred = estimator.predict(X)
    assert pred.shape == (len(smiles),)


@pytest.mark.skipif(not TORCH_OK, reason="torch not installed")
def test_gnn_end_to_end_experiment(registry, tmp_path) -> None:
    """A real graph experiment runs through the ExperimentRunner on CPU."""
    import pandas as pd

    smiles = ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1", "C1CCCCC1"] * 4
    df = pd.DataFrame({"SMILES": smiles, "activity": [1.0, 1.2, 0.8, 2.1, 3.0, 0.9] * 4})
    df["standardized_smiles"] = df["SMILES"]
    df["smiles_valid"] = True
    df["target_column"] = df["activity"]
    candidate = ExperimentCandidate(
        representation="graph", model="gcn", validation="random",
        utility=1.0, compute_cost=1.0, expected_improvement=0.5,
        expected_information_gain=0.2, expected_trustworthiness_gain=0.1,
    )
    runner = ExperimentRunner(
        registry, task_type="regression", n_splits=2, n_repeats=1,
        dataset_hash="abc", preprocessing_version="test",
    )
    record = runner.run(
        candidate, smiles=smiles, df=df, target_column="target_column",
        budget=BudgetState(max_experiments=3, max_minutes=30, max_memory_gb=8),
    )
    assert record.result == "completed"
    assert record.model in ("gcn", "gat", "mpnn")


def test_classification_with_string_labels_keeps_rows(registry) -> None:
    """String class labels must be label-encoded, never float-coerced to NaN
    (regression for the silent whole-dataset drop)."""
    import pandas as pd

    smiles = ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1", "C1CCCCC1"] * 4
    labels = ["active", "inactive"] * 11 + [None, ""]
    df = pd.DataFrame({"SMILES": smiles[:24], "activity": labels})
    df["standardized_smiles"] = df["SMILES"]
    df["smiles_valid"] = True
    df["target_column"] = df["activity"]
    candidate = ExperimentCandidate(
        representation="morgan", model="extra_trees", validation="random",
        utility=1.0, compute_cost=1.0, expected_improvement=0.5,
        expected_information_gain=0.2, expected_trustworthiness_gain=0.1,
    )
    runner = ExperimentRunner(
        registry, task_type="binary", n_splits=2, n_repeats=1,
        dataset_hash="abc", preprocessing_version="test",
    )
    record = runner.run(
        candidate, smiles=smiles[:24], df=df, target_column="target_column",
        budget=BudgetState(max_experiments=3, max_minutes=30, max_memory_gb=8),
    )
    assert record.result == "completed"
    assert record.tags["dropped_rows"]["missing_target"] == 2
    assert record.metrics, "metrics must be produced from encoded labels"


# -- experiment tracker ----------------------------------------------------


def _record(signature_hint: str, *, result: str = "completed") -> ExperimentRecord:
    return ExperimentRecord(
        id=signature_hint,
        dataset_hash="d",
        preprocessing_version="v1",
        representation="morgan",
        model="ridge",
        hyperparameters={"alpha": 1.0},
        split="random",
        random_seed=42,
        metrics={"rmse": 0.5},
        trust={"predictive": {"rmse": {"mean": 0.5}}},
        runtime_seconds=1.0,
        result=result,
    )


def test_tracker_records_and_dedups(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path, max_experiments=3)
    record = _record("e1")
    tracker.add(record)
    assert tracker.signature_seen(record.signature)
    assert tracker.budget.experiments_done == 1
    assert len(tracker.all_as_dicts()) == 1


def test_tracker_failed_does_not_charge_budget(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path, max_experiments=3)
    tracker.add(_record("e1", result="failed"))
    assert tracker.budget.experiments_done == 0
    assert len(tracker.failed()) == 1


def test_tracker_can_run_more_respects_budget(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path, max_experiments=1)
    assert tracker.can_run_more()
    tracker.add(_record("e1"))
    assert not tracker.can_run_more()
    assert "experiment budget exhausted" in tracker.stop_reasons()


def test_tracker_persists_and_loads(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path, max_experiments=3)
    tracker.add(_record("e1"))
    tracker.add(_record("e2"))
    loaded = ExperimentTracker.load(tmp_path)
    assert len(loaded.all_as_dicts()) == 2
    assert loaded.budget.experiments_done == 2


def test_tracker_best_uses_metric_priority(tmp_path) -> None:
    tracker = ExperimentTracker(tmp_path)
    good = _record("good")
    good.metrics["rmse"] = 0.2
    bad = _record("bad")
    bad.metrics["rmse"] = 0.9
    tracker.add(bad)
    tracker.add(good)
    best = tracker.best()
    assert best is not None
    assert best.id == "good"