"""Extensions: multiclass + multitask endpoint/runner/trust plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cta_qsar.endpoints.detector import detect_endpoint
from cta_qsar.experiments.runner import ExperimentRunner, _merge_multitask_trust
from cta_qsar.trust.base import primary_metric
from tests.fixtures.datasets import make_regression, random_molecules


def _registry():
    from cta_qsar.core.registry import get_registry

    registry = get_registry()
    registry.auto_discover()
    return registry


def _make_binary_targets(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    smiles = random_molecules(rng, n)
    logp = [np.log1p(i % 7) for i in range(n)]
    df = pd.DataFrame({
        "smiles": smiles,
        "NR-AR": [int((v + (i % 3)) % 2) for i, v in enumerate(logp[:n])],
        "SR-p53": [int((v + i % 2) % 2) for i, v in enumerate(logp[:n])],
    })
    return df


class TestMultitaskDetection:
    def test_uniform_binary_columns(self):
        df = _make_binary_targets()
        det = detect_endpoint(df, "NR-AR", target_columns=["NR-AR", "SR-p53"])
        assert det.task_type == "multitask_binary"
        assert det.n_targets == 2
        assert det.target_columns == ["NR-AR", "SR-p53"]

    def test_single_column_unaffected(self):
        df = _make_binary_targets()
        det = detect_endpoint(df, "NR-AR")
        assert det.task_type == "binary"

    def test_mixed_tasks_not_detected(self):
        df = _make_binary_targets()
        df["kdr"] = np.linspace(0, 1, len(df))
        det = detect_endpoint(df, "NR-AR", target_columns=["NR-AR", "kdr"])
        assert det.task_type == "binary"


class TestMultitaskRunner:
    def test_multitask_binary_metrics_aggregate(self):
        registry = _registry()
        df = _make_binary_targets()
        targets = ["NR-AR", "SR-p53"]
        runner = ExperimentRunner(
            registry,
            task_type="multitask_binary",
            n_splits=2,
            n_repeats=1,
            test_fraction=0.2,
            random_seed=3,
            hyperparameter_search=False,
        )
        from cta_qsar.core.interfaces import ExperimentCandidate

        candidate = ExperimentCandidate(
            representation="morgan",
            model="random_forest",
            validation="random",
            utility=0.5,
            compute_cost=1.0,
            reason="unit test",
        )
        record = runner.run(
            candidate,
            smiles=df["smiles"].astype(str).tolist(),
            df=df,
            target_column=targets[0],
            budget=_budget(),
            target_columns=targets,
        )
        assert record.result == "completed"
        assert "roc_auc" in record.metrics
        assert 0.0 <= record.metrics["roc_auc"] <= 1.0
        assert record.tags["targets"] == targets
        assert set(record.tags["per_target_primary"]) == set(targets)
        assert record.trust["predictive"]["primary_metric"] == "roc_auc"
        fold_auc = record.trust["predictive"]["roc_auc"]["mean"]
        target_aucs = [record.tags["per_target_primary"][t] for t in targets]
        assert abs(fold_auc - float(np.mean(target_aucs))) < 1e-3

    def test_multitask_regression_metrics(self):
        registry = _registry()
        reg = make_regression(n=200, columns=("smiles", "y1"), seed=11)
        df = reg.rename(columns={"y1": "y1"})
        df["y2"] = df["y1"] * 0.8 + 0.3
        runner = ExperimentRunner(
            registry,
            task_type="multitask_regression",
            n_splits=2,
            n_repeats=1,
            test_fraction=0.2,
            random_seed=3,
            hyperparameter_search=False,
        )
        from cta_qsar.core.interfaces import ExperimentCandidate

        candidate = ExperimentCandidate(
            representation="morgan",
            model="ridge",
            validation="random",
            utility=0.5,
            compute_cost=1.0,
            reason="unit test",
        )
        record = runner.run(
            candidate,
            smiles=df["smiles"].astype(str).tolist(),
            df=df,
            target_column="y1",
            budget=_budget(),
            target_columns=["y1", "y2"],
        )
        assert record.result == "completed"
        assert "rmse" in record.metrics
        assert record.metrics["rmse"] > 0

    def test_merge_multitask_trust(self):
        base = {"predictive": {"mcc": {"mean": 0.5, "std": 0.1}, "primary_metric": "mcc"}}
        other = {"predictive": {"mcc": {"mean": 0.9, "std": 0.2}, "primary_metric": "mcc"}}
        merged = _merge_multitask_trust([base, other])
        assert merged["predictive"]["mcc"]["mean"] == pytest.approx(0.7)
        assert merged["predictive"]["mcc"]["std"] == pytest.approx(0.2)


class TestMulticlassRunner:
    def test_multiclass_metrics_and_n_classes(self):
        registry = _registry()
        rng = np.random.default_rng(9)
        smiles = random_molecules(rng, 300)
        logp = [np.log1p(i % 7) for i in range(len(smiles))]
        y = np.asarray([int((v * 3 + i) % 3) for i, v in enumerate(logp)])
        labels = np.asarray(["Low", "Moderate", "High"])[y]
        df = pd.DataFrame({"smiles": smiles, "class": labels})
        runner = ExperimentRunner(
            registry,
            task_type="multiclass",
            n_splits=2,
            n_repeats=1,
            test_fraction=0.2,
            random_seed=3,
            hyperparameter_search=False,
        )
        from cta_qsar.core.interfaces import ExperimentCandidate

        candidate = ExperimentCandidate(
            representation="morgan",
            model="random_forest",
            validation="random",
            utility=0.5,
            compute_cost=1.0,
            reason="unit test",
        )
        record = runner.run(
            candidate,
            smiles=df["smiles"].astype(str).tolist(),
            df=df,
            target_column="class",
            budget=_budget(),
        )
        assert record.result == "completed"
        assert "mcc" in record.metrics
        assert primary_metric("multiclass") == "mcc"
        assert record.trust["predictive"]["primary_metric"] == "mcc"


def _budget():
    from cta_qsar.experiments.budget import BudgetState

    return BudgetState(max_experiments=1, max_minutes=10.0, max_memory_gb=4.0)