"""Unit tests: budgeted grid hyperparameter search in the experiment runner."""

from __future__ import annotations

import numpy as np

from cta_qsar.core.interfaces import ExperimentCandidate
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.experiments.runner import ExperimentRunner, _fold_primary_score
from cta_qsar.validation.base import make_cv_folds


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.auto_discover()
    return registry


def _runner(hyperparameter_search: bool) -> ExperimentRunner:
    return ExperimentRunner(
        _registry(),
        task_type="regression",
        n_splits=3,
        n_repeats=1,
        random_seed=7,
        hyperparameter_search=hyperparameter_search,
    )


def _candidate(model: str = "elastic_net") -> ExperimentCandidate:
    return ExperimentCandidate(
        representation="morgan",
        model=model,
        validation="random",
        hyperparameter_budget=3,
    )


def _budget() -> BudgetState:
    return BudgetState(max_experiments=4, max_minutes=30, max_memory_gb=8)


def _data(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 * x1 - 0.5 * x2 + rng.normal(scale=0.2, size=n)
    return np.column_stack([x1, x2]), y


def test_hyperparameter_search_returns_best_combo_from_space() -> None:
    runner = _runner(hyperparameter_search=True)
    X, y = _data()
    runner._folds_cache = make_cv_folds(  # noqa: SLF001
        len(y), y, strategy="random", n_splits=3, n_repeats=1,
        random_seed=runner.random_seed, test_fraction=0.2,
    )
    candidate = _candidate()
    runner._folds_cache = make_cv_folds(  # noqa: SLF001
        len(y), y, strategy="random", n_splits=3, n_repeats=1,
        random_seed=runner.random_seed, test_fraction=0.2,
    )
    result = runner._pick_hyperparams(  # noqa: SLF001
        candidate, n_features=X.shape[1], budget=_budget(), X=X, y=y
    )
    assert set(result) == {"alpha", "l1_ratio"}
    assert result["alpha"] in [0.01, 0.1, 1.0]
    assert result["l1_ratio"] in [0.2, 0.5, 0.8]
    registry = _registry()

    def score(params: dict) -> float:
        return _fold_primary_score(
            runner._build_estimator(  # noqa: SLF001
                registry.get("model", "elastic_net"), "elastic_net", X.shape[1], params
            ),
            X, y, runner._folds_cache, "regression",  # noqa: SLF001
        )

    assert score(result) >= score({"alpha": 0.01, "l1_ratio": 0.2})


def test_hyperparameter_search_disabled_returns_first_choice() -> None:
    runner = _runner(hyperparameter_search=False)
    X, y = _data()
    candidate = _candidate()
    result = runner._pick_hyperparams(  # noqa: SLF001
        candidate, n_features=X.shape[1], budget=_budget(), X=X, y=y
    )
    assert result == {"alpha": 0.01, "l1_ratio": 0.2}


def test_hyperparameter_search_skips_graph_inputs() -> None:
    runner = _runner(hyperparameter_search=True)
    candidate = _candidate()
    X: list[object] = [object() for _ in range(10)]
    y = np.arange(10.0)
    runner._folds_cache = make_cv_folds(  # noqa: SLF001
        len(y), y, strategy="random", n_splits=3, n_repeats=1,
        random_seed=runner.random_seed, test_fraction=0.2,
    )
    result = runner._pick_hyperparams(  # noqa: SLF001
        candidate, n_features=4, budget=_budget(), X=X, y=y
    )
    assert result == {"alpha": 0.01, "l1_ratio": 0.2}