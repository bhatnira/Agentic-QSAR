"""Unit tests: experiment candidate planning and utility scoring."""

from __future__ import annotations

from cta_qsar.core.interfaces import QSARCase
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.experiments.planner import generate_candidates, pick_first_usable


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.auto_discover()
    return registry


def _case(n: int = 200, task_type: str = "regression") -> QSARCase:
    return QSARCase(
        dataset_size=n,
        n_unique_molecules=n,
        task_type=task_type,
        endpoint_name="pIC50",
        endpoint_confidence=0.6,
    )


def _budget() -> BudgetState:
    return BudgetState(max_experiments=3, max_minutes=30, max_memory_gb=8)


def test_candidates_generated_and_utility_sorted() -> None:
    registry = _registry()
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan", "rdkit_descriptors"],
        enabled_models=["ridge", "random_forest"],
        validated_splits=["random", "scaffold"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=200,
        dataset_props={"task_type": "regression", "n_rows": 200},
        hardware_tier="cpu",
    )
    assert len(candidates) > 0
    utilities = [c.utility for c in candidates]
    assert utilities == sorted(utilities, reverse=True)
    ranks = [c.rank for c in candidates]
    assert ranks == list(range(1, len(candidates) + 1))


def test_morgan_ridge_baseline_first_on_fresh_case() -> None:
    registry = _registry()
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan", "rdkit_descriptors", "maccs"],
        enabled_models=["ridge", "random_forest", "xgboost"],
        validated_splits=["random", "scaffold"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=200,
        dataset_props={"task_type": "regression", "n_rows": 200},
        hardware_tier="cpu",
    )
    top = candidates[0]
    assert top.representation == "morgan"
    assert top.model == "ridge"
    assert top.utility > 0


def test_compute_cost_is_positive_and_recorded() -> None:
    registry = _registry()
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan"],
        enabled_models=["ridge"],
        validated_splits=["random"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=200,
        dataset_props={"task_type": "regression", "n_rows": 200},
        hardware_tier="cpu",
    )
    assert candidates[0].compute_cost > 0
    assert candidates[0].estimated_runtime_seconds > 0
    assert candidates[0].estimated_memory_gb > 0
    assert len(candidates[0].reason) > 0


def test_pick_first_usable_returns_top_candidate() -> None:
    registry = _registry()
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan"],
        enabled_models=["ridge"],
        validated_splits=["random"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=200,
        dataset_props={"task_type": "regression", "n_rows": 200},
        hardware_tier="cpu",
    )
    chosen = pick_first_usable(candidates, _budget(), history=[])
    assert chosen is not None


def test_pick_first_usable_skips_already_done_experiments() -> None:
    registry = _registry()
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan"],
        enabled_models=["ridge"],
        validated_splits=["random"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=200,
        dataset_props={"task_type": "regression", "n_rows": 200},
        hardware_tier="cpu",
    )
    top = candidates[0]
    history = [
        {
            "representation": top.representation,
            "model": top.model,
            "split": top.validation,
            "seed": 42,
            "result": "completed",
        }
    ]
    chosen = pick_first_usable(candidates, _budget(), history=history)
    # the only candidate was already executed -> nothing left to pick
    assert chosen is None or chosen != top


def test_pick_first_usable_returns_none_when_budget_exhausted() -> None:
    budget = _budget()
    budget.max_experiments = 1
    budget.record(1.0)  # one experiment done; cap reached
    chosen = pick_first_usable([], budget, history=[])
    assert chosen is None


def test_small_data_prefers_simple_models() -> None:
    """With <200 rows, nonlinear models must not outrank the cheap baseline."""
    registry = _registry()
    candidates = generate_candidates(
        registry=registry,
        case=_case(n=60),
        enabled_representations=["morgan"],
        enabled_models=["ridge", "random_forest", "xgboost"],
        validated_splits=["random"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=60,
        dataset_props={"task_type": "regression", "n_rows": 60},
        hardware_tier="cpu",
    )
    assert candidates[0].model == "ridge"


def test_novelty_bonus_only_for_untried_components() -> None:
    """After the first experiment, only genuinely untried representations and
    model families may earn the info-gain bonus (regression for the
    once-broken `for h in []` novelty check)."""
    registry = _registry()
    history = [
        {
            "representation": "morgan",
            "model": "ridge",
            "split": "random",
            "seed": 42,
            "result": "completed",
        }
    ]
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan", "maccs"],
        enabled_models=["ridge", "svr"],
        validated_splits=["random"],
        budget=_budget(),
        history=history,
        task_type="regression",
        n_samples=200,
        dataset_props={"task_type": "regression", "n_rows": 200},
        hardware_tier="cpu",
    )
    by_pair = {(c.representation, c.model): c for c in candidates}
    executed = by_pair[("morgan", "ridge")]
    assert "untried representation morgan" not in executed.reason
    assert "untried model family ridge" not in executed.reason
    assert "untried representation maccs" in by_pair[("maccs", "ridge")].reason
    assert "untried model family svr" in by_pair[("morgan", "svr")].reason