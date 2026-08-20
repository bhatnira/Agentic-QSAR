"""Integration tests: failure handling, routing, and self-correction loop."""

from __future__ import annotations

import pytest

from cta_qsar.core.interfaces import ExperimentCandidate, QSARCase
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.orchestration.routing import (
    route_from_diagnosis,
    route_from_execute,
    route_from_trust,
    route_next,
)


class CrashingModel:
    """A model plugin that fails at build time (simulates a broken backend)."""

    name = "crashing_model"
    supports = ("regression",)

    def applicability(self, task_type: str, representation_name: str) -> tuple[bool, str]:
        return (True, "always applicable")

    def estimate_cost(self, n_samples: int, n_features: int, representation_name: str) -> object:
        from cta_qsar.core.interfaces import CostEstimate

        return CostEstimate(runtime_seconds=1.0, memory_gb=0.1)

    def build_estimator(
        self, task_type: str, n_classes: int | None = None, **hyperparams: object
    ) -> object:
        raise RuntimeError("simulated model backend failure")

    def hyperparameter_space(self) -> dict:
        return {}


def _case() -> QSARCase:
    return QSARCase(dataset_size=100, n_unique_molecules=100, task_type="regression")


def _candidate() -> ExperimentCandidate:
    return ExperimentCandidate(
        representation="morgan",
        model="ridge",
        validation="random",
        compute_cost=1.0,
        utility=1.0,
    )


def _budget() -> BudgetState:
    return BudgetState(max_experiments=5, max_minutes=30, max_memory_gb=8)


def test_execute_failure_is_recorded_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed experiment must produce a failed record and route back to planning."""
    registry = PluginRegistry()
    registry.auto_discover()
    registry.register("model", CrashingModel())
    monkeypatch.setattr(
        "cta_qsar.orchestration.nodes.get_context",
        lambda: type("Ctx", (), {"registry": registry, "llm": None, "config": None})(),
    )

    from cta_qsar.orchestration.nodes import execute_experiment

    state = {
        "selected_candidate": _candidate().model_dump(),
        "smiles_column": "SMILES",
        "target_column": "y",
        "raw_df": None,
        "standardized_df": None,
        "config": type("Cfg", (), {"experiment": type("E", (), {"n_splits": 2, "n_repeats": 1, "test_fraction": 0.2, "random_seed": 42})(), "compute": type("C", (), {"max_experiments": 5, "max_minutes": 30, "max_memory_gb": 8})()})(),
        "endpoint": {"task_type": "regression"},
        "experiments": [],
    }
    # swap the crashing model for ridge inside the candidate
    state["selected_candidate"] = {
        **_candidate().model_dump(),
        "model": "crashing_model",
    }

    import pandas as pd

    df = pd.DataFrame({"SMILES": ["CCO"] * 20, "y": list(range(20))})
    state["standardized_df"] = df

    # direct call: the node catches exceptions from the runner and records failure
    result = execute_experiment(state)
    assert result.get("failed_experiment") is not None
    assert result["failed_experiment"]["result"] == "failed"
    assert "simulated model backend failure" in result["failed_experiment"]["tags"]["error"]
    # loop must continue, not crash
    assert route_from_execute(result) == "plan_experiment"


def test_route_from_execute_evaluates_when_completed() -> None:
    state = {"failed_experiment": None, "error": None}
    assert route_from_execute(state) == "evaluate_performance"


def test_route_from_trust_goes_to_diagnosis() -> None:
    assert route_from_trust({"current_experiment": {"id": "x"}}) == "diagnose_failure"
    assert route_from_trust({}) == "finalize_report"


def test_route_from_diagnosis_always_proposes() -> None:
    assert route_from_diagnosis({}) == "propose_intervention"


def test_route_next_plan_or_finalize() -> None:
    assert route_next({"experiments": [], "stop_reasons": [], "budget_state": {"max_experiments": 5}}) == "plan_experiment"
    assert route_next({"experiments": [1], "stop_reasons": ["budget exhausted"], "budget_state": {"max_experiments": 5}}) == "finalize_report"


def _routing_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "config": type(
            "Cfg",
            (),
            {"compute": type("C", (), {"max_experiments": 5, "max_minutes": 30})()},
        )(),
        "experiments": [],
        "candidates": [],
        "plan_round": 1,
        "budget_state": {
            "max_experiments": 5,
            "max_minutes": 30,
            "elapsed_minutes": 0.0,
        },
    }
    state.update(overrides)
    return state


class _StubStop:
    def __init__(self, should_stop: bool, reason: str = "continue") -> None:
        self.should_stop = should_stop
        self.reason = reason


class _StubLLM:
    provider_name = "mock"
    model = "heuristic"

    def __init__(self, should_stop: bool = False, calls: list[dict] | None = None) -> None:
        self._should_stop = should_stop
        self.calls = calls if calls is not None else []

    def decide_stop(self, context: dict) -> _StubStop:
        self.calls.append(context)
        return _StubStop(self._should_stop, "LLM says enough" if self._should_stop else "continue")


def test_decide_next_action_continues_and_consults_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from cta_qsar.orchestration.nodes import decide_next_action

    llm = _StubLLM(should_stop=False)
    monkeypatch.setattr(
        "cta_qsar.orchestration.nodes.get_context",
        lambda: type("Ctx", (), {"llm": llm})(),
    )
    decision = decide_next_action(_routing_state())
    assert decision == "plan_experiment"
    assert llm.calls, "the LLM stopping assessment must be consulted"
    assert llm.calls[0]["experiments_remaining"] == 5
    assert llm.calls[0]["no_improvement_rounds"] == 0


def test_decide_next_action_finalizes_on_llm_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from cta_qsar.orchestration.nodes import decide_next_action

    llm = _StubLLM(should_stop=True)
    monkeypatch.setattr(
        "cta_qsar.orchestration.nodes.get_context",
        lambda: type("Ctx", (), {"llm": llm})(),
    )
    # veto only applies once at least two experiments exist (else a single
    # weak first round could never be compared against anything)
    state = _routing_state(
        experiments=[{"result": "completed", "metrics": {}}, {"result": "completed", "metrics": {}}]
    )
    decision = decide_next_action(state)
    assert decision == "finalize_report"


def test_decide_next_action_ignores_llm_stop_with_one_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    from cta_qsar.orchestration.nodes import decide_next_action

    llm = _StubLLM(should_stop=True)
    monkeypatch.setattr(
        "cta_qsar.orchestration.nodes.get_context",
        lambda: type("Ctx", (), {"llm": llm})(),
    )
    state = _routing_state(
        experiments=[{"result": "completed", "metrics": {}}],
        candidates=[{"utility": 1.0}, {"utility": 1.0}],
    )
    assert decide_next_action(state) == "plan_experiment"


def test_decide_next_action_survives_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cta_qsar.orchestration.nodes import decide_next_action

    class BrokenLLM(_StubLLM):
        def decide_stop(self, context: dict) -> _StubStop:
            raise RuntimeError("simulated LLM outage")

    monkeypatch.setattr(
        "cta_qsar.orchestration.nodes.get_context",
        lambda: type("Ctx", (), {"llm": BrokenLLM()})(),
    )
    decision = decide_next_action(_routing_state())
    assert decision == "plan_experiment"  # degrades to heuristic policy


def test_unavailable_plugin_raises_clear_error() -> None:
    from cta_qsar.core.exceptions import PluginError

    registry = PluginRegistry()
    with pytest.raises(PluginError):
        registry.get("model", "not_registered")


def test_generate_candidates_skip_unavailable_representations() -> None:
    from cta_qsar.experiments.planner import generate_candidates

    registry = PluginRegistry()
    registry.auto_discover()
    candidates = generate_candidates(
        registry=registry,
        case=_case(),
        enabled_representations=["morgan", "does_not_exist"],
        enabled_models=["ridge", "missing_model"],
        validated_splits=["random"],
        budget=_budget(),
        history=[],
        task_type="regression",
        n_samples=100,
        dataset_props={"task_type": "regression", "n_rows": 100},
        hardware_tier="cpu",
    )
    names = {(c.representation, c.model) for c in candidates}
    assert ("morgan", "ridge") in names
    assert all(rep != "does_not_exist" for rep, _ in names)