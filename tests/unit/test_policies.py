"""Unit tests: orchestration policies (stopping rules, next-action routing)."""

from __future__ import annotations

from cta_qsar.orchestration.policies import decide_next, evaluate_stopping


def _exp(metrics: dict) -> dict:
    return {"metrics": metrics, "result": "completed", "representation": "morgan", "model": "ridge"}


def test_no_stop_reason_when_budget_healthy() -> None:
    reasons = evaluate_stopping(
        budget={"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=1,
        experiments=[],
        no_improvement_rounds=0,
    )
    assert reasons == []


def test_experiment_budget_exhausted() -> None:
    reasons = evaluate_stopping(
        budget={"max_experiments": 2, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=1,
        experiments=[_exp({"rmse": 0.5}), _exp({"rmse": 0.4})],
        no_improvement_rounds=0,
    )
    assert "experiment budget exhausted" in reasons


def test_time_budget_exhausted() -> None:
    reasons = evaluate_stopping(
        budget={"max_experiments": 12, "max_minutes": 1.0, "elapsed_minutes": 2.0},
        plan_round=1,
        experiments=[],
        no_improvement_rounds=0,
    )
    assert "compute time budget exhausted" in reasons


def test_no_improvement_triggers_stop() -> None:
    experiments = [
        _exp({"rmse": 0.50}),
        _exp({"rmse": 0.49}),
        _exp({"rmse": 0.49}),
    ]
    reasons = evaluate_stopping(
        budget={"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=3,
        experiments=experiments,
        no_improvement_rounds=2,
    )
    assert "no meaningful performance improvement over recent experiments" in reasons


def test_negligible_improvement_triggers_stop() -> None:
    experiments = [_exp({"rmse": 0.5000}), _exp({"rmse": 0.4995})]
    reasons = evaluate_stopping(
        budget={"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=2,
        experiments=experiments,
        no_improvement_rounds=0,
    )
    assert any("negligible" in r for r in reasons)


def test_round_cap_after_plan_rounds() -> None:
    reasons = evaluate_stopping(
        budget={"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=4,
        experiments=[],
        no_improvement_rounds=0,
    )
    assert "no unexplored applicable strategies remain (round cap)" in reasons


def test_decide_next_continues_without_reasons() -> None:
    decision = decide_next(
        budget={"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=1,
        experiments=[],
        no_improvement_rounds=0,
        candidates_remaining=5,
    )
    assert decision == "plan_experiment"


def test_decide_next_finalizes_when_candidates_gone() -> None:
    decision = decide_next(
        budget={"max_experiments": 12, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=2,
        experiments=[_exp({"rmse": 0.4})],
        no_improvement_rounds=0,
        candidates_remaining=0,
    )
    assert decision == "finalize_report"


def test_decide_next_finalizes_on_budget_exhaustion() -> None:
    decision = decide_next(
        budget={"max_experiments": 1, "max_minutes": 30.0, "elapsed_minutes": 0.1},
        plan_round=1,
        experiments=[_exp({"rmse": 0.4})],
        no_improvement_rounds=0,
        candidates_remaining=3,
    )
    assert decision == "finalize_report"