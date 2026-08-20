"""Unit tests: compute budget enforcement."""

from __future__ import annotations

from cta_qsar.experiments.budget import BudgetState


def test_budget_counts_experiments() -> None:
    budget = BudgetState(max_experiments=3, max_minutes=10, max_memory_gb=8)
    assert budget.experiments_remaining == 3
    budget.record(5.0, 0.1)
    budget.record(5.0, 0.1)
    assert budget.experiments_done == 2
    assert budget.experiments_remaining == 1
    assert not budget.exhausted


def test_budget_exhaustion() -> None:
    budget = BudgetState(max_experiments=2, max_minutes=10, max_memory_gb=8)
    budget.record(1.0)
    budget.record(1.0)
    assert budget.experiments_remaining == 0
    assert budget.exhausted
    assert "experiment budget exhausted" in budget.stop_reasons()


def test_time_budget_exhaustion() -> None:
    import time

    budget = BudgetState(max_experiments=100, max_minutes=0.1, max_memory_gb=8)
    budget.started_at = time.time() - 60  # 60 s elapsed vs a 6 s cap
    assert budget.exhausted
    assert "compute time budget exhausted" in budget.stop_reasons()
    assert budget.would_exceed()


def test_memory_tracking_captures_peak() -> None:
    budget = BudgetState(max_experiments=5, max_minutes=10, max_memory_gb=1.0)
    budget.record(1.0, memory_gb=0.5)
    budget.record(1.0, memory_gb=2.5)
    assert budget.max_memory_gb == 2.5  # tracked peak, not the configured cap


def test_would_exceed_time_budget() -> None:
    budget = BudgetState(max_experiments=5, max_minutes=10, max_memory_gb=8)
    # far below the cap -> allowed
    assert not budget.would_exceed(runtime_seconds=1.0)


def test_to_dict_summary() -> None:
    budget = BudgetState(max_experiments=4, max_minutes=10, max_memory_gb=8)
    budget.record(2.0)
    summary = budget.to_dict()
    assert summary["experiments_done"] == 1
    assert summary["experiments_remaining"] == 3
    assert summary["total_runtime_seconds"] == 2.0