"""Scientific experiment tracker.

The tracker owns the run's in-memory + on-disk experiment memory and the
compute budget, and answers the questions the planner needs: which experiment
signatures have already been executed (or failed), whether the budget allows
another experiment, and which configuration performed best so far.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cta_qsar.core.interfaces import ExperimentRecord
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.memory.experiment_memory import ExperimentMemory


class ExperimentTracker:
    """Track experiments + budget for one run and prevent repeats."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        max_experiments: int = 12,
        max_minutes: float = 30,
        max_memory_gb: float = 16,
        metric_priority: tuple[str, ...] = ("rmse", "roc_auc", "balanced_accuracy", "mcc", "r2"),
    ) -> None:
        self.memory = ExperimentMemory(run_dir=Path(run_dir))
        self.budget = BudgetState(
            max_experiments=max_experiments,
            max_minutes=max_minutes,
            max_memory_gb=max_memory_gb,
        )
        self.metric_priority = metric_priority

    # -- recording ---------------------------------------------------------
    def add(self, record: ExperimentRecord) -> None:
        """Record a completed/failed experiment and charge it to the budget."""
        self.memory.add(record)
        if record.result == "completed":
            self.budget.record(record.runtime_seconds, record.memory_gb)

    def signature_seen(self, signature: str) -> bool:
        return signature in self.memory.signatures_seen()

    def seen_signatures(self) -> set[str]:
        return self.memory.signatures_seen()

    # -- queries -----------------------------------------------------------
    def best(self) -> ExperimentRecord | None:
        """Best completed experiment by the first shared metric."""
        return self.memory.best_experiment(self.metric_priority)

    def completed(self) -> list[ExperimentRecord]:
        return [r for r in self.memory.records if r.result == "completed"]

    def failed(self) -> list[ExperimentRecord]:
        return [r for r in self.memory.records if r.result == "failed"]

    def all_as_dicts(self) -> list[dict[str, Any]]:
        return self.memory.all_as_dicts()

    def summary(self) -> dict[str, Any]:
        best = self.best()
        return {
            "experiments_done": self.budget.experiments_done,
            "experiments_remaining": self.budget.experiments_remaining,
            "completed": len(self.completed()),
            "failed": len(self.failed()),
            "elapsed_minutes": round(self.budget.elapsed_minutes, 2),
            "best_experiment_id": best.id if best else None,
            "stop_reasons": self.budget.stop_reasons(),
        }

    # -- stopping ----------------------------------------------------------
    def can_run_more(self, runtime_seconds: float | None = None) -> bool:
        """True if another experiment is still within budget."""
        return not self.budget.exhausted and not self.budget.would_exceed(runtime_seconds)

    def stop_reasons(self) -> list[str]:
        return self.budget.stop_reasons()

    @classmethod
    def load(cls, run_dir: str | Path) -> ExperimentTracker:
        """Rebuild a tracker from persisted experiment memory."""
        path = Path(run_dir)
        tracker = cls(path)
        tracker.memory = ExperimentMemory.load(path)
        for record in tracker.memory.records:
            if record.result == "completed":
                tracker.budget.record(record.runtime_seconds, record.memory_gb)
        return tracker