"""Compute budget enforcement for the agent loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetState:
    max_experiments: int
    max_minutes: float
    max_memory_gb: float
    started_at: float = field(default_factory=time.time)
    experiments_done: int = 0
    total_runtime_seconds: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def experiments_remaining(self) -> int:
        return max(0, self.max_experiments - self.experiments_done)

    @property
    def exhausted(self) -> bool:
        if self.experiments_remaining <= 0:
            return True
        return self.max_minutes > 0 and self.elapsed_minutes >= self.max_minutes

    def would_exceed(self, runtime_seconds: float | None = None) -> bool:
        """True if adding another experiment would break the time budget."""
        if self.experiments_remaining <= 1 and self.experiments_done > 0:
            # allow exactly max_experiments
            return False
        if self.max_minutes <= 0:
            return False
        extra = runtime_seconds or 0.0
        return self.elapsed_minutes + extra / 60.0 >= self.max_minutes

    def record(self, runtime_seconds: float, memory_gb: float = 0.0) -> None:
        self.experiments_done += 1
        self.total_runtime_seconds += runtime_seconds
        self.max_memory_gb = max(self.max_memory_gb, memory_gb)

    def stop_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.experiments_remaining <= 0:
            reasons.append("experiment budget exhausted")
        if self.max_minutes > 0 and self.elapsed_minutes >= self.max_minutes:
            reasons.append("compute time budget exhausted")
        return reasons

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_experiments": self.max_experiments,
            "max_minutes": self.max_minutes,
            "elapsed_minutes": round(self.elapsed_minutes, 2),
            "experiments_done": self.experiments_done,
            "experiments_remaining": self.experiments_remaining,
            "total_runtime_seconds": round(self.total_runtime_seconds, 2),
        }