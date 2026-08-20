"""Self-improving planner policy: deterministic, audited meta-adaptation.

The planner's ranking utility and the stopping policy's settle threshold are
the system's *decision policy*. With ``policy.adaptive`` enabled, the system
learns both from its own trace after every completed iteration:

  * utility weights  -- per dataset class, the planner re-weights its three
    value signals (expected improvement, information gain, trustworthiness
    gain) proportionally to how well each predicted the realized marginal
    gain, via a bounded multiplicative update (no randomness, no mutation).
  * settle delta -- the "negligible improvement" threshold in the stopping
    policy is learned as a quantile of this class's observed round-over-round
    improvements, so stopping is calibrated to realized dynamics instead of a
    hardcoded 0.005.

State is namespaced by dataset class and persisted as JSONL next to the
knowledge evidence store; policy adaptation is a read-mostly, conservative
force -- weights are clamped to [0.5, 2.0] and every update is recorded as a
``policy_update`` trace event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_WEIGHTS = {"weight_improvement": 1.0, "weight_information": 1.0, "weight_trust": 1.0}
DEFAULT_SETTLE_DELTA = 0.005
WEIGHT_BOUNDS = (0.5, 2.0)


@dataclass
class PolicyState:
    """Per-dataset-class policy state."""

    dataset_class: str
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    improvement_history: list[float] = field(default_factory=list)
    settle_delta: float | None = None
    updates_applied: int = 0
    last_event: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_class": self.dataset_class,
            "weights": self.weights,
            "improvement_history": self.improvement_history,
            "settle_delta": self.settle_delta,
            "updates_applied": self.updates_applied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyState:
        return cls(
            dataset_class=str(data.get("dataset_class", "")),
            weights={
                k: float(v)
                for k, v in (data.get("weights") or DEFAULT_WEIGHTS).items()
                if k in DEFAULT_WEIGHTS
            },
            improvement_history=[float(v) for v in data.get("improvement_history", [])],
            settle_delta=(
                float(data["settle_delta"]) if data.get("settle_delta") is not None else None
            ),
            updates_applied=int(data.get("updates_applied", 0)),
        )


class PolicyStore:
    """Jsonl-backed map of dataset class -> PolicyState (mirrors EvidenceStore)."""

    def __init__(self, path: Any) -> None:
        self.path = path
        self.states: dict[str, PolicyState] = {}

    def get(self, dataset_class: str) -> PolicyState:
        return self.states.setdefault(dataset_class, PolicyState(dataset_class))

    def update(self, dataset_class: str, state: PolicyState) -> None:
        self.states[dataset_class] = state

    def effective_settle_delta(self, dataset_class: str) -> float:
        state = self.states.get(dataset_class)
        if state is None or state.settle_delta is None:
            return DEFAULT_SETTLE_DELTA
        return state.settle_delta

    @classmethod
    def load(cls, path: Any, parents: bool = True) -> PolicyStore:
        from pathlib import Path

        path = Path(path)
        store = cls(path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    import json

                    state = PolicyState.from_dict(json.loads(line))
                    store.states[state.dataset_class] = state
                except Exception:  # noqa: BLE001
                    continue
        elif parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        return store

    def save(self) -> None:
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for state in sorted(self.states.values(), key=lambda s: s.dataset_class):
                fh.write(json.dumps(state.to_dict()) + "\n")