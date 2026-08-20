"""Federation: multiple independent strategy agents compete per dataset.

In the OPEN-ADMET spirit, a *challenge* runs several independently-configured
agents over the same dataset; each agent autonomously plans, executes and
finalizes its own model build. Their results are ranked on the primary metric
(leaderboard -- the competition) and digested back into the shared evidence
store and the in-memory knowledge graph (accumulation), which later agents --
and later challenges -- inherit. Ordering is deterministic: cards execute in
the order given, strategies are fixed by ``StrategyCard``, and nothing is
randomized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategyCard:
    """One independent agent configuration -- a fixed, explainable strategy."""

    name: str
    description: str = ""
    search: bool = True
    trust_gate: bool = True
    adaptive_policy: bool = False
    llm_provider: str = "mock"
    enabled_representations: list[str] | None = None
    enabled_models: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "search": self.search,
            "trust_gate": self.trust_gate,
            "adaptive_policy": self.adaptive_policy,
            "llm_provider": self.llm_provider,
            "enabled_representations": self.enabled_representations,
            "enabled_models": self.enabled_models,
        }


def default_cards(*, with_nvidia: bool = False) -> list[StrategyCard]:
    """The fixed default roster of competing agents (deterministic).

    ``with_nvidia`` gates the LLM-driven card (only included when an NVIDIA /
    OpenAI API key is present in the environment).
    """
    cards = [
        StrategyCard(
            name="gate:aggressive",
            description="search on, trust gate on, heuristic planner (baseline agent)",
        ),
        StrategyCard(
            name="fast:nosearch",
            description="search off, gate on -- cheap and quick",
            search=False,
        ),
        StrategyCard(
            name="gate:none",
            description="search on, gate disabled -- stops at the first stop-trigger",
            trust_gate=False,
        ),
        StrategyCard(
            name="evolve",
            description="search on, gate on, self-improving planner policy (shared store)",
            adaptive_policy=True,
        ),
    ]
    if with_nvidia:
        cards.append(
            StrategyCard(
                name="llm:nvidia",
                description="search on, gate on, real NVIDIA LLM strategy selection",
                llm_provider="nvidia",
            )
        )
    return cards


def primary_key_for(task_type: str) -> str:
    """Primary metric name per task type (mirrors the benchmark harness)."""
    base = task_type.removeprefix("multitask_")
    if base == "regression":
        return "rmse"
    if base == "multiclass":
        return "mcc"
    return "roc_auc"


def better_is_higher(primary: str) -> bool:
    return primary not in ("rmse", "mae")


@dataclass
class AgentOutcome:
    """One competing agent's full result."""

    card: str
    seed: int
    primary: str
    primary_value: float
    best_model: str = ""
    best_hyperparams: str = ""
    n_experiments: int = 0
    runtime_seconds: float = 0.0
    run_id: str = ""
    failed: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card,
            "seed": self.seed,
            "primary": self.primary,
            "primary_value": round(self.primary_value, 4),
            "best_model": self.best_model,
            "best_hyperparams": self.best_hyperparams,
            "n_experiments": self.n_experiments,
            "runtime_seconds": round(self.runtime_seconds, 1),
            "run_id": self.run_id,
            "failed": self.failed,
        }


@dataclass
class ChallengeReport:
    """Leaderboard (compete) + knowledge graph diff (accumulation) for a challenge."""

    dataset: str
    seed: int
    task_type: str
    primary: str
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    winner: dict[str, Any] | None = None
    kg_before: dict[str, Any] | None = None
    kg_after: dict[str, Any] | None = None
    kg_diff: dict[str, Any] = field(default_factory=dict)
    dataset_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "task_type": self.task_type,
            "primary": self.primary,
            "dataset_class": self.dataset_class,
            "leaderboard": self.leaderboard,
            "winner": self.winner,
            "kg_before": self.kg_before,
            "kg_after": self.kg_after,
            "kg_diff": self.kg_diff,
        }


def build_leaderboard(outcomes: list[AgentOutcome], primary: str) -> list[dict[str, Any]]:
    """Rank completed agents by primary metric (higher-is-better unless rmse)."""
    completed = [o for o in outcomes if o.ok and o.primary == primary]
    completed.sort(
        key=lambda o: o.primary_value,
        reverse=better_is_higher(primary),
    )
    ranked = []
    for rank, outcome in enumerate(completed, start=1):
        row = outcome.to_dict()
        row["rank"] = rank
        ranked.append(row)
    for outcome in [o for o in outcomes if not o.ok]:
        row = outcome.to_dict()
        row["rank"] = None
        ranked.append(row)
    return ranked