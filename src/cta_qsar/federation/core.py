"""Challenge session: coordination of competing agents + knowledge accumulation.

A session owns ONE in-memory EvidenceStore and ONE in-memory KnowledgeGraph
that live for the whole challenge. After each competing agent completes, its
outcome is digested into the evidence store (and persisted to the shared
evidence path so later agents -- and later challenges -- see it), the graph
is merged, and the next agent plans on the accumulated knowledge. The final
report combines the leaderboard (compete) with the graph diff (accumulate +
evolve).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cta_qsar.federation.cards import (
    AgentOutcome,
    ChallengeReport,
    build_leaderboard,
    primary_key_for,
)
from cta_qsar.knowledge.facts import EvidenceStore, dataset_class
from cta_qsar.knowledge.graph import KnowledgeGraph

AgentFn = Callable[..., dict[str, Any]]

INPROGRESS_COLUMNS = [
    "dataset", "scenario", "seed", "task_type", "rows",
    "primary", "primary_value", "run_id", "best_model",
]


@dataclass
class ChallengeSession:
    """Sequential, deterministic multi-agent challenge over one dataset."""

    evidence_path: str | Path
    agent_fn: AgentFn
    registry: Any | None = None
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    kg: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    kg_edges_before: set[tuple[str, str, str]] = field(default_factory=set)
    kg_sources_before: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.evidence_path = Path(self.evidence_path)
        if self.evidence_path.exists():
            self.evidence = EvidenceStore.load(self.evidence_path)
        else:
            self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.kg = KnowledgeGraph.from_sources(self.evidence, registry=self.registry)
        self.kg_edges_before = self.kg.edge_set()
        self.kg_sources_before = sorted(self.kg.sources)

    def run_challenge(
        self,
        *,
        df: Any,
        dataset_name: str,
        task_type: str,
        n_rows: int,
        seed: int,
        cards: list[Any],
        primary: str = "",
        inprogress_path: str | Path | None = None,
    ) -> ChallengeReport:
        """Run every card independently, then build leaderboard + KG diff."""
        primary = primary or primary_key_for(task_type)
        cls = dataset_class(task_type, n_rows)
        outcomes: list[AgentOutcome] = []
        for card in cards:
            outcome = self._run_one(card, df, dataset_name, task_type, n_rows, seed, primary)
            outcomes.append(outcome)
            if outcome.ok and inprogress_path is not None:
                self._digest(outcome, dataset_name, task_type, n_rows, inprogress_path)
        leaderboard = build_leaderboard(outcomes, primary)
        winner = next((row for row in leaderboard if row.get("rank")), None)
        return ChallengeReport(
            dataset=dataset_name,
            seed=seed,
            task_type=task_type,
            primary=primary,
            leaderboard=leaderboard,
            winner=winner,
            kg_before=self.kg.stats(edges=len(self.kg_edges_before), sources=self.kg_sources_before),
            kg_after=self.kg.stats(),
            kg_diff=self.kg.diff_edge_set(self.kg_edges_before),
            dataset_class=cls,
        )

    def _run_one(self, card: Any, df: Any, dataset_name: str, task_type: str, n_rows: int, seed: int, primary: str) -> AgentOutcome:
        outcome = self.agent_fn(
            card=card, df=df, dataset_name=dataset_name, task_type=task_type,
            n_rows=n_rows, seed=seed, primary=primary,
        )
        return AgentOutcome(**{k: v for k, v in outcome.items() if k in AgentOutcome.__dataclass_fields__})

    def _digest(self, outcome: AgentOutcome, dataset_name: str, task_type: str, n_rows: int, inprogress_path: str | Path) -> None:
        """Accumulate one agent's outcome: persist + merge into the in-memory KG."""
        path = Path(inprogress_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=INPROGRESS_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "scenario": outcome.card,
                    "seed": outcome.seed,
                    "task_type": task_type,
                    "rows": n_rows,
                    "primary": outcome.primary,
                    "primary_value": outcome.primary_value,
                    "run_id": outcome.run_id or f"{outcome.card}|{outcome.seed}",
                    "best_model": outcome.best_model or "unknown",
                }
            )
        from cta_qsar.knowledge.ingestor import ingest_results_file

        self.evidence = EvidenceStore.load(self.evidence_path)
        ingest_results_file(self.evidence, path)
        self.evidence.save(self.evidence_path)
        self.kg = KnowledgeGraph.from_sources(self.evidence, registry=self.registry)