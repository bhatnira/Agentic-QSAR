"""In-memory knowledge graph: a NetworkX overlay on the evidence layer.

The EvidenceStore persists attributed (subject, predicate, object) triples;
this module materializes them -- together with registry-derived capability
facts and curated chemistry priors -- into a real in-memory directed graph
(MultiDiGraph) so the planner can answer adjacency questions ("what is
connected to this dataset class / strategy?"), rank strategies and snapshot
state for audit.

Design rules (inherited from the evidence layer):
  - the JSONL evidence store remains the source of truth; the graph is a
    read-mostly in-memory index, rebuildable from the store at any time
  - accumulation happens in memory during a session (e.g. a challenge in
    which competing agents ingest their results one by one) and is persisted
    as a JSONL edge dump next to the evidence file
  - everything is deterministic (sorted iteration) so diffs and traces are
    reproducible
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx

from cta_qsar.knowledge.facts import MIN_EVIDENCE, EvidenceStore, Fact

STRATEGY_SPLIT = "["  # strategy objects look like "model+rep[split]"


class KnowledgeGraph:
    """Deterministic in-memory overlay graph over the evidence layer."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.sources: set[str] = set()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_sources(
        cls,
        store: EvidenceStore | None = None,
        *,
        registry: Any | None = None,
        curated: bool = True,
        facts: Iterable[Fact] | None = None,
    ) -> KnowledgeGraph:
        """Build the in-memory graph from the persisted store plus static facts."""
        kg = cls()
        if store is not None:
            kg.merge_store(store)
        if registry is not None:
            from cta_qsar.knowledge.static_builder import build_registry_facts

            kg.merge_facts(build_registry_facts(registry), source="registry")
        if curated:
            from cta_qsar.knowledge.curated_loader import load_curated_facts

            kg.merge_facts(load_curated_facts(), source="curated")
        if facts:
            kg.merge_facts(list(facts), source="caller")
        return kg

    def merge_store(self, store: EvidenceStore) -> None:
        """Materialize all aggregate facts of a store into the graph."""
        self.merge_facts(store.facts_all(), source="evidence")

    def merge_facts(self, facts: Iterable[Fact], *, source: str) -> None:
        """Add (or refresh) one edge per fact; idempotent by (s, p, o)."""
        if source:
            self.sources.add(source)
        graph = self.graph
        for fact in sorted(facts, key=lambda f: (f.subject, f.predicate, f.object)):
            attrs = {k: v for k, v in fact.attrs.items()}
            attrs.update(predicate=fact.predicate, level=fact.level, source=fact.source or source)
            if graph.has_edge(fact.subject, fact.object, fact.predicate):
                graph[fact.subject][fact.object][fact.predicate].update(attrs)
            else:
                graph.add_edge(fact.subject, fact.object, fact.predicate, **attrs)
        _ensure_kinds(graph)

    # ------------------------------------------------------------------- read
    def neighbors(self, node: str, *, predicate: str | None = None) -> list[tuple[str, dict[str, Any]]]:
        """Outgoing (neighbor, attrs) pairs; deterministic order."""
        rows = [
            (target, dict(edge))
            for _s, target, _p, edge in self.graph.edges(node, data=True, keys=True)
            if predicate is None or edge.get("predicate") == predicate
        ]
        return sorted(rows, key=lambda row: (row[1].get("predicate", ""), row[0]))

    def adjacency(self, node: str) -> list[dict[str, Any]]:
        """Machine-readable local adjacency for trace/audit fields."""
        rows = []
        for target, attrs in self.neighbors(node):
            rows.append(
                {
                    "object": target,
                    "predicate": attrs.get("predicate", ""),
                    "mean": attrs.get("mean"),
                    "n": attrs.get("n"),
                    "sign": attrs.get("sign", 1.0),
                }
            )
        return rows

    def best_for(self, dataset_class: str, predicate: str) -> tuple[str, dict[str, Any]] | None:
        """Highest signed mean strategy for a predicate within a dataset class."""
        best: tuple[str, dict[str, Any]] | None = None
        graph = self.graph
        for target in sorted(graph.successors(dataset_class)):
            edge = graph.get_edge_data(dataset_class, target, predicate)
            if edge is None or edge.get("n", 0) < MIN_EVIDENCE or target == "*":
                continue
            sign = edge.get("sign", 1.0)
            score = sign * float(edge.get("mean", 0.0))
            if best is None or score > best[1].get("score", -1e18):
                best = (target, {**edge, "score": round(score, 4)})
        return best

    def stats(self, *, edges: int | None = None, sources: list[str] | None = None) -> dict[str, Any]:
        return {
            "n_nodes": int(self.graph.number_of_nodes()),
            "n_edges": int(self.graph.number_of_edges()) if edges is None else edges,
            "sources": sorted(self.sources) if sources is None else sorted(sources),
        }

    # ----------------------------------------------------------------- audit
    def edge_set(self) -> set[tuple[str, str, str]]:
        """Exact (subject, predicate, object) edge triples (for diffs)."""
        return {
            (subject, predicate, obj)
            for subject, obj, predicate in self.graph.edges(keys=True)
        }

    def diff_edge_set(self, before: set[tuple[str, str, str]]) -> dict[str, Any]:
        """Added/removed (subject, predicate, object) edges vs a snapshot."""
        current = self.edge_set()
        return {
            "added_edges": sorted(current - before),
            "removed_edges": sorted(before - current),
        }

    def diff(self, previous: KnowledgeGraph | None) -> dict[str, Any]:
        """Added/removed edges vs a prior graph instance."""
        before = previous.edge_set() if previous is not None else set()
        return self.diff_edge_set(before)

    # ------------------------------------------------------------- persistence
    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for subject, obj, predicate, edge in self.graph.edges(data=True, keys=True):
            rows.append(
                {
                    "subject": subject,
                    "predicate": edge.get("predicate", predicate),
                    "object": obj,
                    "level": edge.get("level", 2),
                    "source": edge.get("source", ""),
                    "attrs": {k: v for k, v in edge.items() if k not in ("predicate", "level", "source")},
                }
            )
        with path.open("w") as fh:
            for row in sorted(rows, key=lambda r: (r["subject"], r["predicate"], r["object"])):
                fh.write(json.dumps(row) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> KnowledgeGraph:
        kg = cls()
        path = Path(path)
        if not path.exists():
            return kg
        facts: list[Fact] = []
        sources: set[str] = set()
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                facts.append(
                    Fact(
                        subject=row["subject"], predicate=row["predicate"], object=row["object"],
                        level=int(row.get("level", 2)), source=str(row.get("source", "")),
                        attrs={k: v for k, v in (row.get("attrs") or {}).items()},
                    )
                )
                sources.add(str(row.get("source", "evidence")))
        kg.merge_facts(facts, source="")
        kg.sources |= sources
        return kg


def render_graph_summary(kg: KnowledgeGraph, *, dataset_class: str | None = None) -> str:
    """Compact, deterministic digest of the in-memory graph for planner context."""
    stats = kg.stats()
    lines = [
        f"knowledge graph: {stats['n_nodes']} nodes, {stats['n_edges']} edges "
        f"(sources: {', '.join(stats['sources']) or 'empty'})"
    ]
    if dataset_class is not None:
        adjacency = kg.adjacency(dataset_class)
        if adjacency:
            lines.append(f"neighbors of {dataset_class}:")
            for row in adjacency[:8]:
                meta = f"mean={row['mean']:.3f} n={row['n']}" if row.get("mean") is not None else ""
                lines.append(f"  {row['predicate']} -> {row['object']} {meta}")
        else:
            lines.append(f"no evidence yet for {dataset_class}")
    else:
        for node in sorted(kg.graph.nodes())[:6]:
            lines.append(f"  {node}: {len(list(kg.graph.successors(node)))} outgoing edges")
    return "\n".join(lines)


def _ensure_kinds(graph: nx.MultiDiGraph) -> None:
    """Tag nodes with a coarse kind for provenance and rendering."""
    for node in graph.nodes():
        if graph.nodes[node].get("kind"):
            continue
        if "|" in str(node):
            kind = "dataset_class"
        elif STRATEGY_SPLIT in str(node):
            kind = "strategy"
        else:
            kind = "plugin"
        graph.nodes[node]["kind"] = kind