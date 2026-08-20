"""Unit tests for the in-memory knowledge graph (NetworkX overlay)."""

from __future__ import annotations

from cta_qsar.knowledge.facts import EvidenceStore, Fact
from cta_qsar.knowledge.graph import KnowledgeGraph, _ensure_kinds, render_graph_summary


def _store_with_facts() -> EvidenceStore:
    store = EvidenceStore()
    store.add_value("regression|small", "scenario", "morgan+ridge[random]", 0.9, run_id="r1", level=4)
    store.add_value("regression|small", "scenario", "morgan+ridge[random]", 0.8, run_id="r2", level=4)
    store.add_value("regression|small", "scenario", "maccs+xgboost[scaffold]", 1.2, run_id="r1", level=4)
    store.add_value("regression|small", "scenario", "maccs+xgboost[scaffold]", 1.1, run_id="r2", level=4)
    store.add_value("binary|tiny", "scenario", "graph+gcn[random]", 0.51, run_id="r3", level=4)
    return store


class TestBuild:
    def test_empty_store_gives_empty_graph(self):
        kg = KnowledgeGraph.from_sources(EvidenceStore(), curated=False)
        assert kg.stats()["n_nodes"] == 0 and kg.stats()["n_edges"] == 0
        assert kg.diff(KnowledgeGraph()) == {"added_edges": [], "removed_edges": []}

    def test_merge_store_materializes_triples(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        stats = kg.stats()
        assert stats["n_nodes"] == 5  # 3 classes-ish/strategies: 2 classes + 3 strategies
        assert stats["n_edges"] == 3
        assert stats["sources"] == ["evidence"]

    def test_node_kinds_assigned(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        kg.merge_facts([Fact("morgan", "is_kind", "fingerprint")], source="registry")
        _ensure_kinds(kg.graph)
        assert kg.graph.nodes["regression|small"]["kind"] == "dataset_class"
        assert kg.graph.nodes["morgan+ridge[random]"]["kind"] == "strategy"
        assert kg.graph.nodes["morgan"]["kind"] == "plugin"

    def test_merge_refreshes_edge_attrs(self):
        kg = KnowledgeGraph.from_sources(curated=False)
        kg.merge_facts(
            [Fact("regression|small", "scenario", "a+b[c]", attrs={"mean": 1.0, "n": 1})], source="evidence"
        )
        kg.merge_facts(
            [Fact("regression|small", "scenario", "a+b[c]", attrs={"mean": 1.5, "n": 2})], source="evidence"
        )
        attrs = kg.graph.get_edge_data("regression|small", "a+b[c]", "scenario")
        assert attrs["mean"] == 1.5 and attrs["n"] == 2


class TestRead:
    def test_neighbors_filtered_by_predicate(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        rows = kg.neighbors("regression|small", predicate="scenario")
        assert [r[0] for r in rows] == ["maccs+xgboost[scaffold]", "morgan+ridge[random]"]

    def test_best_for_uses_signed_mean(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        best, attrs = kg.best_for("regression|small", "scenario")
        assert best == "maccs+xgboost[scaffold]"
        assert attrs["score"] == 1.15

    def test_best_for_insufficient_evidence_returns_none(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        assert kg.best_for("binary|tiny", "scenario") is None  # only 1 observation

    def test_adjacency_machine_readable(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        rows = kg.adjacency("regression|small")
        assert rows[0]["object"] == "maccs+xgboost[scaffold]"
        assert rows[0]["predicate"] == "scenario"
        assert rows[0]["n"] == 2

    def test_render_summary_deterministic(self):
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        first = render_graph_summary(kg, dataset_class="regression|small")
        second = render_graph_summary(kg, dataset_class="regression|small")
        assert first == second
        assert "neighbors of regression|small" in first


class TestPersistenceAndDiff:
    def test_jsonl_round_trip(self, tmp_path):
        path = tmp_path / "kg.jsonl"
        kg = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        kg.to_jsonl(path)
        reloaded = KnowledgeGraph.load_jsonl(path)
        assert kg.edge_set() == reloaded.edge_set()
        assert reloaded.stats()["n_nodes"] == kg.stats()["n_nodes"]

    def test_load_missing_file_is_empty(self, tmp_path):
        kg = KnowledgeGraph.load_jsonl(tmp_path / "absent.jsonl")
        assert kg.stats()["n_edges"] == 0

    def test_diff_reports_added_removed(self):
        before = KnowledgeGraph.from_sources(_store_with_facts(), curated=False)
        after = KnowledgeGraph()
        after.merge_facts(
            [Fact("regression|small", "scenario", "new+model[c]", attrs={"mean": 1.0, "n": 3})],
            source="evidence",
        )
        diff = after.diff(before)
        assert ("regression|small", "scenario", "new+model[c]") in diff["added_edges"]
        removed = {e for e in diff["removed_edges"]}
        assert ("regression|small", "scenario", "morgan+ridge[random]") in removed


class TestRegistryFacts:
    def test_registry_facts_add_capability_edges(self):
        class Rep:
            name = "morgan"
            kind = "fingerprint"

        class Model:
            name = "ridge"
            kind = "classical"

            def supports(self):
                return ("regression", "binary")

        class Registry:
            def list(self, kind):
                return ["morgan"] if kind == "representation" else ["ridge"]

            def get(self, kind, name):
                return Rep() if kind == "representation" else Model()

        kg = KnowledgeGraph.from_sources(facts=[], registry=Registry(), curated=False)
        assert kg.graph.get_edge_data("morgan", "fingerprint", "is_kind") is not None
        assert kg.graph.get_edge_data("ridge", "regression", "supports") is not None
        assert kg.graph.get_edge_data("ridge+morgan", "true", "applicable") is not None