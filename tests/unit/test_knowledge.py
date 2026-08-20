"""Tests for the knowledge-graph evidence layer."""

from __future__ import annotations

import csv
import json

import pytest

from cta_qsar.knowledge.curated_loader import load_curated_facts
from cta_qsar.knowledge.explain import (
    append_trace,
    counterfactual_report,
    explain_decision,
    render_evidence_board,
)
from cta_qsar.knowledge.facts import EvidenceStore, Fact, dataset_class
from cta_qsar.knowledge.ingestor import ingest_jsonl, ingest_results_file
from cta_qsar.knowledge.static_builder import build_registry_facts, registry_version


def _write_results(tmp_path, path: str, rows: list[dict]) -> str:
    """Helper: write a benchmark-style results.csv."""
    out = tmp_path / path
    fields = ["dataset", "scenario", "best_model", "primary", "primary_value",
              "n_experiments", "runtime_seconds", "seed", "task_type", "rows", "run_id"]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return str(out)


class TestEvidenceStore:
    def test_window_keeps_most_recent(self):
        store = EvidenceStore()
        for i in range(25):
            store.add_value("cls|small", "agent-mock", "maccs+svr", i, run_id=f"r{i}", level=4)
        fact = store.fact("cls|small", "agent-mock", "maccs+svr")
        assert fact.attrs["n"] == 20

    def test_idempotent_merge_same_run(self):
        store = EvidenceStore()
        store.add_value("cls|small", "agent-mock", "maccs+svr", 1.5, run_id="r1", level=4)
        store.add_value("cls|small", "agent-mock", "maccs+svr", 1.5, run_id="r1", level=4)
        store.add_value("cls|small", "agent-mock", "maccs+svr", 2.5, run_id="r2", level=4)
        assert store.fact("cls|small", "agent-mock", "maccs+svr").attrs["n"] == 2

    def test_roundtrip_persists_values_and_sign(self):
        import tempfile

        store = EvidenceStore()
        store.add_value("cls|small", "agent-mock", "maccs+svr", 1.0, run_id="r1", level=4, sign=-1.0)
        store.add_value("cls|small", "agent-mock", "maccs+svr", 3.0, run_id="r2", level=4, sign=-1.0)
        store.add_value("cls|large", "*", "*", 0.9, run_id="r3", level=1, sign=1.0)
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/evidence.jsonl"
            store.save(p)
            loaded = EvidenceStore.load(p)
        fact = loaded.fact("cls|small", "agent-mock", "maccs+svr")
        assert fact.attrs["mean"] == 2.0
        assert fact.attrs["std"] == 1.0
        assert fact.attrs["sign"] == -1.0
        assert loaded.fact("cls|large", "*", "*").attrs["mean"] == 0.9

    def test_facts_for_fine_then_coarse_fallback(self):
        store = EvidenceStore()
        # coarse class aggregate only (level 1)
        store.add_value("cls|small", "*", "*", 0.9, run_id="r1", level=1)
        store.add_value("cls|small", "*", "*", 0.8, run_id="r2", level=1)
        # fine scenario facts below min_n
        store.add_value("cls|small", "agent-mock", "maccs+svr", 1.0, run_id="r3", level=4)
        facts = store.facts_for("cls|small", min_n=2)
        assert all(f.object == "*" for f in facts)
        assert facts[0].attrs["n"] == 2

    def test_exclude_predicate_counterfactual(self):
        store = EvidenceStore()
        for model, value in (("maccs+svr", 1.0), ("morgan+xgboost", 2.0)):
            for run in ("r0", "r1"):
                store.add_value("cls|small", "agent-mock", model, value, run_id=run, level=4, sign=-1.0)
                store.add_value("cls|small", "grid", "ridge", 5.0, run_id=run, level=4, sign=-1.0)
        best = store.best_for("cls|small", "agent-mock")
        assert best is not None and best[0] == "maccs+svr"
        alternative = store.facts_for("cls|small", exclude_predicate="agent-mock")
        assert all(f.predicate != "agent-mock" for f in alternative)
        assert alternative and alternative[0].object == "ridge"

    def test_dataset_class_buckets(self):
        assert dataset_class("regression", 100) == "regression|tiny"
        assert dataset_class("regression", 1000) == "regression|small"
        assert dataset_class("binary", 3000) == "binary|medium"
        assert dataset_class("binary", 10000) == "binary|large"


class TestIngestor:
    def test_results_csv_ingestion(self, tmp_path):
        path = _write_results(tmp_path, "results.csv", [
            {"dataset": "esol", "scenario": "agent-mock", "best_model": "maccs+svr[scaffold]",
             "primary": "rmse", "primary_value": 1.21, "seed": 0, "task_type": "regression", "rows": 1128},
            {"dataset": "esol", "scenario": "agent-mock", "best_model": "maccs+svr[scaffold]",
             "primary": "rmse", "primary_value": 1.19, "seed": 1, "task_type": "regression", "rows": 1128},
        ])
        store = EvidenceStore()
        updated = ingest_results_file(store, path)
        assert updated == 6  # 2 rows x 3 granularity levels
        fact = store.fact("regression|small", "agent-mock", "maccs+svr[scaffold]")
        assert fact.attrs["n"] == 2
        assert fact.attrs["mean"] == pytest.approx(1.20)
        assert fact.attrs["sign"] == -1.0  # rmse is lower-is-better

    def test_reingest_is_idempotent(self, tmp_path):
        path = _write_results(tmp_path, "results.csv", [
            {"dataset": "esol", "scenario": "agent-mock", "best_model": "maccs+svr[scaffold]",
             "primary": "rmse", "primary_value": 1.21, "seed": 0, "task_type": "regression", "rows": 1128},
        ])
        store = EvidenceStore()
        ingest_results_file(store, path)
        ingest_results_file(store, path)
        assert store.fact("regression|small", "agent-mock", "maccs+svr[scaffold]").attrs["n"] == 1

    def test_jsonl_ingestion(self, tmp_path):
        records = [
            {"result": "completed", "dataset": "d1", "model": "ridge", "representation": "morgan",
             "split": "scaffold", "primary": "rmse", "primary_value": 0.5, "seed": 1, "run_id": "x1"},
            {"result": "failed", "dataset": "d1", "model": "ridge", "primary": "rmse", "primary_value": 9.9, "seed": 2},
        ]
        path = tmp_path / "experiments.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records))
        store = EvidenceStore()
        updated = ingest_jsonl(store, path, task_type="regression", rows=100)
        assert updated == 1
        assert store.fact("regression|tiny", "internal-run", "ridge").attrs["n"] == 1


class TestStaticAndCurated:
    def test_registry_facts(self):
        from cta_qsar.core.registry import get_registry

        registry = get_registry()
        registry.auto_discover()
        facts = build_registry_facts(registry)
        assert facts
        assert any(f.predicate == "requires" and f.object == "graph" for f in facts)
        assert any(f.predicate == "supports" for f in facts)
        assert registry_version(registry) == registry_version(registry)

    def test_curated_facts_have_sources(self):
        facts = load_curated_facts()
        assert len(facts) >= 5
        assert all(f.source for f in facts)
        assert all(f.subject and f.predicate and f.object for f in facts)


class TestExplain:
    def test_evidence_board_renders(self):
        facts = [
            Fact("regression|small", "agent-mock", "maccs+svr[scaffold]",
                 level=4, attrs={"mean": 1.2, "std": 0.1, "n": 3, "sign": -1.0}),
        ]
        text = render_evidence_board(facts)
        assert "maccs+svr[scaffold]" in text and "mean=1.200" in text

    def test_empty_board(self):
        assert "no accumulated evidence" in render_evidence_board([])

    def test_decision_trace_is_template_only(self):
        trace = {
            "round": 1,
            "chosen": "maccs+svr[scaffold]",
            "reason": "evidence-backed",
            "evidence": [],
            "winner": "ridge",
            "winner_boost": 0.15,
            "adjacency": ["maccs+xgboost[scaffold]"],
        }
        text = explain_decision(trace)
        assert "chose strategy" in text
        assert "+0.15" in text
        assert "maccs+xgboost[scaffold]" in text

    def test_counterfactual_uses_drop_one(self):
        store = EvidenceStore()
        for run in ("r1", "r2"):
            store.add_value("cls", "agent-mock", "maccs+svr", 1.0, run_id=run, level=4, sign=-1.0)
            store.add_value("cls", "agent-mock", "morgan+xgboost", 2.0, run_id=run, level=4, sign=-1.0)
        report = counterfactual_report(store, "cls", predicate="agent-mock")
        assert "Without maccs+svr" in report

    def test_append_trace_writes_jsonl(self, tmp_path):
        trace = {"round": 2, "chosen": "a", "reason": "x", "evidence": [], "winner": None, "winner_boost": None, "adjacency": []}
        path = tmp_path / "plan_trace.jsonl"
        append_trace(trace, path)
        append_trace(trace, path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["chosen"] == "a"


class TestEvidenceDrivenPlanner:
    def test_generate_candidates_uses_evidence(self):
        from cta_qsar.core.interfaces import QSARCase
        from cta_qsar.core.registry import get_registry
        from cta_qsar.experiments.budget import BudgetState
        from cta_qsar.experiments.planner import generate_candidates
        from cta_qsar.knowledge.facts import Fact

        registry = get_registry()
        registry.auto_discover()
        case = QSARCase(
            dataset_size=500,
            n_unique_molecules=500,
            task_type="regression",
            endpoint_name="solubility",
            endpoint_confidence=0.9,
            endpoint_reasoning="test",
            risks=[],
        )
        evidence = [Fact("regression|small", "agent-mock", "maccs+svr[scaffold]",
                         level=4, attrs={"mean": -1.2, "std": 0.1, "n": 4, "sign": -1.0})]
        budget = BudgetState(max_experiments=5, max_minutes=30, max_memory_gb=8)
        with_evidence = generate_candidates(
            registry=registry, case=case, enabled_representations=["morgan", "maccs"],
            enabled_models=["ridge", "svr"], validated_splits=["random"], budget=budget,
            history=[], task_type="regression", n_samples=500, dataset_props={}, hardware_tier="cpu",
            evidence=evidence,
        )
        no_evidence = generate_candidates(
            registry=registry, case=case, enabled_representations=["morgan", "maccs"],
            enabled_models=["ridge", "svr"], validated_splits=["random"], budget=budget,
            history=[], task_type="regression", n_samples=500, dataset_props={}, hardware_tier="cpu",
        )
        boosted = next(c for c in with_evidence if c.representation == "maccs" and c.model == "svr")
        plain = next(c for c in no_evidence if c.representation == "maccs" and c.model == "svr")
        assert boosted.utility > plain.utility
        assert "evidence" in (boosted.reason or "")

    def test_winner_boost_from_history(self):
        from cta_qsar.experiments.planner import _winner_boost_map

        history = [
            {"result": "completed", "representation": "morgan", "model": "ridge",
             "primary": "rmse", "primary_value": 0.5, "split": "random", "seed": 42},
            {"result": "completed", "representation": "maccs", "model": "svr",
             "primary": "rmse", "primary_value": 0.9, "split": "scaffold", "seed": 42},
        ]
        boost = _winner_boost_map(history)
        assert boost == {"morgan|ridge": 0.15}
        assert _winner_boost_map([{"result": "failed"}]) == {}