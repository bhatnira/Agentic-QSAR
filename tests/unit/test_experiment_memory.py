"""Unit tests: scientific memory, experiment records, and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

from cta_qsar.core.interfaces import ExperimentRecord
from cta_qsar.memory.experiment_memory import ExperimentMemory, make_run_id


def _record(**overrides: object) -> ExperimentRecord:
    fields: dict = {
        "id": "abc12345",
        "dataset_hash": "deadbeef00",
        "preprocessing_version": "rdkit-standardization-1.0.0",
        "representation": "morgan",
        "model": "ridge",
        "split": "random",
        "random_seed": 42,
        "metrics": {"rmse": 0.5},
        "trust": {},
        "result": "completed",
    }
    fields.update(overrides)
    return ExperimentRecord(**fields)


def test_record_signature_is_unique_per_configuration() -> None:
    a = _record()
    b = _record(id="different")
    assert a.signature == b.signature  # id must not affect the signature
    c = _record(representation="maccs")
    d = _record(random_seed=7)
    assert a.signature != c.signature
    assert a.signature != d.signature


def test_signature_ignores_metrics_and_trust() -> None:
    a = _record(metrics={"rmse": 0.5}, trust={})
    b = _record(metrics={"rmse": 0.9}, trust={"deep": {}})
    assert a.signature == b.signature


def test_memory_persists_records_to_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    memory = ExperimentMemory(run_dir=run_dir)
    memory.add(_record())
    memory.add(_record(id="second", random_seed=7))
    assert (run_dir / "experiments.jsonl").exists()
    lines = (run_dir / "experiments.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["signature"] == _record().signature


def test_memory_reloads_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "reload"
    memory = ExperimentMemory(run_dir=run_dir)
    memory.add(_record())
    loaded = ExperimentMemory.load(run_dir)
    assert len(loaded.records) == 1
    assert loaded.records[0].representation == "morgan"


def test_best_experiment_by_metric_priority(tmp_path: Path) -> None:
    memory = ExperimentMemory(run_dir=tmp_path / "best")
    memory.add(_record(id="worse", metrics={"rmse": 0.9}))
    memory.add(_record(id="better", metrics={"rmse": 0.3}))
    best = memory.best_experiment(("rmse", "r2"))
    assert best is not None
    assert best.id == "better"


def test_best_experiment_skips_failed_records(tmp_path: Path) -> None:
    memory = ExperimentMemory(run_dir=tmp_path / "skip")
    memory.add(_record(id="failed", result="failed", metrics={}))
    memory.add(_record(id="ok", metrics={"rmse": 0.4}))
    best = memory.best_experiment(("rmse", "r2"))
    assert best is not None
    assert best.result == "completed"


def test_run_id_is_timestamped(tmp_path: Path) -> None:
    run_id = make_run_id("regression")
    assert run_id.startswith("20")
    assert "regression" in run_id