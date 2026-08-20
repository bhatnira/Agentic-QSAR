"""Integration tests: full autonomous runs with the mock LLM on CPU.

These exercise the complete LangGraph workflow from CSV to final report
exactly as the CLI does, but with the deterministic heuristic LLM so no API
and no GPU are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cta_qsar.agents.scientist import QSARScientist
from cta_qsar.core.config import Config


def _run(scientist: QSARScientist, data: Path, **kwargs) -> dict:
    return scientist.run(data, **kwargs)


def _artifacts(output_root: Path, run_id: str) -> dict[str, Path]:
    run_dir = output_root / run_id
    return {
        "report_json": run_dir / "report.json",
        "report_md": run_dir / "report.md",
        "experiments": run_dir / "experiments.jsonl",
        "provenance": run_dir / "provenance.json",
        "environment": run_dir / "environment.txt",
    }


@pytest.mark.integration
def test_full_regression_run(tmp_path: Path, regression_csv: Path, run_config: Config) -> None:
    output = tmp_path / "runs"
    scientist = QSARScientist(run_config, output_root=output)
    report = _run(scientist, regression_csv)
    assert report["run_id"]
    assert len(report["experiments"]) >= 2
    assert report["endpoint"]["task_type"] == "regression"
    assert report["best_experiment"] is not None

    artifacts = _artifacts(output, report["run_id"])
    for artifact in artifacts.values():
        assert artifact.exists(), f"missing artifact {artifact}"
    persisted = json.loads(artifacts["report_json"].read_text())
    assert persisted == report
    provenance = json.loads(artifacts["provenance"].read_text())
    assert provenance["llm"]["provider"] == "mock"
    assert "dataset_hash" in provenance


@pytest.mark.integration
def test_full_classification_run(tmp_path: Path, classification_csv: Path, run_config: Config) -> None:
    output = tmp_path / "runs"
    scientist = QSARScientist(run_config, output_root=output)
    report = _run(scientist, classification_csv)
    assert report["endpoint"]["task_type"] == "binary"
    metrics_keys = set(report["best_experiment"]["metrics"])
    assert "roc_auc" in metrics_keys or "balanced_accuracy" in metrics_keys


@pytest.mark.integration
def test_report_contains_scientific_sections(
    tmp_path: Path, regression_csv: Path, run_config: Config
) -> None:
    scientist = QSARScientist(run_config, output_root=tmp_path / "runs")
    report = _run(scientist, regression_csv)
    for section in (
        "dataset_profile",
        "endpoint",
        "data_quality",
        "standardization",
        "chemical_space",
        "validation_strategy",
        "representations_considered",
        "models_considered",
        "experiments",
        "performance_comparison",
        "generalization_results",
        "applicability_domain_results",
        "best_experiment",
        "final_model_selection_rationale",
        "limitations",
        "recommended_next_experiments",
        "executive_summary",
    ):
        assert report.get(section) is not None, f"report missing section {section}"


@pytest.mark.integration
def test_marked_molecules_preserved_in_output(
    tmp_path: Path, dirty_csv: Path, run_config: Config
) -> None:
    """Invalid SMILES and missing targets must be flagged, never silently removed."""
    scientist = QSARScientist(run_config, output_root=tmp_path / "runs")
    report = _run(scientist, dirty_csv)
    quality = report["data_quality"]
    assert quality["invalid_smiles"] > 0
    assert quality["duplicate_molecules"]["n_duplicates"] >= 1
    assert quality["conflicting_labels"]["n_conflicting_groups"] >= 1
    assert quality["duplicate_rows"] >= 1


@pytest.mark.integration
def test_scaffold_heavy_dataset_triggers_generalization_evidence(
    tmp_path: Path, scaffold_heavy_csv: Path, run_config: Config
) -> None:
    scientist = QSARScientist(run_config, output_root=tmp_path / "runs")
    report = _run(scientist, scaffold_heavy_csv)
    for experiment in report["experiments"]:
        assert experiment["split"] in ("random", "scaffold")
    generalization = report["generalization_results"]
    assert generalization is not None


@pytest.mark.integration
def test_imbalanced_classification_reported(
    tmp_path: Path, imbalanced_csv: Path, run_config: Config
) -> None:
    scientist = QSARScientist(run_config, output_root=tmp_path / "runs")
    report = _run(scientist, imbalanced_csv)
    balance = report["data_quality"]["class_balance"]
    assert balance["applicable"]
    assert balance["imbalance_ratio"] > 3


@pytest.mark.integration
def test_budget_exhaustion_stops_early(tmp_path: Path, regression_csv: Path) -> None:
    config = Config(
        llm={"provider": "mock"},
        compute={"max_minutes": 30.0, "max_experiments": 1, "max_memory_gb": 8.0},
        experiment={"n_splits": 2, "n_repeats": 1, "test_fraction": 0.2, "random_seed": 42},
        representations={"enabled": ["morgan"]},
        models={"enabled": ["ridge"]},
        validation={"enabled": ["random"]},
        tracking={"enabled": False},
        reporting={"output_dir": "runs"},
    )
    scientist = QSARScientist(config, output_root=tmp_path / "runs")
    report = _run(scientist, regression_csv)
    assert len(report["experiments"]) == 1
    assert report["experiments"][0]["result"] == "completed"
    assert any(
        "budget exhausted" in r for r in report["computational_cost"]["stop_reasons"]
    )


@pytest.mark.integration
def test_self_correction_runs_multiple_experiments(
    tmp_path: Path, regression_csv: Path, run_config: Config
) -> None:
    """The agent must return to plan_experiment after the first experiment."""
    scientist = QSARScientist(run_config, output_root=tmp_path / "runs")
    report = _run(scientist, regression_csv)
    assert len(report["experiments"]) >= 2
    assert report["best_experiment"] is not None


@pytest.mark.integration
def test_experiment_already_done_is_not_repeated(tmp_path: Path, regression_csv: Path) -> None:
    config = Config(
        llm={"provider": "mock"},
        compute={"max_minutes": 30.0, "max_experiments": 2, "max_memory_gb": 8.0},
        experiment={"n_splits": 2, "n_repeats": 1, "test_fraction": 0.2, "random_seed": 42},
        representations={"enabled": ["morgan"]},
        models={"enabled": ["ridge"]},
        validation={"enabled": ["random", "scaffold"]},
        tracking={"enabled": False},
        reporting={"output_dir": "runs"},
    )
    scientist = QSARScientist(config, output_root=tmp_path / "runs")
    report = _run(scientist, regression_csv)
    signatures = {
        (e["representation"], e["model"], e["split"]) for e in report["experiments"]
    }
    assert len(signatures) == len(report["experiments"])  # no repeats


@pytest.mark.integration
def test_profile_command_works(tmp_path: Path, regression_csv: Path, tiny_config: Config) -> None:
    scientist = QSARScientist(tiny_config)
    state = scientist.profile(regression_csv)
    assert state["profile"]["n_rows"] > 0
    assert state["endpoint"]["task_type"] == "regression"
    assert "risks" in state["quality_report"] or "n_rows" in state["quality_report"]