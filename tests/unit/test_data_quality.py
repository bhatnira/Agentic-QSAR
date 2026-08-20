"""Unit tests: chemical/data quality control."""

from __future__ import annotations

import pandas as pd

from cta_qsar.chemistry.validation import quality_report
from tests.fixtures import datasets


def _report(df: pd.DataFrame, task_type: str = "regression") -> dict:
    return quality_report(
        df,
        smiles_column=df.columns[0],
        target_column=df.columns[1],
        task_type=task_type,
        endpoint={"task_type": task_type},
    )


def test_clean_dataset_has_no_issues() -> None:
    df = datasets.make_regression(n=150)
    report = _report(df)
    assert report["duplicate_molecules"]["n_duplicates"] == 0
    assert report["conflicting_labels"]["n_conflicting_groups"] == 0
    assert report["missing_values"]["n_cells_missing"] == 0
    assert report["invalid_smiles"] == 0


def test_duplicate_molecules_detected() -> None:
    df = datasets.with_duplicates(datasets.make_regression(n=100), n_duplicates=4)
    report = _report(df)
    assert report["duplicate_molecules"]["n_duplicates"] >= 1
    assert report["duplicate_molecules"]["n_duplicate_rows"] >= 4


def test_conflicting_labels_detected() -> None:
    df = datasets.with_conflicting_labels(datasets.make_regression(n=100), n_groups=3)
    report = _report(df)
    assert report["conflicting_labels"]["n_conflicting_groups"] >= 1
    example = report["conflicting_labels"]["examples"][0]
    assert len(example["targets"]) > 1


def test_missing_values_reported() -> None:
    df = datasets.with_missing_values(datasets.make_regression(n=120), fraction=0.2)
    report = _report(df)
    assert report["missing_values"]["n_cells_missing"] > 0
    assert report["missing_value_columns"] == ["pIC50"]


def test_outliers_flagged_but_not_removed() -> None:
    import numpy as np

    y = [1.0 + (i % 3) * 0.1 for i in range(60)] + [50.0, 55.0, 60.0]
    df = pd.DataFrame(
        {"SMILES": datasets.random_molecules(np.random.default_rng(1), 63), "y": y}
    )
    report = _report(df)
    assert report["outliers"]["applicable"]
    assert report["outliers"]["n_extreme"] >= 1
    assert len(df) == 63  # nothing removed
    assert "NOT removed" in report["outliers"]["note"]


def test_class_balance_for_imbalanced_classification() -> None:
    df = datasets.make_classification(n=300, imbalance=12.0)
    report = _report(df, task_type="binary")
    balance = report["class_balance"]
    assert balance["applicable"]
    assert balance["imbalance_ratio"] > 3
    assert 0 < balance["minority_fraction"] < 0.5


def test_target_distribution_summary() -> None:
    df = datasets.make_regression(n=100)
    report = _report(df)
    dist = report["target_distribution"]
    assert dist["applicable"]
    assert dist["n"] == len(df)  # unique molecules after generator dedup
    assert dist["std"] > 0