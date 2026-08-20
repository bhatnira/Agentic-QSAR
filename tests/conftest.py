"""Shared pytest fixtures: small CPU-only config, mock LLM, dataset files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cta_qsar.core.config import Config
from tests.fixtures import datasets


@pytest.fixture
def tiny_config() -> Config:
    """A fast CPU-only config: 1-2 cheap experiments, 3-fold CV, no tracking."""
    return Config(
        llm={"provider": "mock", "model": "heuristic"},
        compute={
            "max_minutes": 30.0,
            "max_experiments": 2,
            "max_memory_gb": 8.0,
            "gpu_required": False,
        },
        dataset={"max_rows": 1000, "standardize": True, "drop_invalid": False},
        experiment={
            "n_splits": 3,
            "n_repeats": 1,
            "test_fraction": 0.2,
            "random_seed": 42,
            "min_cv_score_improvement": 0.01,
        },
        representations={"enabled": ["morgan", "rdkit_descriptors"]},
        models={"enabled": ["ridge", "random_forest"]},
        validation={"enabled": ["random", "scaffold"]},
        trust={"required": ["predictive", "generalization", "applicability_domain"]},
        uncertainty={"enabled": False},
        explainability={"enabled": False},
        tracking={"enabled": False},
        reporting={"format": "markdown", "output_dir": "runs"},
    )


@pytest.fixture
def run_config() -> Config:
    """Config for full end-to-end runs: a few experiments, scaffold evidence."""
    return Config(
        llm={"provider": "mock", "model": "heuristic"},
        compute={
            "max_minutes": 30.0,
            "max_experiments": 3,
            "max_memory_gb": 8.0,
            "gpu_required": False,
        },
        dataset={"max_rows": 5000, "standardize": True, "drop_invalid": False},
        experiment={
            "n_splits": 3,
            "n_repeats": 1,
            "test_fraction": 0.2,
            "random_seed": 42,
            "min_cv_score_improvement": 0.01,
        },
        representations={"enabled": ["morgan"]},
        models={"enabled": ["ridge", "random_forest"]},
        validation={"enabled": ["random", "scaffold"]},
        trust={"required": ["predictive", "generalization"]},
        uncertainty={"enabled": False},
        explainability={"enabled": False},
        tracking={"enabled": False},
        reporting={"format": "markdown", "output_dir": "runs"},
    )


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    df = datasets.make_regression(n=150)
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def classification_csv(tmp_path: Path) -> Path:
    df = datasets.make_classification(n=150)
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def imbalanced_csv(tmp_path: Path) -> Path:
    df = datasets.make_classification(n=300, imbalance=12.0)
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def scaffold_heavy_csv(tmp_path: Path) -> Path:
    df = datasets.make_scaffold_heavy(n=180)
    path = tmp_path / "scaffold_heavy.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def dirty_csv(tmp_path: Path) -> Path:
    df = datasets.make_regression(n=150)
    df = datasets.with_missing_values(df, fraction=0.05)
    df = datasets.with_invalid_smiles(df, fraction=0.1)
    df = datasets.with_duplicates(df, n_duplicates=5)
    df = datasets.with_conflicting_labels(df, n_groups=3)
    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def ambiguous_endpoint_csv(tmp_path: Path) -> Path:
    df = datasets.make_ambiguous_endpoint(n=120)
    path = tmp_path / "ambiguous.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_llm():
    from cta_qsar.llm.mock import HeuristicModel

    return HeuristicModel(model="heuristic")


def check_df_columns(df: pd.DataFrame, expected: list[str]) -> None:
    """Test helper: assert the required columns exist in the right order."""
    assert list(df.columns[: len(expected)]) == expected