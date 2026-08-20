"""Unit tests: endpoint detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cta_qsar.endpoints.detector import EndpointDetector
from tests.fixtures import datasets

detector = EndpointDetector()


def test_regression_endpoint() -> None:
    df = datasets.make_regression(n=200)
    detection = detector.detect(df, "pIC50")
    assert detection.task_type == "regression"
    assert detection.column_found
    assert detection.confidence > 0.4
    assert detection.transformation_status in ("untouched", "log-transformed")


def test_regression_not_inferred_from_name_alone() -> None:
    """A continuous column with an innocuous name must still resolve via values."""
    df = pd.DataFrame(
        {
            "SMILES": datasets.random_molecules(np.random.default_rng(0), 120),
            "mystery_value": np.random.default_rng(1).normal(5.0, 1.2, 120),
        }
    )
    detection = detector.detect(df, "mystery_value")
    assert detection.task_type == "regression"
    assert "mystery_value" in detection.reasoning


def test_binary_classification_from_integers() -> None:
    df = datasets.make_classification(n=200)
    detection = detector.detect(df, "active")
    assert detection.task_type == "binary"
    assert detection.n_classes == 2


def test_binary_classification_from_strings() -> None:
    df = pd.DataFrame(
        {
            "SMILES": datasets.random_molecules(np.random.default_rng(2), 100),
            "toxic": ["yes", "no"] * 50,
        }
    )
    detection = detector.detect(df, "toxic")
    assert detection.task_type == "binary"
    assert set(detection.class_labels) == {"no", "yes"}


def test_multiclass_classification() -> None:
    df = datasets.make_multiclass(n=160)
    detection = detector.detect(df, "class")
    assert detection.task_type == "multiclass"
    assert detection.n_classes == 4


def test_ambiguous_high_cardinality_endpoint() -> None:
    df = datasets.make_ambiguous_endpoint(n=100)
    detection = detector.detect(df, "batch_id")
    assert detection.ambiguous
    assert detection.task_type == "unknown"
    assert detection.ask_for


def test_missing_target_column() -> None:
    df = datasets.make_regression(n=50)
    detection = detector.detect(df, "does_not_exist")
    assert not detection.column_found
    assert detection.task_type == "unknown"


def test_multitask_detection() -> None:
    df = datasets.make_multitask(n=80)
    detection = detector.detect(df, "targets")
    assert detection.task_type == "multitask_regression"
    assert detection.n_targets == 2


def test_known_task_type_honored() -> None:
    df = pd.DataFrame(
        {"SMILES": datasets.random_molecules(np.random.default_rng(3), 100), "y": np.arange(100)}
    )
    detection = detector.detect(df, "y", known_task_type="regression")
    assert detection.task_type == "regression"