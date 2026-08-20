"""Representation plugin base classes and shared utilities.

A representation is a deterministic transformation of a SMILES list into a
feature matrix (or dataset object).  Instances are cheap; they hold only the
state needed to make ``transform`` reproducible.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from cta_qsar.core.interfaces import CostEstimate


class Representation(abc.ABC):
    """Abstract representation plugin."""

    name: str = ""
    version: str = "0.1.0"
    kind: str = "fingerprint"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abc.abstractmethod
    def fit(self, smiles: list[str]) -> Representation:
        """Fit on the training molecules (e.g., learn feature names)."""

    @abc.abstractmethod
    def transform(self, smiles: list[str]) -> np.ndarray:
        """Transform SMILES into a feature matrix.

        Rows are in the same order as ``smiles``; invalid molecules are the
        caller's responsibility (standardization runs first).
        """

    def fit_transform(self, smiles: list[str]) -> np.ndarray:
        return self.fit(smiles).transform(smiles)

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        """Return (applicable, reason).  Base: applicable for any task."""
        return True, "default"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        runtime = max(0.5, n_molecules * 0.002)
        return CostEstimate(
            runtime_seconds=runtime,
            memory_gb=max(0.1, n_molecules * 2048 * 4e-9),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "params": self.params,
        }


class FingerprintRepresentation(Representation):
    kind = "fingerprint"

    def transform(self, smiles: list[str]) -> np.ndarray:
        raise NotImplementedError

    def fit(self, smiles: list[str]) -> FingerprintRepresentation:
        return self


class DescriptorRepresentation(Representation):
    kind = "descriptors"

    def transform(self, smiles: list[str]) -> np.ndarray:
        raise NotImplementedError

    def fit(self, smiles: list[str]) -> DescriptorRepresentation:
        return self


class GraphRepresentation(Representation):
    kind = "graph"

    def transform(self, smiles: list[str]) -> np.ndarray:
        raise NotImplementedError


class EmbeddingRepresentation(Representation):
    kind = "embedding"

    def transform(self, smiles: list[str]) -> np.ndarray:
        raise NotImplementedError