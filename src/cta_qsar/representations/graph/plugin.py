"""Graph representation plugin.

Unlike fingerprints, the "graph" representation does not produce a dense
matrix: ``transform`` returns the featurized :class:`MolGraph` list, which the
GNN model plugin consumes directly.  Its matrix form is used only for
similarity analysis, so classical models on graphs are not offered.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import GraphRepresentation
from cta_qsar.representations.graph.featurizer import MolGraph, featurize


class MolecularGraph(GraphRepresentation):
    name = "graph"
    version = "1.0.0"

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401

            return True, "molecular graph representation for GNN models (CPU-capable)"
        except ImportError:
            return False, "torch not installed"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        return CostEstimate(runtime_seconds=max(1.0, n_molecules * 0.01), memory_gb=0.5)

    def fit(self, smiles: list[str]) -> MolecularGraph:
        return self

    def transform(self, smiles: list[str]) -> list[MolGraph]:
        return featurize(smiles)

PLUGINS = [MolecularGraph]
