"""RDKit whole-molecule descriptors representation."""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.chemistry.descriptors import (
    compute_rdkit_descriptors,
    sanitize_descriptor_frame,
)
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import DescriptorRepresentation


class RDKitDescriptors(DescriptorRepresentation):
    name = "rdkit_descriptors"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: list[str] = []

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "200+ physicochemical descriptors; captures global molecular properties"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        return CostEstimate(
            runtime_seconds=max(1.0, n_molecules * 0.01),
            memory_gb=max(0.1, n_molecules * 210 * 8e-9),
        )

    def fit(self, smiles: list[str]) -> RDKitDescriptors:
        frame = compute_rdkit_descriptors(smiles)
        _, names = sanitize_descriptor_frame(frame, fill_small=True)
        self.feature_names = names
        return self

    def transform(self, smiles: list[str]) -> np.ndarray:
        frame = compute_rdkit_descriptors(smiles)
        if not self.feature_names:
            self.fit(smiles)
        available = [c for c in self.feature_names if c in frame.columns]
        matrix = frame[available].fillna(frame[available].median()).to_numpy(dtype=float)
        std = matrix.std(axis=0)
        matrix[:, std > 0] /= np.where(std > 0, std, 1)[np.newaxis, :]
        return matrix

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "n_features": len(self.feature_names),
        }

PLUGINS = [RDKitDescriptors]
