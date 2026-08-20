"""Topological-torsion fingerprint representation."""

from __future__ import annotations

from typing import Any

from cta_qsar.chemistry.fingerprints import torsion_fingerprints
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import FingerprintRepresentation


class TorsionFingerprint(FingerprintRepresentation):
    name = "torsion"
    version = "1.0.0"

    def __init__(self, n_bits: int = 2048) -> None:
        super().__init__(n_bits=n_bits)
        self.n_bits = n_bits

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "topological torsion fingerprints"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        return CostEstimate(runtime_seconds=max(0.4, n_molecules * 0.0015), memory_gb=max(0.1, n_molecules * self.n_bits * 4e-9))

    def transform(self, smiles: list[str]) -> Any:
        return torsion_fingerprints(smiles, n_bits=self.n_bits)

PLUGINS = [TorsionFingerprint]
