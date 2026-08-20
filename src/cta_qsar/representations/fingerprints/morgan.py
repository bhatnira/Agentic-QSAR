"""Morgan (ECFP) fingerprint representation."""

from __future__ import annotations

from typing import Any

from cta_qsar.chemistry.fingerprints import morgan_fingerprints
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import FingerprintRepresentation


class MorganFingerprint(FingerprintRepresentation):
    name = "morgan"
    version = "1.0.0"

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        super().__init__(radius=radius, n_bits=n_bits)
        self.radius = radius
        self.n_bits = n_bits

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "generic circular fingerprints; baseline representation"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        runtime = max(0.5, n_molecules * 0.001)
        return CostEstimate(runtime_seconds=runtime, memory_gb=max(0.1, n_molecules * self.n_bits * 4e-9))

    def transform(self, smiles: list[str]) -> Any:
        return morgan_fingerprints(smiles, radius=self.radius, n_bits=self.n_bits)

PLUGINS = [MorganFingerprint]
