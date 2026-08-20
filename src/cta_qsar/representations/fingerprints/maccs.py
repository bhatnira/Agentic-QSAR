"""MACCS structural keys representation."""

from __future__ import annotations

from typing import Any

from cta_qsar.chemistry.fingerprints import maccs_fingerprints
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import FingerprintRepresentation


class MACCSFingerprint(FingerprintRepresentation):
    name = "maccs"
    version = "1.0.0"

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "166-bit MACCS structural keys"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        return CostEstimate(runtime_seconds=max(0.3, n_molecules * 0.0005), memory_gb=0.1)

    def transform(self, smiles: list[str]) -> Any:
        return maccs_fingerprints(smiles)

PLUGINS = [MACCSFingerprint]
