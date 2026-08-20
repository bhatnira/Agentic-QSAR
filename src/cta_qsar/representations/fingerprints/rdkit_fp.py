"""RDKit topological fingerprint representation."""

from __future__ import annotations

from typing import Any

from cta_qsar.chemistry.fingerprints import rdkit_fingerprints
from cta_qsar.representations.base import FingerprintRepresentation


class RDKitFingerprint(FingerprintRepresentation):
    name = "rdkit_fp"
    version = "1.0.0"

    def __init__(self, n_bits: int = 2048) -> None:
        super().__init__(n_bits=n_bits)
        self.n_bits = n_bits

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "daylight-style topological fingerprints"

    def transform(self, smiles: list[str]) -> Any:
        return rdkit_fingerprints(smiles, n_bits=self.n_bits)

PLUGINS = [RDKitFingerprint]
