"""Mordred descriptors representation (optional dependency).

Registers the plugin even when mordred is not installed; ``transform`` raises
``PluginUnavailableError`` with a clear message so the planner can skip it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import DescriptorRepresentation


def _mordred_available() -> bool:
    try:
        import mordred  # noqa: F401

        return True
    except ImportError:
        return False


class MordredDescriptors(DescriptorRepresentation):
    name = "mordred"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._calc = None
        self.feature_names: list[str] = []

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        if not _mordred_available():
            return False, "mordred not installed"
        return True, "1600+ descriptors if mordred is installed"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        return CostEstimate(runtime_seconds=max(2.0, n_molecules * 0.05), memory_gb=0.5)

    def fit(self, smiles: list[str]) -> MordredDescriptors:
        if not _mordred_available():
            raise PluginUnavailableError(
                "Mordred is not installed; pip install mordred to enable this representation."
            )
        from mordred import Calculator, descriptors

        self._calc = Calculator(descriptors, ignore_3D=True)
        frame = self._calc.pandas([self._to_mol(s) for s in smiles])
        self.feature_names = [c for c in frame.columns if frame[c].notna().any()]
        return self

    def transform(self, smiles: list[str]) -> np.ndarray:
        if self._calc is None:
            self.fit(smiles)
        from mordred import Calculator, descriptors  # type: ignore[import-not-found]

        if self._calc is None:
            self._calc = Calculator(descriptors, ignore_3D=True)
        frame = self._calc.pandas([self._to_mol(s) for s in smiles])
        keep = [c for c in self.feature_names if c in frame.columns]
        matrix = frame[keep].fillna(0.0).to_numpy(dtype=float)
        std = matrix.std(axis=0)
        matrix[:, std > 0] /= np.where(std > 0, std, 1)[np.newaxis, :]
        return matrix

    @staticmethod
    def _to_mol(smiles: str) -> Any:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise PluginUnavailableError(f"invalid molecule: {smiles}")
        return mol

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "available": _mordred_available(),
            "n_features": len(self.feature_names) if self.feature_names else None,
        }

PLUGINS = [MordredDescriptors]
