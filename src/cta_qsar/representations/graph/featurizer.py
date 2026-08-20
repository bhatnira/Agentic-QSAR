"""Molecular graph featurization for GNN models.

Converts SMILES to (adjacency, node-feature) tensors in pure PyTorch.
The graph representation itself is consumed by deep models; the registry also
keeps a "graph" representation key that triggers GNN-capable model plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

_ELEMENTS = {
    "C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "F": 5, "Cl": 6, "Br": 7,
    "I": 8, "B": 9, "Si": 10, "Se": 11, "Na": 12, "K": 13, "Li": 14,
    "Ca": 15, "Mg": 16, "Fe": 17, "Zn": 18, "Cu": 19,
}
N_FEATURES = 6


@dataclass
class MolGraph:
    """Lightweight graph container that works with or without torch."""

    smiles: str
    atom_features: np.ndarray  # (n_atoms, N_FEATURES)
    adjacency: np.ndarray  # (n_atoms, n_atoms)
    n_atoms: int

    def to_torch(self) -> tuple[Any, Any]:
        import torch

        return (
            torch.tensor(self.atom_features, dtype=torch.float32),
            torch.tensor(self.adjacency, dtype=torch.float32),
        )


def atom_features(mol: Chem.Mol) -> np.ndarray:
    feats: list[list[float]] = []
    for atom in mol.GetAtoms():
        f = [0.0] * N_FEATURES
        f[0] = _ELEMENTS.get(atom.GetSymbol(), len(_ELEMENTS)) / len(_ELEMENTS)
        f[1] = atom.GetDegree() / 6.0
        f[2] = float(atom.GetImplicitValence()) / 4.0
        f[3] = float(atom.GetFormalCharge())
        f[4] = float(atom.GetIsAromatic())
        f[5] = float(atom.GetTotalNumHs()) / 5.0
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def smiles_to_graph(smiles: str) -> MolGraph:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0
    adj = adj + np.eye(n, dtype=np.float32)  # self-loops
    return MolGraph(smiles=str(smiles), atom_features=atom_features(mol), adjacency=adj, n_atoms=n)


def featurize(smiles_list: list[str], max_atoms: int = 128) -> list[MolGraph]:
    return [smiles_to_graph(s) for s in smiles_list]