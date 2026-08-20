"""Deterministic RDKit fingerprint calculators."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, MACCSkeys  # noqa: F401  (Draw kept for future viz)
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect

from cta_qsar.core.exceptions import ChemistryError

FPParams = dict[str, Any]


def _mols(smiles: list[str]) -> list[Chem.Mol | None]:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        out.append(mol)
    return out


def _to_array(bitvectors: list) -> np.ndarray:

    if not bitvectors:
        return np.zeros((0, 0), dtype=np.uint8)
    arr = np.zeros((len(bitvectors), bitvectors[0].GetNumBits()), dtype=np.uint8)
    for i, bv in enumerate(bitvectors):
        on = list(bv.GetOnBits())
        if on:
            arr[i, on] = 1
    return arr


def morgan_fingerprints(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    vectors = []
    for mol in _mols(smiles):
        if mol is None:
            raise ChemistryError("Cannot fingerprint invalid molecule (see standardization)")
        vectors.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return _to_array(vectors)


def rdkit_fingerprints(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    vectors = []
    for mol in _mols(smiles):
        if mol is None:
            raise ChemistryError("Cannot fingerprint invalid molecule (see standardization)")
        vectors.append(Chem.RDKFingerprint(mol, fpSize=n_bits))
    return _to_array(vectors)


def maccs_fingerprints(smiles: list[str]) -> np.ndarray:
    vectors = []
    for mol in _mols(smiles):
        if mol is None:
            raise ChemistryError("Cannot fingerprint invalid molecule (see standardization)")
        vectors.append(MACCSkeys.GenMACCSKeys(mol))
    return _to_array(vectors)


def atompair_fingerprints(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    vectors = []
    for mol in _mols(smiles):
        if mol is None:
            raise ChemistryError("Cannot fingerprint invalid molecule (see standardization)")
        vectors.append(GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits))
    return _to_array(vectors)


def torsion_fingerprints(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    vectors = []
    for mol in _mols(smiles):
        if mol is None:
            raise ChemistryError("Cannot fingerprint invalid molecule (see standardization)")
        vectors.append(Torsions.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits))
    return _to_array(vectors)


def tanimoto_similarity_matrix(fps: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity from a bit matrix (memory-safe for <= 10k rows)."""
    n = len(fps)
    if n == 0:
        return np.zeros((0, 0))
    dense = (fps > 0).astype(np.float64)
    dot = dense @ dense.T
    norms = np.sum(dense, axis=1)
    denom = norms[:, None] + norms[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom > 0, dot / np.maximum(denom, 1e-12), 0.0)
    np.fill_diagonal(sim, 1.0)
    return sim