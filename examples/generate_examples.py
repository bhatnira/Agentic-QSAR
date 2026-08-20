"""Generate synthetic QSAR example datasets with real chemistry.

Usage:  python3 examples/generate_examples.py
Output: examples/regression.csv, examples/classification.csv
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

SEED = 42
rng = np.random.default_rng(SEED)

FRAGMENTS = [
    "c1ccccc1", "c1ccc(cc1)C", "c1cc(ccc1)O", "c1ccc(cc1)Cl", "c1ccc(cc1)F",
    "c1ccc(cc1)N", "c1ccc(cc1)NC(C)C", "c1ccc(cc1)OC", "CCO", "CC(C)C",
    "CCN(CC)CC", "C1CCCCC1", "C1CCNCC1", "c1ccccn1", "c1ccncc1", "c1ccc2ccccc2c1",
    "c1ccc2c(c1)ccc3ccccc23", "CC(=O)", "COC(=O)", "C#N", "CC(C)(C)", "c1scnc1",
    "c1sccc1", "c1nccnc1", "CF", "CCl", "CCC", "CCCC", "CC(C)N", "c1ccc(cc1)[N+](=O)[O-]",
]


def random_smiles(max_fragments: int = 4) -> str:
    n = rng.integers(1, max_fragments + 1)
    smiles = "".join(rng.choice(FRAGMENTS, size=n))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return random_smiles(max_fragments)
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        return random_smiles(max_fragments)
    return Chem.MolToSmiles(mol)


def lipophilicity_feature(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return Crippen.MolLogP(mol)
    except Exception:  # noqa: BLE001
        return 0.0


def size_feature(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(Lipinski.HeavyAtomCount(mol))


def aromatic_fraction(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    atoms = mol.GetAtoms()
    return sum(1 for a in atoms if a.GetIsAromatic()) / max(len(atoms), 1)


def make_regression(n: int = 400) -> pd.DataFrame:
    rows = []
    while len(rows) < n:
        smiles = random_smiles()
        if smiles in {r[0] for r in rows}:
            continue
        logp = lipophilicity_feature(smiles)
        mw = _mw(smiles)
        aromatic = aromatic_fraction(smiles)
        y = 2.2 + 0.55 * logp - 0.003 * mw + 1.1 * aromatic + rng.normal(0, 0.35)
        rows.append((smiles, round(float(max(y, 0.1)), 3), round(logp, 2), round(mw, 1)))
    return pd.DataFrame(rows, columns=["SMILES", "pIC50", "LogP", "MW"])


def make_classification(n: int = 400) -> pd.DataFrame:
    rows = []
    while len(rows) < n:
        smiles = random_smiles()
        if smiles in {r[0] for r in rows}:
            continue
        logp = lipophilicity_feature(smiles)
        mw = _mw(smiles)
        aromatic = aromatic_fraction(smiles)
        score = 0.6 * logp + 0.4 * aromatic - 0.01 * mw
        probability = 1 / (1 + np.exp(-score))
        label = int(rng.random() < probability)
        rows.append((smiles, label, round(float(score), 3)))
    return pd.DataFrame(rows, columns=["SMILES", "active", "score"])


def _mw(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(Descriptors.MolWt(mol))


def main() -> None:
    import pathlib

    out_dir = pathlib.Path(__file__).parent
    regression = make_regression()
    regression.to_csv(out_dir / "regression.csv", index=False)
    classification = make_classification()
    classification.to_csv(out_dir / "classification.csv", index=False)
    print(f"regression: {regression.shape} -> examples/regression.csv")
    print(f"classification: {classification.shape} -> examples/classification.csv")


if __name__ == "__main__":
    main()