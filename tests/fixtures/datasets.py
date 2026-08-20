"""Synthetic QSAR dataset builders for tests.

All datasets are generated deterministically (fixed seed) with RDKit so the
entire test-suite stays CPU-only and does not need any downloaded files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

SEED = 42

FRAGMENTS = [
    "c1ccccc1",
    "c1ccc(cc1)C",
    "c1cc(ccc1)O",
    "c1ccc(cc1)Cl",
    "c1ccc(cc1)F",
    "c1ccc(cc1)N",
    "c1ccc(cc1)OC",
    "CCO",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccn1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "CC(=O)",
    "COC(=O)",
    "C#N",
    "c1scnc1",
    "CF",
    "CCl",
    "CCC",
    "CC(C)N",
    "c1ccc(cc1)[N+](=O)[O-]",
]

SCAFFOLDS = [
    "c1ccc2ccccc2c1",      # naphthalene series
    "c1ccc(cc1)C1CCNCC1",  # aniline-piperidine series
    "c1scnc1C",            # thiazole series
]


def random_molecules(rng: np.random.Generator, n: int, max_fragments: int = 4) -> list[str]:
    molecules: list[str] = []
    attempts = 0
    while len(molecules) < n and attempts < n * 50:
        attempts += 1
        count = int(rng.integers(1, max_fragments + 1))
        smiles = "".join(rng.choice(FRAGMENTS, size=count))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:  # noqa: BLE001
            continue
        molecules.append(Chem.MolToSmiles(mol))
    return molecules


def _mol_features(smiles: str) -> tuple[float, float, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 0.0
    logp = Crippen.MolLogP(mol)
    heavy = float(Lipinski.HeavyAtomCount(mol))
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(mol.GetNumAtoms(), 1)
    return float(logp), heavy, aromatic


def make_regression(
    n: int = 200,
    *,
    columns: tuple[str, str] = ("SMILES", "pIC50"),
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    seen: set[str] = set()
    for smiles in random_molecules(rng, n):
        if smiles in seen:
            continue
        seen.add(smiles)
        logp, mw, aromatic = _mol_features(smiles)
        y = 2.2 + 0.55 * logp - 0.003 * mw + 1.1 * aromatic + rng.normal(0, 0.35)
        rows.append((smiles, round(float(max(y, 0.1)), 4)))
    return pd.DataFrame(rows, columns=list(columns))


def make_classification(
    n: int = 200,
    *,
    columns: tuple[str, str] = ("SMILES", "active"),
    seed: int = SEED,
    imbalance: float | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    seen: set[str] = set()
    for smiles in random_molecules(rng, n):
        if smiles in seen:
            continue
        seen.add(smiles)
        logp, mw, aromatic = _mol_features(smiles)
        bias = 0.0 if imbalance is None else np.log(imbalance)
        score = 0.6 * logp + 0.4 * aromatic - 0.01 * mw + bias
        probability = 1 / (1 + np.exp(-score))
        label = int(rng.random() < probability)
        rows.append((smiles, label))
    return pd.DataFrame(rows, columns=list(columns))


def make_multiclass(n: int = 180, *, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for smiles in random_molecules(rng, n):
        logp, mw, aromatic = _mol_features(smiles)
        score = 0.6 * logp + 0.4 * aromatic - 0.01 * mw
        label = int(score) % 4
        rows.append((smiles, label))
    return pd.DataFrame(rows, columns=["SMILES", "class"])


def make_scaffold_heavy(n: int = 200, *, seed: int = SEED) -> pd.DataFrame:
    """Dataset built from a small number of scaffolds (series dependence)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        core = SCAFFOLDS[i % len(SCAFFOLDS)]
        mol = Chem.MolFromSmiles(core)
        for _ in range(int(rng.integers(0, 3))):
            frag = rng.choice(FRAGMENTS)
            combo = Chem.MolFromSmiles(core + frag)
            if combo is not None:
                mol = combo
        smiles = Chem.MolToSmiles(mol)
        logp, mw, aromatic = _mol_features(smiles)
        y = 1.5 + 0.6 * logp + 0.8 * aromatic + rng.normal(0, 0.3)
        rows.append((smiles, round(float(y), 4), int(i // 10)))
    return pd.DataFrame(rows, columns=["SMILES", "pIC50", "series"])


def with_missing_values(df: pd.DataFrame, fraction: float = 0.05, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    target = out.columns[1]
    mask = rng.random(len(out)) < fraction
    out.loc[mask, target] = np.nan
    return out


def with_invalid_smiles(df: pd.DataFrame, fraction: float = 0.1, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    smiles_col = out.columns[0]
    mask = rng.random(len(out)) < fraction
    out.loc[mask, smiles_col] = "not-a-smiles-[["
    return out


def with_duplicates(df: pd.DataFrame, n_duplicates: int = 5, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dupes = df.sample(n=n_duplicates, random_state=int(rng.integers(0, 2**31)))
    return pd.concat([df, dupes], ignore_index=True)


def with_conflicting_labels(df: pd.DataFrame, n_groups: int = 3, seed: int = SEED) -> pd.DataFrame:
    """Duplicate SMILES rows carrying different target values."""
    rng = np.random.default_rng(seed)
    extras = []
    sample = df.sample(n=n_groups, random_state=int(rng.integers(0, 2**31)))
    for _, row in sample.iterrows():
        flipped = row.copy()
        flipped.iloc[1] = row.iloc[1] + (1.5 if isinstance(row.iloc[1], (int, float)) else 1)
        extras.append(flipped)
    return pd.concat([df, pd.DataFrame(extras)], ignore_index=True)


def make_ambiguous_endpoint(n: int = 200, *, seed: int = SEED) -> pd.DataFrame:
    """A target with 20+ unique non-numeric labels -> ambiguous endpoint."""
    rng = np.random.default_rng(seed)
    labels = [f"compound_batch_{i}" for i in range(25)]
    rows = []
    for smiles in random_molecules(rng, n):
        rows.append((smiles, rng.choice(labels), rng.integers(0, 2)))
    return pd.DataFrame(rows, columns=["SMILES", "batch_id", "active"])


def make_multitask(n: int = 150, *, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for smiles in random_molecules(rng, n):
        logp, mw, aromatic = _mol_features(smiles)
        y1 = 1.0 + 0.5 * logp + rng.normal(0, 0.2)
        y2 = 2.0 - 0.4 * aromatic + rng.normal(0, 0.2)
        rows.append((smiles, [round(y1, 3), round(y2, 3)]))
    return pd.DataFrame(rows, columns=["SMILES", "targets"])