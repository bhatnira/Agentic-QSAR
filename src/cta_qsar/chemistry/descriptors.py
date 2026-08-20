"""RDKit descriptor computation.

Uses the base ``Descriptors`` module (no dependency on optional mordred).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

_BASE_BLOCKLIST = {"Ipc"}


def rdkit_descriptor_list() -> list[str]:
    """Names of all stable RDKit descriptor calculators."""
    names = [d[0] for d in Descriptors._descList]
    return [n for n in names if n not in _BASE_BLOCKLIST]


def compute_rdkit_descriptors(smiles: list[str]) -> pd.DataFrame:
    """Compute the full standard RDKit descriptor table.

    Returns a DataFrame with one row per molecule; NaN marks descriptors that
    could not be computed for that molecule.
    """
    names = rdkit_descriptor_list()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    rows: list[Any] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            rows.append([float("nan")] * len(names))
            continue
        try:
            rows.append(list(calc.CalcDescriptors(mol)))
        except Exception:  # noqa: BLE001
            rows.append([float("nan")] * len(names))
    return pd.DataFrame(rows, columns=names).astype(float)


def descriptor_counts(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize how many descriptors are complete/usable."""
    n_constant = int((df.nunique() <= 1).sum())
    n_missing = int(df.isna().all().sum())
    return {
        "n_descriptors": int(df.shape[1]),
        "n_constant": n_constant,
        "n_all_missing": n_missing,
        "usable_descriptors": int(df.shape[1]) - n_constant - n_missing,
    }


def sanitize_descriptor_frame(
    df: pd.DataFrame, *, fill_small: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Return (usable matrix, feature names): drop constant/all-NaN cols,
    median-impute remaining NaN, z-score to unit variance."""
    frame = df.copy()
    usable = [c for c in frame.columns if frame[c].notna().any() and frame[c].nunique() > 1]
    frame = frame[usable]
    if fill_small:
        medians = frame.median()
        frame = frame.fillna(medians)
    matrix = frame.fillna(frame.median()).to_numpy(dtype=float)
    std = matrix.std(axis=0)
    matrix[:, std > 0] /= np.where(std > 0, std, 1)[np.newaxis, :]
    return matrix, usable