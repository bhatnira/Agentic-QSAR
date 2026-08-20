"""Characterization of chemical space: diversity, coverage, distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


def summarize_chemical_space(
    df: Any, smiles_column: str, *, max_samples: int = 2000
) -> dict[str, Any]:
    """Characterize diversity, coverage, and heavy-atom/MW/logP distributions."""
    from cta_qsar.chemistry.fingerprints import morgan_fingerprints

    smiles = list(dict.fromkeys(df[smiles_column].dropna().astype(str).tolist()))
    n = min(len(smiles), max_samples)
    subset = smiles[:n]

    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    mws, logps, heavy_atoms = [], [], []
    for s in subset:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        mws.append(Descriptors.MolWt(mol))
        logps.append(Crippen.MolLogP(mol))
        heavy_atoms.append(Lipinski.HeavyAtomCount(mol))
    summary: dict[str, Any] = {
        "n_unique_molecules": len(smiles),
        "sampled_for_descriptors": len(subset),
        "molecular_weight": _dist_summary(mws),
        "logp": _dist_summary(logps),
        "heavy_atoms": _dist_summary(heavy_atoms),
        "chemical_classes": _chemical_classes(subset),
    }
    try:
        fps = morgan_fingerprints(subset, radius=2, n_bits=2048)
        if len(fps) > 2:
            nn = NearestNeighbors(n_neighbors=2, metric="jaccard", algorithm="brute")
            nn.fit(fps)
            distances, _ = nn.kneighbors(fps)
            intra_sim = 1.0 - distances[:, 1:].mean(axis=1)
            nn_sim = _dist_summary(intra_sim.tolist())
            summary["mean_nn_similarity"] = float(intra_sim.mean())
            summary["nn_tanimoto_min"] = nn_sim["min"]
            summary["nn_tanimoto_q25"] = nn_sim["q25"]
            summary["nn_tanimoto_median"] = nn_sim["median"]
    except Exception as exc:  # noqa: BLE001
        summary["nn_similarity_error"] = str(exc)
    return summary


def _dist_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.array(values)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def _chemical_classes(smiles_subset: list[str]) -> dict[str, Any]:
    from rdkit import Chem

    classes = {"contains_aromatic": 0, "contains_ring": 0, "monocyclic": 0, "polycyclic": 0}
    n_parsed = 0
    for s in smiles_subset:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        n_parsed += 1
        rings = mol.GetRingInfo().NumRings()
        classes["contains_aromatic"] += int(any(a.GetIsAromatic() for a in mol.GetAtoms()))
        classes["contains_ring"] += int(rings > 0)
        classes["monocyclic"] += int(rings == 1)
        classes["polycyclic"] += int(rings > 1)
    return {"parsed": n_parsed, **{k: v / max(n_parsed, 1) for k, v in classes.items()}}