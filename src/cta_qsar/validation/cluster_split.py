"""Structural cluster split plugin (Butina clustering on Morgan fingerprints)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cta_qsar.core.interfaces import SplitPlan
from cta_qsar.validation.base import _base_split_plan


def cluster_groups(smiles: list[str], radius: int = 2, n_bits: int = 2048, threshold: float = 0.4) -> np.ndarray:
    """Butina cluster ids with default Tanimoto cutoff 0.4."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mols = [m for m in (Chem.MolFromSmiles(str(s)) for s in smiles) if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]
    if not fps:
        return np.array([], dtype=int)
    sims = [DataStructs.BulkTanimotoSimilarity(fps[i], fps) for i in range(len(fps))]
    n = len(fps)
    for i in range(n):
        sims[i][i] = -1.0
    import heapq

    used = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        sim_row = sims[i]
        queue = [(-sim_row[j], j) for j in range(n) if not used[j] and sim_row[j] >= threshold]
        heapq.heapify(queue)
        while queue:
            neg, j = heapq.heappop(queue)
            if used[j] or -neg < threshold:
                continue
            used[j] = True
            cluster.append(j)
            row = sims[j]
            for k in range(n):
                if not used[k] and row[k] >= threshold:
                    heapq.heappush(queue, (-row[k], k))
        groups.append(cluster)
    member_of = np.empty(len(smiles), dtype=int)
    member_of.fill(-1)
    for gid, members in enumerate(groups):
        for idx in members:
            member_of[idx] = gid
    return member_of


class ClusterSplit:
    name = "cluster"

    def applicability(self, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "Butina cluster split; stricter than scaffold for analog similarity"

    def split(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        task_type: str,
        n_splits: int = 5,
        n_repeats: int = 2,
        random_seed: int = 42,
        test_fraction: float = 0.2,
    ) -> SplitPlan:
        return _base_split_plan(
            "cluster",
            n_splits=n_splits,
            n_repeats=n_repeats,
            test_fraction=test_fraction,
            random_seed=random_seed,
            description="Butina chemical cluster folds (Tanimoto >= 0.4)",
        )

PLUGINS = [ClusterSplit]
