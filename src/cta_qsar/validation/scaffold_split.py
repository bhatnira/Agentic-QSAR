"""Scaffold split plugin: test molecules belong to unseen Bemis-Murcko scaffolds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cta_qsar.core.interfaces import SplitPlan
from cta_qsar.validation.base import _base_split_plan


def scaffold_id(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else None


class ScaffoldSplit:
    name = "scaffold"

    def applicability(self, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        return True, "Bemis-Murcko scaffold split; tests chemical-series generalization"

    def scaffold_groups(self, smiles: list[str]) -> np.ndarray:
        ids = [scaffold_id(s) for s in smiles]
        safe = [f"_none_{i}" if g is None else g for i, g in enumerate(ids)]
        return np.asarray(safe)

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
            "scaffold",
            n_splits=n_splits,
            n_repeats=n_repeats,
            test_fraction=test_fraction,
            random_seed=random_seed,
            description=(
                "Bemis-Murcko scaffold-based folds; every test molecule has a "
                "scaffold seen during training on zero folds"
            ),
        )

PLUGINS = [ScaffoldSplit]
