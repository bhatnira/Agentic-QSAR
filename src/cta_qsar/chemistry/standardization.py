"""Molecular standardization using RDKit.

Design rules:
* the original SMILES is never destroyed --- it is kept as ``original_smiles``
* every transformation is recorded in a per-molecule log
* molecules are never silently dropped; each removal carries a reason
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rdkit import Chem

from cta_qsar.core.exceptions import ChemistryError

try:  # RDKit >= 2024 exposes rdMolStandardize under Chem.MolStandardize
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:  # older RDKit exposes it directly
    from rdkit.Chem import rdMolStandardize  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

STANDARDIZATION_VERSION = "rdkit-standardization-1.0.0"


@dataclass
class StandardizationEntry:
    """Per-molecule standardization outcome."""

    original_smiles: str
    standardized_smiles: str | None
    valid: bool
    canonical: bool
    desalted: bool
    neutralized: bool
    tautomer_stripped: bool
    transformations: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_smiles": self.original_smiles,
            "standardized_smiles": self.standardized_smiles,
            "valid": self.valid,
            "canonical": self.canonical,
            "desalted": self.desalted,
            "neutralized": self.neutralized,
            "tautomer_stripped": self.tautomer_stripped,
            "transformations": self.transformations,
            "error": self.error,
        }


@dataclass
class StandardizationReport:
    entries: list[StandardizationEntry] = field(default_factory=list)
    n_valid: int = 0
    n_invalid: int = 0
    n_changed: int = 0
    n_desalted: int = 0
    n_neutralized: int = 0
    n_tautomer_stripped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STANDARDIZATION_VERSION,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "n_changed": self.n_changed,
            "n_desalted": self.n_desalted,
            "n_neutralized": self.n_neutralized,
            "n_tautomer_stripped": self.n_tautomer_stripped,
            "invalid_examples": [
                e.original_smiles for e in self.entries[:5] if not e.valid
            ],
        }


class MolecularStandardizer:
    """Standardizes a SMILES column with full provenance."""

    def __init__(
        self,
        *,
        canonical: bool = True,
        neutralise: bool = True,
        desalt: bool = True,
        remove_tautomers: bool = False,
    ) -> None:
        self.canonical = canonical
        self.neutralise = neutralise
        self.desalt = desalt
        self.remove_tautomers = remove_tautomers
        self._charger = rdMolStandardize.Uncharger()
        if remove_tautomers:
            self._tautomerizer = rdMolStandardize.TautomerEnumerator()

    @staticmethod
    def parse_smiles(smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ChemistryError(f"Unparseable SMILES: {smiles!r}")
        return mol

    def standardize_one(self, smiles: str) -> StandardizationEntry:
        entry = StandardizationEntry(
            original_smiles=smiles,
            standardized_smiles=smiles,
            valid=False,
            canonical=False,
            desalted=False,
            neutralized=False,
            tautomer_stripped=False,
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            entry.error = "invalid_smiles"
            return entry
        entry.valid = True
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:  # noqa: BLE001
            entry.error = f"sanitization_failed: {exc}"
            return entry

        transforms: list[str] = []
        if self.desalt:
            stripped = rdMolStandardize.FragmentRemover().remove(mol)
            if stripped and stripped.GetNumAtoms() < mol.GetNumAtoms():
                mol = stripped
                entry.desalted = True
                transforms.append("desalt")

        if self.neutralise:
            before_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            neutralized = self._charger.uncharge(mol)
            after_charge = sum(a.GetFormalCharge() for a in neutralized.GetAtoms())
            entry.neutralized = before_charge != after_charge
            mol = neutralized
            if entry.neutralized:
                transforms.append("neutralise")

        if self.remove_tautomers:
            taut = self._tautomerizer.Canonicalize(mol)
            if taut is not None:
                entry.tautomer_stripped = taut.GetSmiles() != Chem.MolToSmiles(mol)
                mol = taut
                if entry.tautomer_stripped:
                    transforms.append("canonical_tautomer")

        standardized = Chem.MolToSmiles(mol)
        if self.canonical:
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(standardized))
            if canonical != standardized:
                transforms.append("canonicalize")
            standardized = canonical
        entry.standardized_smiles = standardized
        entry.canonical = True
        entry.transformations = transforms
        return entry

    def fit(self, df: pd.DataFrame, smiles_column: str) -> MolecularStandardizer:
        """Validate the column exists; no state needed."""
        if smiles_column not in df.columns:
            raise ChemistryError(f"SMILES column {smiles_column!r} not in dataset")
        return self

    def transform(self, df: pd.DataFrame, smiles_column: str) -> tuple[pd.DataFrame, StandardizationReport]:
        """Return (augmented df, report)."""
        report = StandardizationReport()
        rows: list[pd.DataFrame] = []
        transformed_cols = "standardized_smiles", "smiles_valid", "smiles_validation_error"
        for idx, row in df.iterrows():
            raw = row[smiles_column]
            if not isinstance(raw, str) or not raw.strip():
                entry = StandardizationEntry(
                    original_smiles=str(raw), standardized_smiles=None,
                    valid=False, canonical=False, desalted=False,
                    neutralized=False, tautomer_stripped=False,
                    error="empty_smiles",
                )
            else:
                entry = self.standardize_one(raw.strip())
            report.entries.append(entry)
            new = pd.DataFrame(
                {
                    transformed_cols[0]: [entry.standardized_smiles],
                    transformed_cols[1]: [entry.valid],
                    transformed_cols[2]: [entry.error],
                },
                index=[idx],
            )
            augmented = pd.concat([row.to_frame().T, new], axis=1)
            rows.append(augmented)
        out = pd.concat(rows)
        out = out.reset_index(drop=True)
        valid_mask = out["smiles_valid"]
        report.n_valid = int(valid_mask.sum())
        report.n_invalid = len(out) - report.n_valid
        report.n_changed = int(
            (out[smiles_column].astype(str) != out["standardized_smiles"]).sum()
        )
        report.n_desalted = sum(1 for e in report.entries if e.desalted)
        report.n_neutralized = sum(1 for e in report.entries if e.neutralized)
        report.n_tautomer_stripped = sum(1 for e in report.entries if e.tautomer_stripped)
        return out, report

    def fit_transform(
        self, df: pd.DataFrame, smiles_column: str
    ) -> tuple[pd.DataFrame, StandardizationReport]:
        self.fit(df, smiles_column)
        return self.transform(df, smiles_column)