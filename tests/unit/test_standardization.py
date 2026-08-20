"""Unit tests: chemical standardization (RDKit)."""

from __future__ import annotations

import pandas as pd

from cta_qsar.chemistry.standardization import MolecularStandardizer


def test_original_smiles_never_destroyed() -> None:
    df = pd.DataFrame({"SMILES": ["c1ccccc1", "CC(=O)O.[Na]", "bad-smiles-[[", None]})
    out, report = MolecularStandardizer().fit_transform(df, "SMILES")
    assert list(out["SMILES"]) == list(df["SMILES"])
    assert report.n_invalid == 2  # bad-smiles + None
    assert out["smiles_valid"].tolist() == [True, True, False, False]
    assert out["smiles_validation_error"].iloc[2] == "invalid_smiles"


def test_invalid_molecules_flagged_not_silently_removed() -> None:
    df = pd.DataFrame({"SMILES": ["CCO", "Nonsense123", "CCC"]})
    out, report = MolecularStandardizer().fit_transform(df, "SMILES")
    assert len(out) == 3  # never dropped
    assert report.n_invalid == 1
    assert report.n_valid == 2
    assert "Nonsense123" in out.iloc[:, 0].tolist()


def test_canonicalization_smiles() -> None:
    df = pd.DataFrame({"SMILES": ["c1ccccc1", "C1=CC=CC=C1"]})
    out, _ = MolecularStandardizer().fit_transform(df, "SMILES")
    canonical = out["standardized_smiles"]
    assert canonical.iloc[0] == canonical.iloc[1]  # same molecule, same canonical form


def test_salt_stripping_and_charge_neutralization() -> None:
    df = pd.DataFrame({"SMILES": ["CC(=O)O.[Na]", "c1ccccc1C(=O)[O-]"]})
    _, report = MolecularStandardizer().fit_transform(df, "SMILES")
    assert report.n_desalted >= 1
    assert report.n_neutralized >= 1


def test_transformation_log_records_changes() -> None:
    df = pd.DataFrame({"SMILES": ["CC(=O)O.[Na]"]})
    out, report = MolecularStandardizer().fit_transform(df, "SMILES")
    entry = report.entries[0]
    assert entry.valid
    assert entry.desalted
    assert "desalt" in entry.transformations
    assert out["standardized_smiles"].iloc[0] != "CC(=O)O.[Na]"


def test_empty_smiles_flagged() -> None:
    df = pd.DataFrame({"SMILES": ["  ", ""]})
    out, report = MolecularStandardizer().fit_transform(df, "SMILES")
    assert report.n_invalid == 2
    assert out["smiles_validation_error"].tolist() == ["empty_smiles", "empty_smiles"]


def test_stereochemistry_preserved() -> None:
    df = pd.DataFrame({"SMILES": ["C[C@H](Cl)CC", "C[C@@H](Cl)CC"]})
    out, _ = MolecularStandardizer().fit_transform(df, "SMILES")
    std = out["standardized_smiles"].tolist()
    assert len(set(std)) == 2  # enantiomers must not collapse