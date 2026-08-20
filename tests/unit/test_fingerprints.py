"""Unit tests: fingerprint calculators against modern RDKit APIs."""

from __future__ import annotations

import numpy as np

from cta_qsar.chemistry.fingerprints import (
    morgan_fingerprints,
    torsion_fingerprints,
)


def test_torsion_fingerprints_produce_bits_on_modern_rdkit() -> None:
    """Regression: the bit-vect torsion API was removed in RDKit >= 2024.09;
    the IntVect-based fallback must still yield a populated width-n_bits matrix."""
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin: 4+-atom paths exist
        "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",  # amigdalin
        "Cc1occc1C(=O)Nc2ccccc2",  # fenfuram
        "CCO",  # ethanol: no torsion paths, allowed to stay empty
    ]
    X = torsion_fingerprints(smiles, n_bits=2048)
    assert X.shape == (4, 2048)
    assert X.dtype == np.uint8
    assert int(X.sum(axis=1)[0]) > 0  # aspirin populates bits
    assert int(X[3].sum()) == 0  # ethanol legitimately empty


def test_morgan_fingerprints_shape_and_content() -> None:
    X = morgan_fingerprints(["c1ccccc1", "CCO"], n_bits=2048)
    assert X.shape == (2, 2048)
    assert int(X.sum()) > 0
    assert not np.array_equal(X[0], X[1])