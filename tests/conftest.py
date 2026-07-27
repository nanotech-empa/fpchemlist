"""Shared fixtures for fpchemlist's test suite."""

import pytest
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import MolFromSmiles

from fpchemlist import Substance
from fpchemlist.types import InvariantConfig


# ---------------------------------------------------------------------
# Simple, dependency-free building blocks (pure SMILES, no files needed)
# ---------------------------------------------------------------------


@pytest.fixture
def benzene_mol() -> Mol:
    return MolFromSmiles("c1ccccc1")


@pytest.fixture
def toluene_mol() -> Mol:
    return MolFromSmiles("Cc1ccccc1")


@pytest.fixture
def benzene() -> Substance:
    return Substance("benzene", "c1ccccc1")


@pytest.fixture
def toluene() -> Substance:
    return Substance("toluene", "Cc1ccccc1")


@pytest.fixture
def phenol() -> Substance:
    return Substance("phenol", "Oc1ccccc1")


@pytest.fixture
def sodium_acetate() -> Substance:
    """A multi-component substance (salt): two disconnected fragments."""
    return Substance("sodium_acetate", "CC(=O)[O-].[Na+]")


@pytest.fixture
def substances_dict() -> dict:
    """A plain dict of SMILES, as accepted by Chemlist(dict)."""
    return {
        "benzene": "c1ccccc1",
        "toluene": "Cc1ccccc1",
        "phenol": "Oc1ccccc1",
    }


@pytest.fixture
def fpgen():
    """A standalone Morgan fingerprint generator (radius=3, no bond types)."""
    return GetMorganGenerator(radius=3, useBondTypes=False)


@pytest.fixture
def default_config() -> InvariantConfig:
    return InvariantConfig()


# ---------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def cdxml_dir(tmp_path, request):
    """A directory containing .cdxml fixture files."""
    fixtures_dir = request.config.rootpath / "tests" / "fixtures" / "cdxml"
    dest = tmp_path / "cdxml_input"
    dest.mkdir()
    for f in fixtures_dir.glob("*.cdxml"):
        (dest / f.name).write_bytes(f.read_bytes())
    return dest


@pytest.fixture
def empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    return d
