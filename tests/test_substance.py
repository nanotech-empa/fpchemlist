"""Tests for fpchemlist.substance.Substance."""

import pickle
from pathlib import Path

import pytest
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from fpchemlist import Substance, InvariantConfig
from fpchemlist.types import SubstanceParseError


# ---------------------------------------------------------------------
# Construction: happy paths
# ---------------------------------------------------------------------


class TestConstruction:
    def test_from_single_smiles(self):
        s = Substance("benzene", "c1ccccc1")
        assert s.name == "benzene"
        assert len(s.mols) == 1
        assert isinstance(s.mols[0], Mol)

    def test_from_multi_component_smiles(self):
        # dot-separated SMILES -> one fragment per component
        s = Substance("salt", "CC(=O)[O-].[Na+]")
        assert len(s.mols) == 2

    def test_from_tuple_of_mol(self):
        mols = (MolFromSmiles("c1ccccc1"), MolFromSmiles("CCO"))
        s = Substance("pair", mols)
        assert s.mols == mols

    def test_from_single_element_tuple(self):
        mols = (MolFromSmiles("c1ccccc1"),)
        s = Substance("single", mols)
        assert len(s.mols) == 1

    def test_from_cdxml_file(self, cdxml_dir):
        # requires tests/fixtures/cdxml/benzene.cdxml with exactly one structure
        f = cdxml_dir / "benzene.cdxml"
        s = Substance("benzene_from_file", f)
        assert len(s.mols) == 1

    def test_from_multi_structure_cdxml_file(self, cdxml_dir):
        # requires tests/fixtures/cdxml/multi_structure.cdxml with >1 structure
        f = cdxml_dir / "sodium_acetate.cdxml"
        s = Substance("multi", f)
        assert len(s.mols) == 2

    def test_from_path_object_not_just_str(self, cdxml_dir):
        f = cdxml_dir / "benzene.cdxml"
        assert isinstance(f, Path)
        s = Substance("benzene_path", f)
        assert len(s.mols) == 1


# ---------------------------------------------------------------------
# Construction: validation / error paths
# ---------------------------------------------------------------------


class TestConstructionErrors:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            Substance("", "c1ccccc1")

    def test_non_string_name_raises(self):
        with pytest.raises(TypeError):
            Substance(123, "c1ccccc1")  # type: ignore

    def test_invalid_smiles_raises_parse_error(self):
        with pytest.raises(SubstanceParseError):
            Substance("bad", "not_a_real_smiles!!!")

    def test_partial_bad_multi_component_smiles_raises(self):
        # one valid fragment + one invalid fragment -> whole substance rejected
        with pytest.raises(SubstanceParseError):
            Substance("partially_bad", "CCO.NOTASMILES")

    def test_wrong_input_type_raises_type_error(self):
        with pytest.raises(TypeError):
            Substance("bad_type", 12345)  # type: ignore

    def test_tuple_with_non_mol_element_raises(self):
        with pytest.raises(TypeError):
            Substance("bad_tuple", (MolFromSmiles("CCO"), "not_a_mol"))

    def test_nonexistent_file_path_treated_as_smiles(self):
        # a str that isn't an existing file falls through to the SMILES
        # branch; make sure that path doesn't silently succeed on garbage
        with pytest.raises(SubstanceParseError):
            Substance("missing_file", "/no/such/file.cdxml")


# ---------------------------------------------------------------------
# mols property / setter
# ---------------------------------------------------------------------


class TestMolsProperty:
    def test_reassigning_mols_validates_tuple_of_mol(self, benzene):
        with pytest.raises(TypeError):
            benzene.mols = [MolFromSmiles("CCO")]  # list, not tuple

        with pytest.raises(TypeError):
            benzene.mols = (MolFromSmiles("CCO"), "not_a_mol")

    def test_reassigning_mols_updates_structures(self, benzene):
        new_mols = (MolFromSmiles("CCO"),)
        benzene.mols = new_mols
        assert benzene.mols == new_mols

    def test_reassigning_mols_clears_fingerprint_cache(self, benzene, fpgen):
        fp_before = benzene.fingerprint(fpgen)
        assert len(benzene._fp_cache) == 1

        benzene.mols = (MolFromSmiles("CCO"),)
        assert benzene._fp_cache == {}  # cache invalidated

        fp_after = benzene.fingerprint(fpgen)
        assert fp_after != fp_before  # genuinely different structure now


# ---------------------------------------------------------------------
# get_invariants
# ---------------------------------------------------------------------


class TestGetInvariants:
    def test_default_invariants_are_atomic_numbers(self, benzene):
        invariants = benzene.get_invariants()
        assert len(invariants) == 1
        assert invariants[0] == [
            atom.GetAtomicNum() for atom in benzene.mols[0].GetAtoms()
        ]

    def test_one_invariant_list_per_structure(self, sodium_acetate):
        invariants = sodium_acetate.get_invariants()
        assert len(invariants) == len(sodium_acetate.mols)
        for inv, mol in zip(invariants, sodium_acetate.mols):
            assert len(inv) == mol.GetNumAtoms()

    def test_halogen_inv_merges_iodine_into_bromine(self):
        s = Substance("iodobenzene", "Ic1ccccc1")
        config = InvariantConfig(halogen_inv=True)
        invariants = s.get_invariants(config)[0]
        assert 53 not in invariants  # Iodine (53) remapped
        assert 35 in invariants  # to Bromine (35)

    def test_bcn_inv_merges_boron_and_nitrogen_into_carbon(self):
        s = Substance("pyridine", "c1cbncc1")  # contains Nitrogen and Boron
        config = InvariantConfig(bcn_inv=True)
        invariants = s.get_invariants(config)[0]
        assert 7 not in invariants
        assert 5 not in invariants
        assert 6 in invariants  # remapped to Carbon

    def test_topology_flattens_all_invariants_to_one(self, phenol):
        config = InvariantConfig(topology=True)
        invariants = phenol.get_invariants(config)[0]
        assert set(invariants) == {1}
        assert len(invariants) == phenol.mols[0].GetNumAtoms()

    def test_default_config_differs_from_special_configs(self, benzene):
        default = benzene.get_invariants()
        topology = benzene.get_invariants(InvariantConfig(topology=True))
        assert default != topology


# ---------------------------------------------------------------------
# fingerprint / caching
# ---------------------------------------------------------------------


class TestFingerprint:
    def test_returns_one_fingerprint_per_structure(self, sodium_acetate, fpgen):
        fps = sodium_acetate.fingerprint(fpgen)
        assert len(fps) == len(sodium_acetate.mols)

    def test_wrong_fpgen_type_raises_type_error(self, benzene):
        with pytest.raises(TypeError):
            benzene.fingerprint("not_a_generator")

    def test_repeated_call_hits_cache(self, benzene, fpgen):
        fp1 = benzene.fingerprint(fpgen)
        fp2 = benzene.fingerprint(fpgen)
        assert fp1 is fp2  # same cached object, not recomputed

    def test_different_config_does_not_collide_in_cache(self, benzene, fpgen):
        fp_default = benzene.fingerprint(fpgen, InvariantConfig())
        fp_topology = benzene.fingerprint(fpgen, InvariantConfig(topology=True))
        assert fp_default[0].ToBase64() != fp_topology[0].ToBase64()
        assert len(benzene._fp_cache) == 2

    def test_different_generator_does_not_collide_in_cache(self, benzene):
        gen_r2 = GetMorganGenerator(radius=2)
        gen_r3 = GetMorganGenerator(radius=3)
        fp_r2 = benzene.fingerprint(gen_r2)
        fp_r3 = benzene.fingerprint(gen_r3)
        assert fp_r2[0].ToBase64() != fp_r3[0].ToBase64()
        assert len(benzene._fp_cache) == 2

    def test_cache_survives_across_repeated_calls_for_large_batch(self, fpgen):
        # loose performance/regression check: repeated fingerprinting of
        # the same substance shouldn't recompute invariants each time
        s = Substance("mol", "c1ccccc1")
        for _ in range(50):
            s.fingerprint(fpgen)
        assert len(s._fp_cache) == 1


# ---------------------------------------------------------------------
# Pickling (Substance is expected to survive round-tripping via Chemlist)
# ---------------------------------------------------------------------


class TestSubstancePickling:
    def test_mols_tuple_is_picklable(self, benzene):
        # this is the format Chemlist.pickle_substances now relies on
        data = pickle.dumps(benzene.mols)
        restored = pickle.loads(data)
        assert len(restored) == len(benzene.mols)
        assert restored[0].GetNumAtoms() == benzene.mols[0].GetNumAtoms()
