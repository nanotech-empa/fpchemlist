"""Tests for fpchemlist.chemlist.Chemlist."""

import pickle

import pytest
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64

from fpchemlist import Chemlist, Substance


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


class TestConstruction:
    def test_default_constructor_is_empty(self):
        cl = Chemlist()
        assert len(cl) == 0
        assert cl.substances == {}

    def test_from_dict(self, substances_dict):
        cl = Chemlist(substances_dict)
        assert len(cl) == 3
        assert set(cl.substances) == set(substances_dict)
        assert all(isinstance(s, Substance) for s in cl)

    def test_from_directory(self, cdxml_dir):
        # requires tests/fixtures/cdxml/*.cdxml
        cl = Chemlist(cdxml_dir)
        assert len(cl) > 0

    def test_from_directory_skips_unparsable_files_without_raising(
        self, cdxml_dir, capsys
    ):
        cl = Chemlist(cdxml_dir)
        captured = capsys.readouterr()
        assert "could not be parsed" in captured.out
        # the good files should still have loaded despite the bad one
        assert len(cl) > 0

    def test_from_empty_directory(self, empty_dir):
        cl = Chemlist(empty_dir)
        assert len(cl) == 0

    def test_from_nonexistent_path_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            Chemlist(missing)

    def test_from_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError):
            Chemlist(12345)  # type: ignore

    def test_default_radius_and_bond_types(self):
        cl = Chemlist()
        assert cl.radius == 3
        assert cl.bond_types is False

    def test_custom_radius_and_bond_types(self, substances_dict):
        cl = Chemlist(substances_dict, radius=2, bond_types=True)
        assert cl.radius == 2
        assert cl.bond_types is True


# ---------------------------------------------------------------------
# substances / radius / bond_types properties
# ---------------------------------------------------------------------


class TestProperties:
    def test_substances_setter_rejects_non_dict(self):
        cl = Chemlist()
        with pytest.raises(TypeError):
            cl.substances = ["not", "a", "dict"]  # type: ignore

    def test_substances_setter_rejects_wrong_value_type(self):
        cl = Chemlist()
        with pytest.raises(TypeError):
            cl.substances = {"benzene": "c1ccccc1"}  # str, not Substance

    def test_substances_setter_copies_input_dict(self, substances_dict):
        cl = Chemlist(substances_dict)
        external = dict(cl.substances)
        external["intruder"] = Substance("intruder", "CCO")
        assert "intruder" not in cl.substances  # internal state untouched

    def test_radius_must_be_positive(self):
        cl = Chemlist()
        with pytest.raises(ValueError):
            cl.radius = 0
        with pytest.raises(ValueError):
            cl.radius = -1

    def test_radius_must_be_int(self):
        cl = Chemlist()
        with pytest.raises(TypeError):
            cl.radius = 2.5  # type: ignore

    def test_bond_types_must_be_bool(self):
        cl = Chemlist()
        with pytest.raises(TypeError):
            cl.bond_types = "yes"  # type: ignore

    def test_changing_radius_invalidates_fpgen_cache(self, substances_dict):
        cl = Chemlist(substances_dict)
        gen1 = cl.fpgen
        cl.radius = cl.radius + 1
        gen2 = cl.fpgen
        assert gen1 is not gen2

    def test_changing_bond_types_invalidates_fpgen_cache(self, substances_dict):
        cl = Chemlist(substances_dict)
        gen1 = cl.fpgen
        cl.bond_types = not cl.bond_types
        gen2 = cl.fpgen
        assert gen1 is not gen2

    def test_fpgen_is_cached_when_settings_unchanged(self, substances_dict):
        cl = Chemlist(substances_dict)
        assert cl.fpgen is cl.fpgen

    def test_fpgen_type(self, substances_dict):
        cl = Chemlist(substances_dict)
        assert isinstance(cl.fpgen, FingerprintGenerator64)


# ---------------------------------------------------------------------
# Container dunder methods
# ---------------------------------------------------------------------


class TestContainerBehavior:
    def test_len(self, substances_dict):
        assert len(Chemlist(substances_dict)) == len(substances_dict)

    def test_iter_yields_substances(self, substances_dict):
        cl = Chemlist(substances_dict)
        names = {s.name for s in cl}
        assert names == set(substances_dict)

    def test_getitem_by_name(self, substances_dict):
        cl = Chemlist(substances_dict)
        assert cl["benzene"].name == "benzene"

    def test_getitem_missing_key_raises_key_error(self, substances_dict):
        cl = Chemlist(substances_dict)
        with pytest.raises(KeyError):
            cl["does_not_exist"]

    def test_repr_contains_count_and_settings(self, substances_dict):
        cl = Chemlist(substances_dict, radius=4, bond_types=True)
        r = repr(cl)
        assert str(len(cl)) in r
        assert "4" in r
        assert "True" in r


# ---------------------------------------------------------------------
# update_substances
# ---------------------------------------------------------------------


class TestUpdateSubstances:
    def test_adds_new_entries_without_overwrite(self, substances_dict):
        cl = Chemlist({"benzene": substances_dict["benzene"]})
        cl.update_substances({"toluene": substances_dict["toluene"]})
        assert set(cl.substances) == {"benzene", "toluene"}

    def test_does_not_overwrite_existing_by_default(self):
        cl = Chemlist({"benzene": "c1ccccc1"})
        original = cl["benzene"]
        cl.update_substances({"benzene": "CCO"}, overwrite=False)
        assert cl["benzene"] is original  # unchanged

    def test_overwrites_existing_when_requested(self):
        cl = Chemlist({"benzene": "c1ccccc1"})
        original = cl["benzene"]
        cl.update_substances({"benzene": "CCO"}, overwrite=True)
        assert cl["benzene"] is not original

    def test_overwrite_must_be_bool(self, substances_dict):
        cl = Chemlist(substances_dict)
        with pytest.raises(TypeError):
            cl.update_substances({}, overwrite="yes")  # type: ignore

    def test_prints_new_substance_count(self, substances_dict, capsys):
        cl = Chemlist({"benzene": substances_dict["benzene"]})
        cl.update_substances(
            {"toluene": substances_dict["toluene"], "benzene": "c1ccccc1"}
        )
        captured = capsys.readouterr()
        assert "2" in captured.out  # two substances found
        assert "1" in captured.out  # exactly one genuinely new name ("toluene")


# ---------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------


class TestCompare:
    def test_returns_one_result_per_substance(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        results = cl.compare(benzene)
        assert len(results) == len(cl)
        assert all(
            isinstance(coeff, float) and isinstance(sub, Substance)
            for coeff, sub in results
        )

    def test_identical_structure_scores_1(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        results = dict((s.name, coeff) for coeff, s in cl.compare(benzene))
        assert results["benzene"] == pytest.approx(1.0)

    def test_reference_must_be_substance(self, substances_dict):
        cl = Chemlist(substances_dict)
        with pytest.raises(TypeError):
            cl.compare("c1ccccc1")  # type: ignore

    def test_reference_with_multiple_structures_raises(
        self, substances_dict, sodium_acetate
    ):
        cl = Chemlist(substances_dict)
        with pytest.raises(ValueError):
            cl.compare(sodium_acetate)

    def test_multi_fragment_library_entry_uses_max_similarity(self, benzene):
        # a library substance with two fragments: one identical to the
        # reference, one very different -- the identical one should win
        cl = Chemlist({"mixture": "c1ccccc1.CCCCCCCCCC"})
        results = dict((s.name, coeff) for coeff, s in cl.compare(benzene))
        assert results["mixture"] == pytest.approx(1.0)

    def test_custom_fpgen_is_respected(self, substances_dict, benzene):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        cl = Chemlist(substances_dict)
        custom_gen = GetMorganGenerator(radius=1)
        results_default = cl.compare(benzene)
        results_custom = cl.compare(benzene, fpgen=custom_gen)
        # not asserting exact values, just that a different generator can
        # be supplied and produces a valid, same-shaped result
        assert len(results_default) == len(results_custom)

    def test_invariant_options_affect_results(self):
        cl = Chemlist({"iodobenzene": "Ic1ccccc1"})
        ref = Substance("bromobenzene", "Brc1ccccc1")
        without_halogen_inv = dict(
            (s.name, c) for c, s in cl.compare(ref, halogen_inv=False)
        )
        with_halogen_inv = dict(
            (s.name, c) for c, s in cl.compare(ref, halogen_inv=True)
        )
        assert with_halogen_inv["iodobenzene"] > without_halogen_inv["iodobenzene"]


# ---------------------------------------------------------------------
# find_closest
# ---------------------------------------------------------------------


class TestFindClosest:
    def test_returns_two_equal_length_tuples(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        mols, legends = cl.find_closest(benzene, n=2)
        assert len(mols) == len(legends) == 2

    def test_results_sorted_by_decreasing_similarity(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        mols, legends = cl.find_closest(benzene, n=3)
        assert legends[0].startswith("benzene")  # exact match should rank first

    def test_n_zero_returns_empty_tuples(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        mols, legends = cl.find_closest(benzene, n=0)
        assert mols == ()
        assert legends == ()

    def test_n_larger_than_library_returns_all(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        mols, legends = cl.find_closest(benzene, n=1000)
        assert len(mols) == len(cl)

    def test_empty_chemlist_returns_empty_tuples(self, benzene):
        cl = Chemlist()
        mols, legends = cl.find_closest(benzene, n=5)
        assert mols == ()
        assert legends == ()

    def test_legend_format_includes_name_and_score(self, substances_dict, benzene):
        cl = Chemlist(substances_dict)
        _, legends = cl.find_closest(benzene, n=1)
        assert legends[0].startswith("benzene:")


# ---------------------------------------------------------------------
# Pickling round-trip
# ---------------------------------------------------------------------


class TestPickling:
    def test_pickle_empty_chemlist_raises(self, tmp_path):
        cl = Chemlist()
        with pytest.raises(ValueError):
            cl.pickle_substances(tmp_path / "empty.pickle")

    def test_round_trip_preserves_names_and_structures(self, substances_dict, tmp_path):
        cl = Chemlist(substances_dict)
        path = tmp_path / "substances.pickle"
        cl.pickle_substances(path)

        restored = Chemlist(path)
        assert set(restored.substances) == set(cl.substances)
        for name in cl.substances:
            assert (
                restored[name].mols[0].GetNumAtoms() == cl[name].mols[0].GetNumAtoms()
            )

    def test_round_trip_preserves_multi_fragment_substances(self, tmp_path):
        cl = Chemlist({"salt": "CC(=O)[O-].[Na+]"})
        path = tmp_path / "salt.pickle"
        cl.pickle_substances(path)

        restored = Chemlist(path)
        assert len(restored["salt"].mols) == 2

    def test_unpickle_rejects_malformed_file(self, tmp_path):
        path = tmp_path / "bad.pickle"
        with open(path, "wb") as f:
            pickle.dump({"benzene": "not_a_tuple_of_mol"}, f)
        with pytest.raises(TypeError):
            Chemlist(path)
