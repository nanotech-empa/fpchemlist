"""Tests for fpchemlist.types: validation helpers and small value types."""

from pathlib import Path

import pytest
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from fpchemlist.types import (
    require_type,
    require_path,
    validate_tuple_of_mol,
    validate_fpgen,
    InvariantConfig,
)


# ---------------------------------------------------------------------
# require_type
# ---------------------------------------------------------------------


class TestRequireType:
    def test_passes_through_valid_value(self):
        assert require_type("hello", str) == "hello"

    def test_passes_through_with_tuple_of_types(self):
        assert require_type(5, (int, float)) == 5
        assert require_type(5.0, (int, float)) == 5.0

    def test_raises_type_error_for_wrong_type(self):
        with pytest.raises(TypeError):
            require_type(5, str)

    def test_error_message_includes_name(self):
        with pytest.raises(TypeError, match="radius"):
            require_type("not_an_int", int, name="radius")

    def test_error_message_includes_actual_type(self):
        with pytest.raises(TypeError, match="str"):
            require_type("oops", int, name="value")

    def test_error_message_joins_multiple_expected_types(self):
        with pytest.raises(TypeError, match="int or float"):
            require_type("oops", (int, float), name="value")  # type: ignore

    def test_bool_is_not_silently_accepted_as_int(self):
        # bool is technically an int subclass in Python; documenting the
        # actual (permissive) behavior rather than assuming it's rejected
        assert require_type(True, int) is True

    def test_default_name_used_when_not_provided(self):
        with pytest.raises(TypeError, match="value"):
            require_type(5, str)


# ---------------------------------------------------------------------
# require_path
# ---------------------------------------------------------------------


class TestRequirePath:
    def test_converts_str_to_path(self):
        result = require_path("some/file.txt")
        assert isinstance(result, Path)
        assert result == Path("some/file.txt")

    def test_passes_through_existing_path_object(self):
        p = Path("some/file.txt")
        result = require_path(p)
        assert result == p

    def test_raises_type_error_for_non_path_like_value(self):
        with pytest.raises(TypeError):
            require_path(12345)

    def test_raises_type_error_for_none(self):
        with pytest.raises(TypeError):
            require_path(None)

    def test_error_message_includes_name(self):
        with pytest.raises(TypeError, match="file_path"):
            require_path(12345, name="file_path")

    def test_accepts_relative_and_absolute_paths(self):
        assert require_path("./relative") == Path("./relative")
        assert require_path("/absolute/path") == Path("/absolute/path")


# ---------------------------------------------------------------------
# validate_tuple_of_mol
# ---------------------------------------------------------------------


class TestValidateTupleOfMol:
    def test_passes_through_valid_tuple(self):
        mols = (MolFromSmiles("CCO"), MolFromSmiles("c1ccccc1"))
        assert validate_tuple_of_mol(mols) == mols

    def test_accepts_empty_tuple(self):
        # documents current behavior: an empty tuple passes validation
        # here (emptiness is instead rejected upstream, in Substance)
        assert validate_tuple_of_mol(()) == ()

    def test_accepts_single_element_tuple(self):
        mols = (MolFromSmiles("CCO"),)
        assert validate_tuple_of_mol(mols) == mols

    def test_rejects_non_tuple_container(self):
        mols = [MolFromSmiles("CCO")]  # list, not tuple
        with pytest.raises(TypeError):
            validate_tuple_of_mol(mols)

    def test_rejects_tuple_containing_non_mol(self):
        with pytest.raises(TypeError):
            validate_tuple_of_mol((MolFromSmiles("CCO"), "not_a_mol"))

    def test_rejects_tuple_containing_none(self):
        with pytest.raises(TypeError):
            validate_tuple_of_mol((MolFromSmiles("CCO"), None))

    def test_error_message_includes_name(self):
        with pytest.raises(TypeError, match="input"):
            validate_tuple_of_mol([MolFromSmiles("CCO")], name="input")


# ---------------------------------------------------------------------
# validate_fpgen
# ---------------------------------------------------------------------


class TestValidateFpgen:
    def test_accepts_valid_generator(self):
        gen = GetMorganGenerator(radius=3)
        assert validate_fpgen(gen) is gen

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError):
            validate_fpgen("not_a_generator")

    def test_rejects_none(self):
        with pytest.raises(TypeError):
            validate_fpgen(None)

    def test_error_message_mentions_expected_type(self):
        with pytest.raises(TypeError, match="FingerprintGenerator64"):
            validate_fpgen(123)


# ---------------------------------------------------------------------
# InvariantConfig
# ---------------------------------------------------------------------


class TestInvariantConfig:
    def test_defaults(self):
        config = InvariantConfig()
        assert config.halogen_inv is True
        assert config.bcn_inv is False
        assert config.topology is False

    def test_custom_values(self):
        config = InvariantConfig(halogen_inv=False, bcn_inv=True, topology=True)
        assert config.halogen_inv is False
        assert config.bcn_inv is True
        assert config.topology is True

    def test_is_frozen_immutable(self):
        config = InvariantConfig()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.topology = True  # type: ignore

    def test_is_hashable(self):
        config = InvariantConfig()
        hash(config)  # should not raise

    def test_equal_configs_are_equal_and_hash_equal(self):
        a = InvariantConfig(halogen_inv=True, bcn_inv=False, topology=False)
        b = InvariantConfig(halogen_inv=True, bcn_inv=False, topology=False)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_configs_are_not_equal(self):
        a = InvariantConfig(topology=True)
        b = InvariantConfig(topology=False)
        assert a != b

    def test_usable_as_dict_key(self):
        # this is exactly how Substance/Chemlist use it for fingerprint caching
        cache = {InvariantConfig(): "cached_value"}
        assert cache[InvariantConfig()] == "cached_value"
