"""Shared validation helpers and small value types used across fpchemlist."""

from pathlib import Path
from dataclasses import dataclass
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64


class SubstanceParseError(ValueError):
    """Raised when a Substance's input (SMILES, file, etc.) could not be parsed."""


def require_type(value, expected: type | tuple[type, ...], name="value"):
    """Assert that a value is an instance of the expected type(s).

    Parameters
    ----------
    value : object
         The value to check.
    expected : type
        A type, or tuple of types, that `value` must be an instance of.
    name : str, optional
        Human-readable name for `value`, used in the error message.

    Returns
    -------
    object
        `value`, unchanged, if the check passes.

    Raises
    ------
    TypeError
        If `value` is not an instance of `expected`.
    """
    if not isinstance(value, expected):
        types = expected if isinstance(expected, tuple) else (expected,)
        want = " or ".join(t.__name__ for t in types)
        raise TypeError(f"{name} must be {want}, but got {type(value).__name__}")
    return value


def require_path(value, name: str = "value") -> Path:
    """Coerce a value into a `Path`, raising a clear error if it can't be.

    Parameters
    ----------
    value : object
        The value to convert, typically a `str` or `Path`.
    name : str, optional
        Human-readable name for `value`, used in the error message.

    Returns
    -------
    Path
        `value` converted to a `Path`.

    Raises
    ------
    TypeError
        If `value` is not something `Path()` can accept (e.g. not `str`,
        `bytes`, or `os.PathLike`).
    """
    try:
        Path(value)
    except TypeError:
        require_type(value, Path, name)
    return Path(value)


def validate_tuple_of_mol(value, name="mols") -> tuple[Mol, ...]:
    """Assert that a value is a tuple containing only RDKit `Mol` objects.

    Parameters
    ----------
    value : object
        The value to check.
    name : str, optional
        Human-readable name for `value`, used in error messages.

    Returns
    -------
    tuple of Mol
        `value`, unchanged, if the check passes.

    Raises
    ------
    TypeError
        If `value` is not a `tuple`, or if any element of it is not a `Mol`.
    """
    require_type(value, tuple, name)
    for mol in value:
        require_type(mol, Mol, f"entry in {name}")
    return value


def validate_fpgen(fpgen) -> FingerprintGenerator64:
    """Assert that a value is an RDKit fingerprint generator.

    Parameters
    ----------
    fpgen : object
        The value to check.

    Returns
    -------
    fpgen : FingerprintGenerator64
        fpgen, unchanged, if the check passes.

    Raises
    ------
    TypeError
        If `fpgen` is not a `FingerprintGenerator64`.
    """
    if not isinstance(fpgen, FingerprintGenerator64):
        raise TypeError(
            f"fpgen must be type FingerprintGenerator64 "
            f"but got type {type(fpgen).__name__}."
        )
    return fpgen


@dataclass(frozen=True)
class InvariantConfig:
    """Bundles the atom-invariant options used for fingerprinting.

    Parameters
    ----------
    halogen_inv : bool, default True
        If True, do not differentiate between Iodine and Bromine when
        fingerprinting.
    bcn_inv : bool, default False
        If True, do not differentiate between Boron, Carbon and Nitrogen
        when fingerprinting.
    topology : bool, default False
        If True, do not differentiate between any elements at all, i.e.
        fingerprint pure connectivity (topology) only.
    """

    halogen_inv: bool = True
    bcn_inv: bool = False
    topology: bool = False
