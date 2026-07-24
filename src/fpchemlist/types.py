from pathlib import Path
from dataclasses import dataclass
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64


class SubstanceParseError(ValueError):
    """Raised when a Substance's input (SMILES, file, etc.) could not be parsed."""


def require_type(value, expected: type, name="value"):
    if not isinstance(value, expected):
        types = expected if isinstance(expected, tuple) else (expected,)
        want = " or ".join(t.__name__ for t in types)
        raise TypeError(f"{name} must be {want}, but got {type(value).__name__}")
    return value


def require_path(value, name: str = "value") -> Path:
    try:
        Path(value)
    except TypeError:
        require_type(value, Path, name)
        return Path()
    else:
        return Path(value)


def validate_tuple_of_mol(value, name="mols") -> tuple[Mol]:
    require_type(value, tuple, name)
    for mol in value:
        require_type(mol, Mol, f"entry in {name}")
    return value


def validate_fpgen(fpgen):
    if not isinstance(fpgen, FingerprintGenerator64):
        raise TypeError(
            f"fpgen must be type FingerprintGenerator64 "
            f"but got type {type(fpgen).__name__}."
        )


@dataclass(frozen=True)
class InvariantConfig:
    """Bundles the atom-invariant options used for fingerprinting.

    - halogen_inv: do not differentiate between Iodine and Bromine
    - bcn_inv: do not differentiate between Boron, Carbon and Nitrogen
    - topology: do not differentiate between any elements (pure connectivity)
    """

    halogen_inv: bool = True
    bcn_inv: bool = False
    topology: bool = False
