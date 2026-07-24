from .types import (
    require_type,
    validate_tuple_of_mol,
    validate_fpgen,
    InvariantConfig,
    SubstanceParseError,
)
from pathlib import Path
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdmolfiles import MolsFromCDXMLFile
from rdkit.Chem.rdchem import Mol
import os


class Substance:
    def __init__(self, name: str, molecule: str | Path | Mol):
        self._fp_cache = {}
        self.name = name
        self.mols = self._parse_input(molecule)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        require_type(name, str, "name")
        if len(name) == 0:
            raise ValueError("name must not be empty")
        self._name = name

    @property
    def mols(self) -> list[Mol]:
        return self._mols

    @mols.setter
    def mols(self, mols: list[Mol]):
        self._mols = validate_tuple_of_mol(mols)
        self._fp_cache = {}  # (fpgen, InvariantConfig) -> fingerprint; reset whenever mol changes

    @staticmethod
    def _parse_from_cdxml(filename) -> Mol | None:
        mols = MolsFromCDXMLFile(filename)
        return mols

    def _parse_input(self, input: str | Path) -> Mol:
        mols = None

        if isinstance(input, list):
            mols = validate_tuple_of_mol(input, name="input")
        elif isinstance(input, Path | str) and os.path.isfile(input):
            mols = self._parse_from_cdxml(input)
        elif isinstance(input, str):
            mols = tuple([MolFromSmiles(substr) for substr in input.split(".")])
        else:
            raise TypeError(
                f"molecule must be type list[Mol], Path or str, but got {type(input).__name__}"
            )

        if mols is None:
            raise SubstanceParseError(f"Substance '{self.name}' could not be parsed.")

        return mols

    # create chemical fingerprint of molecule
    def fingerprint(self, fpgen, config: InvariantConfig = InvariantConfig()):
        """Return the fingerprint for this molecule."""
        validate_fpgen(fpgen)
        cache_key = (fpgen, config)
        if cache_key not in self._fp_cache:
            fps = [
                fpgen.GetFingerprint(mol, customAtomInvariants=inv)
                for mol, inv in zip(self.mols, self.get_invariants(config))
            ]
            self._fp_cache[cache_key] = fps
        return self._fp_cache[cache_key]

    # definition of different invariants to be used for fingerprinting
    def get_invariants(self, config: InvariantConfig = InvariantConfig()) -> list[int]:
        invariants = []
        for mol in self.mols:
            inv = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if (
                config.bcn_inv
            ):  # do not differentiate between Boron, Carbon and Nitrogen
                inv = [6 if x in (5, 7) else x for x in inv]
            if config.halogen_inv:  # do not differentiate between Iodine and Bromine
                inv = [35 if x == 53 else x for x in inv]
            if config.topology:  # do not differentiate between any elements
                inv = [1] * len(inv)
            invariants.append(inv)
        return invariants
