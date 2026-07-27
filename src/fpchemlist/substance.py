"""Defines Substance, the single-entry building block of a Chemlist."""

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
    """A named chemical substance, backed by one or more RDKit `Mol` objects.

    A `Substance` may hold several structures at once (e.g. the fragments
    of a multi-component SMILES such as a salt, or the multiple structures
    drawn in a single CDXML file). Fingerprints are computed per structure.

    Parameters
    ----------
    name : str
        Name identifying this substance. Must not be empty.
    molecule : str or Path or tuple of Mol
        The input to parse into one or more structures. Accepted forms:

        - `tuple[Mol, ...]`: used as-is (each element must be a `Mol`).
        - `str` or `Path` pointing to an existing CDXML file: every
          structure drawn in the file is parsed.
        - `str` SMILES, optionally multi-component (dot-separated, e.g.
          `"CCO.Cl"`): each component is parsed as a separate structure.

    Raises
    ------
    TypeError
        If `molecule` is not one of the accepted types.
    SubstanceParseError
        If `molecule` could not be parsed into at least one valid `Mol`.
    """

    def __init__(self, name: str, molecule: str | Path | tuple[Mol]):
        self._fp_cache = {}
        self.name = name
        self.mols = self._parse_input(molecule)  # type: ignore

    @property
    def name(self) -> str:
        """str: Name identifying this substance."""
        return self._name

    @name.setter
    def name(self, name: str):
        require_type(name, str, "name")
        if len(name) == 0:
            raise ValueError("name must not be empty")
        self._name = name

    @property
    def mols(self) -> tuple[Mol, ...]:
        """tuple of Mol: The structure(s) making up this substance."""
        return self._mols

    @mols.setter
    def mols(self, mols: tuple[Mol, ...]):
        self._mols = validate_tuple_of_mol(mols)
        self._fp_cache = {}  # (fpgen, InvariantConfig) -> fingerprint; reset whenever mol changes

    def _parse_input(self, input: str | Path | tuple[Mol, ...]) -> tuple[Mol, ...]:
        """Parse the constructor input into a tuple of `Mol` objects.

        Parameters
        ----------
        input : str or Path or tuple of Mol
            The value passed to the constructor; see `Substance` for the
            accepted forms.

        Returns
        -------
        tuple of Mol
            One or more successfully parsed structures.

        Raises
        ------
        TypeError
            If `input` is not a `tuple`, `Path`, or `str`.
        SubstanceParseError
            If parsing did not yield any valid structures (e.g. a bad
            SMILES/CDXML, or a multi-component SMILES where any component
            failed to parse).
        """
        mols = ()

        if isinstance(input, tuple):
            mols = validate_tuple_of_mol(input, name="input")
        elif isinstance(input, (Path, str)) and os.path.isfile(input):
            mols = tuple(MolsFromCDXMLFile(str(input)))
        elif isinstance(input, str):
            mols = tuple([MolFromSmiles(substr) for substr in input.split(".")])
            try:
                validate_tuple_of_mol(mols)
            except TypeError:
                mols = ()
        else:
            raise TypeError(
                f"molecule must be type [Mol], Path or str, but got {type(input).__name__}"
            )

        if len(mols) == 0:
            raise SubstanceParseError(f"Substance '{self.name}' could not be parsed.")

        return mols

    # create chemical fingerprint of molecule
    def fingerprint(self, fpgen, config: InvariantConfig = InvariantConfig()):
        """Return the fingerprints for this substance's structure(s).

        Results are cached per `(fpgen, config)` pair, so repeated calls
        with the same generator and invariant settings are cheap. The cache
        is cleared whenever `mols` is reassigned.

        Parameters
        ----------
        fpgen : FingerprintGenerator64
            RDKit fingerprint generator used to compute the fingerprints.
        config : InvariantConfig, optional
            Atom-invariant options to apply before fingerprinting.

        Returns
        -------
        list
            One fingerprint per entry in `self.mols`, in the same order.

        Raises
        ------
        TypeError
            If `fpgen` is not a `FingerprintGenerator64`.
        """
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
        """Compute per-atom custom invariants for each structure in this substance.

        Parameters
        ----------
        config : InvariantConfig, optional
            Which element groups to merge before fingerprinting (halogens,
            Boron/Carbon/Nitrogen, or full topology-only mode).

        Returns
        -------
        list of list of int
            One invariant list per entry in `self.mols`, in the same order,
            each with one integer per atom in that structure.
        """
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
