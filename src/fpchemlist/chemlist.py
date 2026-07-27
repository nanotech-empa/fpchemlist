"""Defines Chemlist, a named, fingerprintable collection of Substances."""

from .substance import Substance
from .types import (
    SubstanceParseError,
    require_type,
    require_path,
    InvariantConfig,
    validate_tuple_of_mol,
)
from pathlib import Path
import os
import pickle
from rdkit.DataStructs import DiceSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.rdchem import Mol


def validate_substancesdict(substances, name="substances"):
    """Assert that a value is a dict mapping names to `Substance` objects.

    Parameters
    ----------
    substances : object
        The value to check.
    name : str, optional
        Human-readable name for `substances`, used in error messages.

    Raises
    ------
    TypeError
        If `substances` is not a `dict`, or if any of its values is not a
        `Substance`.
    """
    require_type(substances, dict, name)
    if not all(isinstance(val, Substance) for val in substances.values()):
        raise TypeError("All values in dictionary must be type Substance.")


class Chemlist:
    """A named collection of `Substance` objects, searchable by fingerprint similarity.

    Parameters
    ----------
    input : str or Path or dict or None, optional
        Source to populate the collection from. Accepted forms:

        - `None` (default): start with an empty collection.
        - `str`/`Path` to a directory: load every `*.cdxml` file in it.
        - `str`/`Path` to a file: unpickle a previously saved collection.
        - `dict` mapping names to `Substance`-constructible values
          (`str`, `Path`, or `tuple[Mol, ...]`): build one `Substance` per
          entry.
    radius : int, optional
        Morgan fingerprint radius used to build the fingerprint generator.
        Must be positive.
    bond_types : bool, optional
        Whether the fingerprint generator should take bond types into
        account (`useBondTypes`).

    Raises
    ------
    TypeError
        If `input` is not one of the accepted types.
    FileNotFoundError
        If `input` is a `str`/`Path` that is neither an existing directory
        nor an existing file.
    ValueError
        If `radius` is not positive.
    """

    def __init__(
        self,
        input: str | Path | dict | None = None,
        radius: int = 3,
        bond_types: bool = False,
    ):
        self.substances = self._load_input(input)
        self.bond_types = bond_types
        self.radius = radius

    # --------------------------------------------
    # getter and setter
    # --------------------------------------------

    @property
    def substances(self) -> dict:
        """dict: Mapping of substance name to `Substance` object."""
        return self._substances

    @substances.setter
    def substances(self, substances):
        validate_substancesdict(substances)
        self._substances = dict(
            substances
        )  # copy, so the caller mutating their dict later doesn't affect us

    @property
    def bond_types(self) -> bool:
        """bool: Whether the fingerprint generator uses bond types."""
        return self._bond_types

    @bond_types.setter
    def bond_types(self, bond_types):
        require_type(bond_types, bool, "bond_types")
        self._bond_types = bond_types
        self._fpgen_cache = {}  # radius/bond_types changed -> invalidate cached generator

    @property
    def radius(self) -> int:
        """int: Morgan fingerprint radius."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        require_type(radius, int, "radius")
        if radius <= 0:
            raise ValueError(f"radius must be positive, but got {radius}.")
        self._radius = radius
        self._fpgen_cache = (
            None  # radius/bond_types changed -> invalidate cached generator
        )

    def __len__(self) -> int:
        """Return the number of substances in the collection.

        Returns
        -------
        int
            Number of entries in `self.substances`.
        """
        return len(self.substances)

    def __iter__(self):
        """Iterate over the substances in the collection.

        Returns
        -------
        Iterator[Substance]
            An iterator over `self.substances.values()`.
        """
        return iter(self.substances.values())

    def __getitem__(self, name: str) -> Substance:
        """Look up a substance by name.

        Parameters
        ----------
        name : str
            Name of the substance to retrieve.

        Returns
        -------
        Substance
            The substance stored under `name`.

        Raises
        ------
        KeyError
            If `name` is not present in the collection.
        """
        return self.substances[name]

    def __repr__(self) -> str:
        """Return a short human-readable summary of the collection.

        Returns
        -------
        str
            Summary including the number of substances, radius, and
            bond_types setting.
        """
        return f"ChemList({len(self)} molecules, radius={self.radius}, bond_types={self.bond_types})"

    # --------------------------------------------
    # input parsing and updating
    # --------------------------------------------

    def _load_input(self, input) -> dict:
        """Resolve the constructor/update `input` into a substances dict.

        Parameters
        ----------
        input : str or Path or dict or None
            See `Chemlist` for the accepted forms.

        Returns
        -------
        dict
            Mapping of substance name to `Substance` object.

        Raises
        ------
        TypeError
            If `input` is not one of the accepted types.
        FileNotFoundError
            If `input` is a `str`/`Path` that is neither an existing
            directory nor an existing file.
        """
        if input is None:
            return {}

        if isinstance(input, str | Path):
            if os.path.isdir(input):
                return self._load_from_dir(input)
            elif os.path.isfile(input):
                return self.unpickle_substances(input)
            else:
                raise FileNotFoundError(f"No such file or directory: '{input}'")

        elif isinstance(input, dict):
            return self._parse_dict(input)

        raise TypeError(
            f"input must be str, Path or dict, but got {type(input).__name__}"
        )

    @staticmethod
    def unpickle_substances(file_path: str | Path) -> dict:
        """Load a substances dict previously saved with `pickle_substances`.

        Parameters
        ----------
        file_path : str or Path
            Path to the pickle file to read.

        Returns
        -------
        dict
            Mapping of substance name to `Substance` object.

        Raises
        ------
        TypeError
            If the unpickled object is not a valid substances dict.
        """
        file_path = require_path(file_path, "file_path")
        with open(file_path, "rb") as jar:
            pickled_substances = pickle.load(jar)
        require_type(pickled_substances, dict, "pickled data")

        substances = {}
        for name, mols in pickled_substances.items():
            validate_tuple_of_mol(mols)
            substances[name] = Substance(name, mols)
        return substances

    def pickle_substances(self, file_path: str | Path = "./substances.pickle"):
        """Save the current substances dict to disk via pickle.

        Parameters
        ----------
        file_path : str or Path, optional
            Destination path for the pickle file.

        Raises
        ------
        ValueError
            If the collection is empty.
        """
        if self.substances is None or len(self.substances) == 0:
            raise ValueError("No molecules found to pickle.")
        file_path = require_path(file_path, "file_path")

        substances = {}
        for name, substance in self.substances.items():
            substances[name] = substance.mols

        with open(file_path, "wb") as jar:
            pickle.dump(substances, jar, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_from_dir(dirpath: str | Path) -> dict:
        """Build a substances dict from every `*.cdxml` file in a directory.

        Files that fail to parse are skipped (with a message printed),
        rather than aborting the whole load.

        Parameters
        ----------
        dirpath : str or Path
            Directory to scan for `*.cdxml` files.

        Returns
        -------
        dict
            Mapping of substance name (the file stem) to `Substance`
            object, for each file that parsed successfully.
        """
        cdxml_files = list(Path(dirpath).glob("*.cdxml"))
        substances = {}
        for file in cdxml_files:
            name = str(file.stem)
            try:
                substances[name] = Substance(name, file)
            except SubstanceParseError as e_info:
                print(str(e_info))
        return substances

    @staticmethod
    def _parse_dict(register: dict) -> dict:
        """Build a substances dict from a dict of name -> constructor input.

        Parameters
        ----------
        register : dict
            Mapping of substance name to a value accepted by
            `Substance.__init__` (`str`, `Path`, or `tuple[Mol, ...]`).

        Returns
        -------
        dict
            Mapping of substance name to `Substance` object.
        """
        substances = {}
        for key, value in register.items():
            substances[key] = Substance(key, value)
        return substances

    def update_substances(self, input: str | Path | dict, overwrite: bool = False):
        """Merge additional substances into the collection.

        Parameters
        ----------
        input : str or Path or dict
            Source of substances to add; see `Chemlist` for the accepted
            forms.
        overwrite : bool, optional
            If True, entries in `input` take precedence over existing
            entries with the same name. If False (default), existing
            entries are kept and only genuinely new names are added.
        """
        require_type(overwrite, bool, "overwrite")

        new_substances = self._load_input(input)
        n_new = len(set(new_substances) - set(self.substances))
        print(
            f"{len(new_substances)} substance(s) loaded and {n_new} new substance(s) found."
        )
        if overwrite:
            self.substances = self.substances | new_substances
        else:
            self.substances = new_substances | self.substances

    # --------------------------------------------
    # fingerprinting and comparison
    # --------------------------------------------

    @property
    def fpgen(self):
        """FingerprintGenerator64: Cached Morgan fingerprint generator.

        Rebuilt lazily whenever `radius` or `bond_types` is changed.
        """
        if self._fpgen_cache is None:
            self._fpgen_cache = GetMorganGenerator(
                radius=self.radius, useBondTypes=self.bond_types
            )
        return self._fpgen_cache

    def compare(
        self,
        reference: Substance,
        halogen_inv: bool = True,
        bcn_inv: bool = False,
        topology: bool = False,
        fpgen=None,
    ) -> list[tuple[float, Substance]]:
        """Score every substance in the collection against a reference.

        Parameters
        ----------
        reference : Substance
            The query substance to compare against. Must contain exactly
            one structure.
        halogen_inv : bool, optional
            Do not differentiate between Iodine and Bromine.
        bcn_inv : bool, optional
            Do not differentiate between Boron, Carbon and Nitrogen.
        topology : bool, optional
            Do not differentiate between any elements (topology only).
        fpgen : FingerprintGenerator64, optional
            Fingerprint generator to use. Defaults to `self.fpgen`.

        Returns
        -------
        list of tuple of (float, Substance)
            One `(similarity, substance)` pair per substance in the
            collection, unordered. For multi-structure substances, the
            similarity is the maximum Dice similarity over that
            substance's structures against `reference`.

        Raises
        ------
        TypeError
            If `reference` is not a `Substance`.
        ValueError
            If `reference` does not contain exactly one structure.
        """
        require_type(reference, Substance, "reference")
        if len(reference.mols) != 1:
            raise ValueError("reference must contain only one structure.")

        if fpgen is None:
            fpgen = self.fpgen
        config = InvariantConfig(
            halogen_inv=halogen_inv, bcn_inv=bcn_inv, topology=topology
        )

        ref_fp = reference.fingerprint(fpgen, config)[0]

        similarity_list = []
        for substance in self.substances.values():
            substances_fp = substance.fingerprint(fpgen, config)
            coeff = max([DiceSimilarity(fp, ref_fp) for fp in substances_fp])
            similarity_list.append((coeff, substance))

        return similarity_list

    def find_closest(
        self,
        reference: Substance,
        n: int,
        halogen_inv: bool = True,
        bcn_inv: bool = False,
        topology: bool = False,
        fpgen=None,
    ) -> tuple[tuple[Mol, ...], tuple[str, ...]]:
        """Return the `n` substances most similar to a reference.

        Parameters
        ----------
        reference : Substance
            The query substance to compare against. Must contain exactly
            one structure.
        n : int
            Maximum number of results to return.
        halogen_inv : bool, optional
            Do not differentiate between Iodine and Bromine.
        bcn_inv : bool, optional
            Do not differentiate between Boron, Carbon and Nitrogen.
        topology : bool, optional
            Do not differentiate between any elements (topology only).
        fpgen : FingerprintGenerator64, optional
            Fingerprint generator to use. Defaults to `self.fpgen`.

        Returns
        -------
        tuple of tuple of Mol
            The `mols` tuples of the top matches, ordered by decreasing
            similarity. Empty if there are no matches.
        tuple of str
            Legend strings of the form `"{name}: {similarity:.03f}"`, one
            per entry in the first return value, in the same order.
        """
        similarity_list = self.compare(
            reference,
            halogen_inv=halogen_inv,
            bcn_inv=bcn_inv,
            topology=topology,
            fpgen=fpgen,
        )
        similarity_list.sort(reverse=True, key=lambda x: x[0])

        top_substances = [
            (substance.mols, f"{substance.name}: {similarity: .03f}")
            for (similarity, substance) in similarity_list[:n]
        ]

        if not top_substances:
            return (), ()
        mols, legends = tuple(zip(*top_substances))
        return mols, legends
