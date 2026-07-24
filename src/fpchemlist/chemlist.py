from .substance import Substance
from .types import (
    SubstanceParseError,
    require_type,
    require_path,
    InvariantConfig,
)
from pathlib import Path
import os
import pickle
from rdkit.DataStructs import DiceSimilarity
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.Chem.rdchem import Mol


def validate_substancesdict(substances, name="substances"):
    require_type(substances, dict, name)
    if not all(isinstance(val, Substance) for val in substances.values()):
        raise TypeError("All values in dictionary must be type Substance.")


class Chemlist:
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
        return self._substances

    @substances.setter
    def substances(self, substances):
        validate_substancesdict(substances)
        self._substances = dict(
            substances
        )  # copy, so the caller mutating their dict later doesn't affect us

    @property
    def bond_types(self) -> bool:
        return self._bond_types

    @bond_types.setter
    def bond_types(self, bond_types):
        require_type(bond_types, bool, "bond_types")
        self._bond_types = bond_types
        self._fpgen_cache = {}  # radius/bond_types changed -> invalidate cached generator

    @property
    def radius(self) -> int:
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
        return len(self.substances)

    def __iter__(self):
        return iter(self.substances.values())

    def __getitem__(self, name: str) -> Substance:
        return self.substances[name]

    def __repr__(self) -> str:
        return f"ChemList({len(self)} molecules, radius={self.radius}, bond_types={self.bond_types})"

    # --------------------------------------------
    # input parsing and updating
    # --------------------------------------------

    def _load_input(self, input) -> dict:
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
        file_path = require_path(file_path, "file_path")
        with open(file_path, "rb") as jar:
            substances = pickle.load(jar)
        validate_substancesdict(substances, "input")
        return substances

    def pickle_substances(self, file_path: str | Path = "./substances.pickle"):
        if self.substances is None or len(self.substances) == 0:
            raise ValueError("No molecules found to pickle.")
        file_path = require_path(file_path, "file_path")
        with open(file_path, "wb") as jar:
            pickle.dump(self.substances, jar, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_from_dir(dirpath: str | Path) -> dict:
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
        substances = {}
        for key, value in register.items():
            substances[key] = Substance(key, value)
        return substances

    def update_substances(self, input: str | Path | dict, overwrite: bool = False):
        require_type(overwrite, bool, "overwrite")

        new_substances = self._load_input(input)
        n_new = len(set(new_substances) - set(self.substances))
        plural = "s" if n_new != 1 else ""
        print(f"{n_new} new substances{plural} found.")
        if overwrite:
            self.substances = self.substances | new_substances
        else:
            self.substances = new_substances | self.substances

    # --------------------------------------------
    # fingerprinting and comparison
    # --------------------------------------------

    @property
    def fpgen(self):
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
        return tuple(zip(*top_substances))
