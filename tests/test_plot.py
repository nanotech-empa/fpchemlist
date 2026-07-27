"""Tests for fpchemlist.plot: combine_mols, draw_single, mols_to_grid."""

from PIL import Image
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import GetMolFrags

from fpchemlist.plot import combine_mols, draw_single, mols_to_grid


# ---------------------------------------------------------------------
# combine_mols
# ---------------------------------------------------------------------


class TestCombineMols:
    def test_single_mol_returned_unchanged(self, benzene_mol):
        result = combine_mols((benzene_mol,))
        assert result is benzene_mol

    def test_multiple_mols_combined_into_one(self, benzene_mol, toluene_mol):
        result = combine_mols((benzene_mol, toluene_mol))
        assert isinstance(result, Mol)
        assert (
            result.GetNumAtoms()
            == benzene_mol.GetNumAtoms() + toluene_mol.GetNumAtoms()
        )

    def test_combined_mol_has_two_fragments(self, benzene_mol, toluene_mol):
        result = combine_mols((benzene_mol, toluene_mol))
        assert len(GetMolFrags(result)) == 2


# ---------------------------------------------------------------------
# draw_single
# ---------------------------------------------------------------------


class TestDrawSingle:
    def test_returns_pil_image(self, benzene_mol):
        img = draw_single(benzene_mol)
        assert isinstance(img, Image.Image)

    def test_default_size(self, benzene_mol):
        img = draw_single(benzene_mol)
        assert img.size == (150, 150)

    def test_custom_size_is_respected(self, benzene_mol):
        img = draw_single(benzene_mol, size=(100, 150))
        assert img.size == (100, 150)


# ---------------------------------------------------------------------
# mols_to_grid
# ---------------------------------------------------------------------


class TestMolsToGrid:
    def test_returns_pil_image(self, benzene_mol, toluene_mol):
        img = mols_to_grid([(benzene_mol,), (toluene_mol,)])
        assert isinstance(img, Image.Image)

    def test_none_entries_are_skipped(self, benzene_mol):
        img_with_none = mols_to_grid([(benzene_mol,), None])
        img_without_none = mols_to_grid([(benzene_mol,)])
        assert img_with_none.size == img_without_none.size

    def test_multi_fragment_entry_drawn_in_one_tile(self, benzene_mol, toluene_mol):
        # a single grid entry with two structures should still occupy one tile
        img_pair = mols_to_grid([(benzene_mol, toluene_mol)], mols_per_row=6)
        img_single = mols_to_grid([(benzene_mol,)], mols_per_row=6)
        assert img_pair.size == img_single.size

    def test_grid_width_scales_with_mols_per_row(self, benzene_mol):
        mols = [(benzene_mol,)] * 4
        img_2_per_row = mols_to_grid(
            mols, mols_per_row=2, tile_size=(100, 100), padding=0, outer_margin=0
        )
        img_4_per_row = mols_to_grid(
            mols, mols_per_row=4, tile_size=(100, 100), padding=0, outer_margin=0
        )
        assert img_2_per_row.width < img_4_per_row.width

    def test_grid_height_scales_with_row_count(self, benzene_mol):
        # 4 mols at 2/row -> 2 rows; 4 mols at 4/row -> 1 row
        mols = [(benzene_mol,)] * 4
        img_2_per_row = mols_to_grid(
            mols, mols_per_row=2, tile_size=(100, 100), padding=0, outer_margin=0
        )
        img_4_per_row = mols_to_grid(
            mols, mols_per_row=4, tile_size=(100, 100), padding=0, outer_margin=0
        )
        assert img_2_per_row.height > img_4_per_row.height

    def test_padding_and_margin_increase_image_size(self, benzene_mol):
        mols = [(benzene_mol,)]
        tight = mols_to_grid(
            mols, tile_size=(100, 100), padding=0, outer_margin=0, legend_height=0
        )
        padded = mols_to_grid(
            mols, tile_size=(100, 100), padding=20, outer_margin=20, legend_height=0
        )
        assert padded.width > tight.width
        assert padded.height > tight.height

    def test_runs_without_legends(self, benzene_mol, toluene_mol):
        # legends=None is the default; just verify it doesn't raise
        img = mols_to_grid([(benzene_mol,), (toluene_mol,)], legends=None)
        assert isinstance(img, Image.Image)

    def test_empty_input_returns_zero_height_grid(self):
        # 0 images -> 0 rows, but width is still sized for mols_per_row
        # columns; use mols_per_row=1 and no padding/margin for a size
        # that's easy to reason about.
        img = mols_to_grid(
            [],
            mols_per_row=1,
            tile_size=(100, 100),
            legend_height=0,
            padding=0,
            outer_margin=0,
        )
        assert isinstance(img, Image.Image)
        assert img.size == (100, 0)
