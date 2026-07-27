"""Rendering helpers for drawing Substances as grid images."""

from .types import validate_tuple_of_mol
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import CombineMols
from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore
from PIL import Image, ImageDraw, ImageFont
import io


def combine_mols(mols: tuple[Mol, ...]) -> Mol:
    """Merge one or more structures into a single drawable `Mol`.

    Parameters
    ----------
    mols : tuple[Mol, ...]
        One or more RDKit `Mol` objects to combine (e.g. `Substance.mols`).

    Returns
    -------
    Mol
        `mols[0]` unchanged if there is only one structure, otherwise a
        single (disconnected) `Mol` combining all of them via
        `CombineMols`.

    Raises
    ------
    TypeError
        If `mols` is not a `tuple`, or if any element of it is not a `Mol`.
    """
    validate_tuple_of_mol(mols)
    if len(mols) == 1:
        return mols[0]
    else:
        return CombineMols(*mols)


def draw_single(mol, size=(150, 150)):
    """Render a single `Mol` to a 2D image, auto-fit to the canvas.

    Parameters
    ----------
    mol : Mol
        The RDKit molecule to draw.
    size : tuple of (int, int), optional
        Width and height, in pixels, of the output image.

    Returns
    -------
    PIL.Image.Image
        The rendered structure.
    """
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)  # auto-fits to canvas
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def mols_to_grid(
    mols,
    mols_per_row=6,
    tile_size=(150, 150),
    legends=None,
    legend_height=25,
    font_size=20,
    padding=30,
    outer_margin=15,
):
    """Compose a list of structures into a single labeled grid image.

    Each entry in `mols` may itself be a sequence of structures (e.g. a
    `Substance`'s multiple fragments), in which case they are combined via
    `combine_mols` and drawn together in one tile.

    Parameters
    ----------
    mols : Sequence[Sequence[Mol] or None]
        One entry per grid tile. Each entry is a sequence of one or more
        `Mol` objects to combine and draw together; `None` entries are
        skipped.
    mols_per_row : int, optional
        Number of tiles per row.
    tile_size : tuple of (int, int), optional
        Width and height, in pixels, of each structure's drawing area
        (excluding legend space and padding).
    legends : list of str or None, optional
        Legend text to draw under each tile, in the same order as `mols`.
        If None, no legends are drawn.
    legend_height : int, optional
        Vertical space, in pixels, reserved below each structure for its
        legend text.
    font_size : int, optional
        Font size, in points, used for legend text.
    padding : int, optional
        Space, in pixels, added between neighboring tiles.
    outer_margin : int, optional
        Space, in pixels, added around the entire grid's border.

    Returns
    -------
    PIL.Image.Image
        The composed grid image.
    """
    mols = [combine_mols(m) for m in mols if m is not None]
    images = [draw_single(m, size=tile_size) for m in mols]

    if legends is None:
        legends = [""] * len(images)

    n_rows = -(-len(images) // mols_per_row)  # ceil division
    cell_w, cell_h = tile_size[0] + padding, tile_size[1] + legend_height + padding

    grid_w = cell_w * mols_per_row + outer_margin * 2
    grid_h = cell_h * n_rows + outer_margin * 2
    grid_img = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()  # fallback if font file not found

    for i, (im, legend) in enumerate(zip(images, legends)):
        col = i % mols_per_row
        row = i // mols_per_row
        x = outer_margin + col * cell_w + padding // 2
        y = outer_margin + row * cell_h + padding // 2
        grid_img.paste(im, (x, y))

        # center legend text horizontally under the structure
        bbox = draw.textbbox((0, 0), legend, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (tile_size[0] - text_w) // 2
        text_y = y + tile_size[1] + (legend_height - (bbox[3] - bbox[1])) // 2
        draw.text((text_x, text_y), legend, fill="black", font=font)

    return grid_img
