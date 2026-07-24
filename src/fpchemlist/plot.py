from rdkit.Chem import CombineMols
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import io


def combine_mols(mols):
    if len(mols) == 1:
        return mols[0]
    else:
        return CombineMols(*mols)


def draw_single(mol, size=(250, 250)):
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)  # auto-fits to canvas
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def mols_to_grid(
    mols,
    mols_per_row=6,
    tile_size=(250, 250),
    legends=None,
    legend_height=25,
    font_size=20,
    padding=15,
    outer_margin=15,
):
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
