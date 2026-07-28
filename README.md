# fpchemlist
Tools to make a list of substances searchable by fingerprinting their molecular structures.


## Installation
Install via pip directly from GitHub:
```bash
pip install git+https://github.com/nanotech-empa/fpchemlist
```

## Usage

### Substance: A Single Chemical Entry
A single substance can be instantiated by
```python
import fpchemlist as fpc
substance = fpc.Substance(name, input)
```
with ``name`` being a string and ``input`` either a path to a `*.cdxml` file, a SMILES code, or a tuple of already-parsed `rdkit.Chem.rdchem.Mol` objects. A substance can consist of several fragments — e.g. a multi-component SMILES such as `"CC(=O)[O-].[Na+]"`, or a `*.cdxml` file with more than one structure drawn in it — stored as a tuple of one or more ``rdkit.Chem.rdchem.Mol`` objects.

### Chemlist: A Collection of Substances
Create a collection of substances via
```python
import fpchemlist as fpc
collection = fpc.Chemlist(input)
```
with ``input`` being either a path to a directory containing a collection of `*.cdxml` files, or a previously created pickle file. Alternatively, it can be a dictionary whose values are anything accepted by ``Substance`` (``Substance`` objects, SMILES codes, or paths to `*.cdxml` files). Leaving ``input`` unset creates an empty collection.

To update an existing collection use:
```python
collection.update_substances(input, overwrite=False)
```
If ``overwrite`` is set to ``True``, new substances with the same name as already registered substances will overwrite the old entries. If ``False`` (default), only new substances with a name not already in use will be added.

An existing collection can be saved as a pickle file:
```python
collection.pickle_substances(path_to_file)
```

### Fingerprinting and Searching
The main purpose of ``fpchemlist`` is to make a collection of substances searchable for identical or similar structures. To do this, a reference structure has to be defined by creating a reference substance that contains only a single molecule/fragment.
```python
reference = fpc.Substance(ref_name, ref_input)
reference.mols[0]  # display the structure (in a Jupyter notebook)
```

To obtain the ``n`` most similar substances in the collection, use:
```python
similar_mols, legends = collection.find_closest(reference, n, halogen_inv=True, bcn_inv=True, topology=False)
# halogen_inv=True: treat Bromine and Iodine as the same element (default: True)
# bcn_inv=True: treat Boron, Carbon and Nitrogen as the same element (default: False)
# topology=True: treat all atoms as the same element, i.e. compare connectivity only (default: False)
```
``similar_mols`` is a tuple of the matched substances' structures (each itself a tuple of one or more ``Mol`` objects), ready to pass into ``mols_to_grid`` below.

The search results can be displayed visually by calling
```python
grid_img = fpc.mols_to_grid(similar_mols, mols_per_row=6, legends=legends)
grid_img  # a PIL Image, displayed automatically in a Jupyter notebook
```
