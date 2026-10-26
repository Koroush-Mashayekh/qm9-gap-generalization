"""Everything that needs RDKit: reading the SDF and turning a molecule into features.

The same functions are used for the QM9 molecules and for any new molecule, so a prediction for a
new SMILES uses exactly the same descriptors as training. All features are computed on the
heavy-atom graph (hydrogens implicit), which is what you get from a SMILES string.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=config.MORGAN_RADIUS,
                                                     fpSize=config.MORGAN_SIZE)
_FLOAT32_MAX = np.finfo(np.float32).max


def clean_id(raw):
    """'gdb 1<TAB>157.71 ...' -> 'gdb_1'. The SDF title line carries extra numbers after a tab."""
    return str(raw).split("\t")[0].strip().replace(" ", "_")


def open_sdf(path=config.SDF):
    """Random-access reader: supplier[i] returns record i (or None if RDKit cannot parse it)."""
    return Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)


def prepare_mol(raw_mol):
    """Sanitize and remove explicit hydrogens. Returns None if RDKit rejects the molecule."""
    if raw_mol is None:
        return None
    try:
        Chem.SanitizeMol(raw_mol)
        return Chem.RemoveHs(raw_mol)
    except Exception:
        return None


def scaffold_of(mol):
    """Bemis-Murcko scaffold SMILES; acyclic molecules share the label 'ACYCLIC'."""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else "ACYCLIC"


def descriptors_of(mol):
    """All RDKit 2D descriptors as float32; failures and non-finite or huge values become NaN."""
    values = np.full(len(DESCRIPTOR_NAMES), np.nan, dtype=np.float32)
    for k, (_, fn) in enumerate(Descriptors.descList):
        try:
            v = float(fn(mol))
        except Exception:
            continue
        if np.isfinite(v) and abs(v) < _FLOAT32_MAX:
            values[k] = v
    return values


def morgan_counts_of(mol):
    """Count-based Morgan fingerprint (radius 2, 2048 bins), capped at 255 to fit in uint8."""
    counts = _MORGAN.GetCountFingerprintAsNumPy(mol)
    return np.minimum(counts, 255).astype(np.uint8)


def smiles_of(mol):
    """Canonical SMILES (heavy atoms only)."""
    return Chem.MolToSmiles(mol)


def count_radical_electrons(smiles):
    """Unpaired electrons of the molecule written in `smiles` (0 for a normal closed-shell molecule)."""
    mol = Chem.MolFromSmiles(smiles)
    return -1 if mol is None else sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())


# Structural classes used to analyse where the model makes errors (step 7). A molecule can belong
# to several classes. SMARTS patterns are matched on the heavy-atom graph.
_SMARTS = {
    "3-membered ring": "[r3]",
    "4-membered ring": "[r4]",
    "C=O (carbonyl)": "[CX3]=[OX1]",
    "C#N (nitrile)": "C#N",
    "C#C (alkyne)": "C#C",
    "C=C (alkene)": "C=C",
    "contains N": "[#7]",
    "contains O": "[#8]",
    "contains F": "[F]",
}
_SMARTS = {name: Chem.MolFromSmarts(s) for name, s in _SMARTS.items()}
_MULTIPLE_BOND_OR_AROMATIC = Chem.MolFromSmarts("[$(*=*),$(*#*),a]")
STRUCTURAL_CLASSES = ("acyclic", "aromatic ring", "3-membered ring", "4-membered ring", "3 or more rings",
                      "fully saturated", "C=O (carbonyl)", "C#N (nitrile)", "C#C (alkyne)", "C=C (alkene)",
                      "contains N", "contains O", "contains F")


def structural_classes(smiles):
    """Dict {class name: True/False} for one molecule (see STRUCTURAL_CLASSES)."""
    from rdkit.Chem import rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: False for name in STRUCTURAL_CLASSES}
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    out = {"acyclic": n_rings == 0,
           "aromatic ring": rdMolDescriptors.CalcNumAromaticRings(mol) > 0,
           "3 or more rings": n_rings >= 3,
           "fully saturated": not mol.HasSubstructMatch(_MULTIPLE_BOND_OR_AROMATIC)}
    out.update({name: mol.HasSubstructMatch(patt) for name, patt in _SMARTS.items()})
    return {name: bool(out[name]) for name in STRUCTURAL_CLASSES}


def draw_grid(smiles, legends, path, per_row=5):
    """Save a grid picture of molecules with a caption under each one."""
    from rdkit.Chem import Draw
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=(260, 220), legends=legends)
    img.save(str(path))


def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol
