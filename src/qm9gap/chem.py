"""Everything that needs RDKit: reading a molecule from its SMILES and turning it into features.

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
    # added after the first error analysis (step 7)
    "charge-separated (+/-)": "[+].[-]",      # neutral overall, with a + and a - charge (e.g. zwitterions)
    "fused cyclopropane": "[r3;R2]@[r3;R2]",  # a cyclopropane bond shared with another ring
    "acyclic C=C": "[C;!R]=[C;!R]",           # a C=C double bond outside rings
}
_SMARTS = {name: Chem.MolFromSmarts(s) for name, s in _SMARTS.items()}
_MULTIPLE_BOND_OR_AROMATIC = Chem.MolFromSmarts("[$(*=*),$(*#*),a]")
STRUCTURAL_CLASSES = ("acyclic", "aromatic ring", "3-membered ring", "4-membered ring", "3 or more rings",
                      "fully saturated", "C=O (carbonyl)", "C#N (nitrile)", "C#C (alkyne)", "C=C (alkene)",
                      "contains N", "contains O", "contains F",
                      "charge-separated (+/-)", "fused cyclopropane", "acyclic C=C")


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


def structure_problem(mol):
    """Why a QM9 molecule cannot be used ("" = no problem). QM9 molecules are single, neutral,
    closed-shell molecules, so anything else means the structure was not read correctly."""
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return "radical"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "several pieces"
    if Chem.GetFormalCharge(mol) != 0:
        return "net charge"
    return ""


def outside_qm9(mol):
    """Reasons why a molecule is unlike the QM9 molecules (empty list = it looks like QM9).

    QM9 has only neutral, closed-shell molecules with at most 9 heavy atoms of C, N, O and F,
    so predictions for anything else are extrapolations and much less reliable.
    """
    reasons = []
    other = sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - {"C", "N", "O", "F", "H"})
    if other:
        reasons.append("elements not in QM9: " + ", ".join(other))
    if mol.GetNumHeavyAtoms() > 9:
        reasons.append(f"{mol.GetNumHeavyAtoms()} heavy atoms (QM9: at most 9)")
    if Chem.GetFormalCharge(mol) != 0:
        reasons.append("charged (QM9: neutral only)")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        reasons.append("radical (QM9: closed-shell only)")
    return reasons
