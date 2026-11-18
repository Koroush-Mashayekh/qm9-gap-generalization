# Step 1: read QM9 once and save everything the later steps need.
#
# Every molecule is read from its GDB-17 SMILES, so all bonds are known exactly (no guessing from 3D
# coordinates). The 3,054 "uncharacterized" molecules, whose structure changed during the DFT
# optimization, are excluded. For every other molecule: canonical SMILES, Bemis-Murcko scaffold,
# RDKit 2D descriptors and a Morgan count fingerprint.
#
# Outputs in processed/:
#   molecules.parquet     mol_id, smiles, scaffold, n_heavy, gap_eV
#   descriptors.parquet   mol_id + RDKit descriptors (constant ones removed)
#   morgan_counts.npy     Morgan fingerprints, same row order as molecules.parquet
#   prepare_report.json   counts of what was read, excluded and kept
#
# The molecules are processed in chunks of 10,000 and every chunk is saved. If Colab disconnects,
# run the script again and it continues where it stopped (about 20 min in total).
# To recompute everything, delete the folder processed/chunks.

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# let Python find our own code in src/qm9gap (the project is not installed as a package)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qm9gap import chem, config

CHUNK = 10_000
TARGET = config.TARGET  # "gap_eV"


def read_chunk(chunk):
    """Featurize the molecules of one chunk of the QM9 table.

    Returns a table (one row per usable molecule), the Morgan fingerprints (one row per molecule)
    and a dict {mol_id: problem} for the molecules that could not be used.
    """
    rows, morgan, problems = [], [], {}
    for mol_id, smiles, gap in zip(chunk["mol_id"], chunk["smiles_gdb17"], chunk[TARGET]):
        try:
            mol = chem.mol_from_smiles(smiles)
            problem = chem.structure_problem(mol)   # "" when the molecule is fine
        except ValueError:
            problem = "rejected by RDKit"
        if problem:
            problems[mol_id] = problem
            continue

        row = {"mol_id": mol_id,
               "smiles": chem.smiles_of(mol),
               "scaffold": chem.scaffold_of(mol),
               "n_heavy": mol.GetNumHeavyAtoms(),
               TARGET: gap}
        # add the 217 RDKit descriptors as extra columns: {"MolWt": 18.0, "TPSA": 25.3, ...}
        row.update(zip(chem.DESCRIPTOR_NAMES, chem.descriptors_of(mol)))
        rows.append(row)
        morgan.append(chem.morgan_counts_of(mol))

    # one row of 2,048 counts per molecule (reshape also gives the right shape if the chunk is empty)
    morgan = np.array(morgan, dtype=np.uint8).reshape(-1, config.MORGAN_SIZE)
    return pd.DataFrame(rows), morgan, problems


config.make_dirs()

# 1. the QM9 table made by step 0; the gap goes from Hartree to eV
qm9 = pd.read_csv(config.QM9_TABLE)
qm9[TARGET] = qm9["gap"] * config.HARTREE_TO_EV
print(f"QM9 molecules: {len(qm9):,}")

# 2. remove the uncharacterized molecules (first column of the list = index of the molecule)
lines = config.UNCHARACTERIZED.read_text().split("\n")[9:-2]   # skip the header and the footer
uncharacterized = {int(line.split()[0]) for line in lines if line.strip()}
is_uncharacterized = qm9["index"].isin(uncharacterized)
if config.EXCLUDE_UNCHARACTERIZED:
    qm9 = qm9[~is_uncharacterized].reset_index(drop=True)
print(f"Uncharacterized molecules excluded: {int(is_uncharacterized.sum()):,}")

# 3. featurize in chunks (chunks that are already saved are skipped)
t0 = time.time()
chunk_files = []
for start in range(0, len(qm9), CHUNK):
    path = config.CHUNK_DIR / f"chunk_{start:06d}.pkl"
    chunk_files.append(path)
    if not path.exists():
        pd.to_pickle(read_chunk(qm9.iloc[start:start + CHUNK]), path)
        print(f"molecules {start:>7,}-{min(start + CHUNK, len(qm9)):>7,} done | {(time.time() - t0) / 60:5.1f} min")

# 4. join the chunks
tables, morgans, problems = [], [], {}
for path in chunk_files:
    chunk_table, chunk_morgan, chunk_problems = pd.read_pickle(path)
    tables.append(chunk_table)
    morgans.append(chunk_morgan)
    problems.update(chunk_problems)
table = pd.concat(tables, ignore_index=True)  # all chunks as one table
morgan = np.vstack(morgans)                     # all fingerprints as one array

# 5. descriptors: drop the ones that have the same value for every molecule (no target used)
desc = table[chem.DESCRIPTOR_NAMES].astype(np.float32)
constant = [c for c in desc.columns if desc[c].nunique(dropna=False) <= 1]
desc = desc.drop(columns=constant)
desc.insert(0, "mol_id", table["mol_id"])

# 6. save
molecules = table[["mol_id", "smiles", "scaffold", "n_heavy", TARGET]]
molecules.to_parquet(config.MOLECULES, index=False)
desc.to_parquet(config.DESCRIPTORS, index=False)
np.save(config.MORGAN, morgan)

report = {
    "qm9_molecules": int(len(is_uncharacterized)),
    "uncharacterized_excluded": int(is_uncharacterized.sum()) if config.EXCLUDE_UNCHARACTERIZED else 0,
    "structure_problems": len(problems),
    "molecules_kept": len(molecules),
    "descriptors_computed": len(chem.DESCRIPTOR_NAMES),
    "constant_descriptors_removed": len(constant),
    "descriptors_kept": desc.shape[1] - 1,
    "descriptors_with_missing_values": int(desc.drop(columns="mol_id").isna().any().sum()),
    "morgan_bits": morgan.shape[1],
    "unique_scaffolds": int(molecules["scaffold"].nunique()),
    "acyclic_molecules": int((molecules["scaffold"] == "ACYCLIC").sum()),
    "gap_eV_mean": round(float(molecules[TARGET].mean()), 4),
    "gap_eV_std": round(float(molecules[TARGET].std()), 4),
}
full_report = {**report, "constant_descriptors": constant, "problem_molecules": problems}
config.PREP_REPORT.write_text(json.dumps(full_report, indent=2))
for key, value in report.items():
    print(f"{key:32s} {value:,}" if isinstance(value, int) else f"{key:32s} {value}")
