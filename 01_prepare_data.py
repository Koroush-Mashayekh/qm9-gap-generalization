# Step 1: read QM9 once and save everything the later steps need.
#
# For every molecule in the SDF: SMILES, Bemis-Murcko scaffold, RDKit 2D descriptors and a Morgan
# count fingerprint. Molecules that RDKit cannot read, or reads as radicals, are removed.
#
# Outputs in processed/:
#   molecules.parquet     mol_id, smiles, scaffold, n_heavy, gap_eV
#   descriptors.parquet   mol_id + RDKit descriptors (constant ones removed)
#   morgan_counts.npy     Morgan fingerprints, same row order as molecules.parquet
#   excluded_radicals.csv and prepare_report.json
#
# The SDF is processed in chunks of 10,000 records and every chunk is saved. If Colab disconnects,
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


def read_chunk(supplier, start, stop):
    """Read SDF records start..stop-1.

    Returns a table (one row per molecule), the Morgan fingerprints (one row per molecule)
    and the ids of the records that RDKit could not read.
    """
    rows, morgan, failed = [], [], []
    for i in range(start, stop):
        raw = supplier[i]
        if raw is not None and raw.HasProp("_Name"):
            mol_id = chem.clean_id(raw.GetProp("_Name"))
        else:
            mol_id = f"record_{i}"

        mol = chem.prepare_mol(raw)  # sanitize + remove H; None if RDKit rejects it
        if mol is None:
            failed.append(mol_id)
            continue

        row = {"mol_id": mol_id,
               "smiles": chem.smiles_of(mol),
               "scaffold": chem.scaffold_of(mol),
               "n_heavy": mol.GetNumHeavyAtoms()}
        # add the 217 RDKit descriptors as extra columns: {"MolWt": 18.0, "TPSA": 25.3, ...}
        row.update(zip(chem.DESCRIPTOR_NAMES, chem.descriptors_of(mol)))
        rows.append(row)
        morgan.append(chem.morgan_counts_of(mol))

    # one row of 2,048 counts per molecule (reshape also gives the right shape if the chunk is empty)
    morgan = np.array(morgan, dtype=np.uint8).reshape(-1, config.MORGAN_SIZE)
    return pd.DataFrame(rows), morgan, failed


config.make_dirs()
supplier = chem.open_sdf()
n_records = len(supplier)
print(f"SDF records: {n_records:,}")

# 1. read the SDF in chunks (chunks that are already saved are skipped)
t0 = time.time()
chunk_files = []
for start in range(0, n_records, CHUNK):
    stop = min(start + CHUNK, n_records)
    path = config.CHUNK_DIR / f"chunk_{start:06d}.pkl"
    chunk_files.append(path)
    if not path.exists():
        pd.to_pickle(read_chunk(supplier, start, stop), path)
        print(f"records {start:>7,}-{stop:>7,} done | {(time.time() - t0) / 60:5.1f} min")

# 2. join the chunks
tables, morgans, failed = [], [], []
for path in chunk_files:
    chunk_table, chunk_morgan, chunk_failed = pd.read_pickle(path)
    tables.append(chunk_table)
    morgans.append(chunk_morgan)
    failed += chunk_failed
table = pd.concat(tables, ignore_index=True)  # all chunks as one table
morgan = np.vstack(morgans)                     # all fingerprints as one array

# 3. add the target: HOMO-LUMO gap, Hartree -> eV
props = pd.read_csv(config.PROPERTIES_CSV, usecols=["mol_id", "gap"])
props["mol_id"] = props["mol_id"].map(chem.clean_id)
props[TARGET] = props["gap"] * config.HARTREE_TO_EV
table = table.merge(props[["mol_id", TARGET]], on="mol_id", how="left", validate="one_to_one")
has_target = table[TARGET].notna()

# 4. radicals: QM9 molecules are closed-shell, so unpaired electrons mean a wrong graph in the SDF
table["radical_electrons"] = [chem.count_radical_electrons(s) for s in table["smiles"]]
is_radical = table["radical_electrons"] != 0
radicals = table.loc[is_radical & has_target, ["mol_id", "smiles", "radical_electrons", TARGET]]
radicals.to_csv(config.PROC_DIR / "excluded_radicals.csv", index=False)

# keep molecules that have a gap and are not radicals (~ means "not")
if config.EXCLUDE_RADICALS:
    keep = has_target & ~is_radical
else:
    keep = has_target
table = table[keep].reset_index(drop=True)
morgan = morgan[keep.to_numpy()]  # same rows as the table

# 5. descriptors: drop the ones that have the same value for every molecule (no target used)
desc = table[chem.DESCRIPTOR_NAMES].astype(np.float32)
constant = [c for c in desc.columns if desc[c].nunique(dropna=False) <= 1]
desc = desc.drop(columns=constant)
desc.insert(0, "mol_id", table["mol_id"])

# 6. save
molecules = table[["mol_id", "smiles", "scaffold", "n_heavy", TARGET, "radical_electrons"]]
molecules.to_parquet(config.MOLECULES, index=False)
desc.to_parquet(config.DESCRIPTORS, index=False)
np.save(config.MORGAN, morgan)

report = {
    "sdf_records": n_records,
    "rejected_by_rdkit": len(failed),
    "without_target": int((~has_target).sum()),
    "radicals_found": len(radicals),
    "radicals_excluded": len(radicals) if config.EXCLUDE_RADICALS else 0,
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
full_report = {**report, "constant_descriptors": constant, "rejected_ids": failed}
config.PREP_REPORT.write_text(json.dumps(full_report, indent=2))
print(pd.Series(report).to_string())
