"""Step 1 - read QM9 once and save everything the later steps need.

Outputs (in config.PROC_DIR):
    molecules.parquet    mol_id, smiles, scaffold, n_heavy, gap_eV
    descriptors.parquet  mol_id + RDKit 2D descriptors (constant columns removed)
    morgan_counts.npy    Morgan count fingerprints, rows in the same order as molecules.parquet
    prepare_report.json  counts of what was read, rejected and kept

The SDF is processed in chunks of 10,000 records saved to processed/chunks/. If Colab disconnects,
run the script again: finished chunks are skipped. First run: about 20-30 minutes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import argparse
import json
import time

import numpy as np
import pandas as pd

from qm9gap import config
from qm9gap.chem import (DESCRIPTOR_NAMES, clean_id, count_radical_electrons, descriptors_of, morgan_counts_of,
                         open_sdf, prepare_mol, scaffold_of, smiles_of)

CHUNK = 10_000


def process_chunk(supplier, start, stop):
    ids, smiles, scaffolds, n_heavy, desc, morgan, failed = [], [], [], [], [], [], []
    for i in range(start, stop):
        raw = supplier[i]
        mol_id = clean_id(raw.GetProp("_Name")) if raw is not None and raw.HasProp("_Name") \
            else f"record_{i}"
        mol = prepare_mol(raw)
        if mol is None:
            failed.append(mol_id)
            continue
        ids.append(mol_id)
        smiles.append(smiles_of(mol))
        scaffolds.append(scaffold_of(mol))
        n_heavy.append(mol.GetNumHeavyAtoms())
        desc.append(descriptors_of(mol))
        morgan.append(morgan_counts_of(mol))
    meta = pd.DataFrame({"mol_id": ids, "smiles": smiles, "scaffold": scaffolds, "n_heavy": n_heavy})
    desc = pd.DataFrame(np.vstack(desc) if desc else np.empty((0, len(DESCRIPTOR_NAMES))),
                        columns=DESCRIPTOR_NAMES, dtype=np.float32)
    morgan = np.vstack(morgan) if morgan else np.empty((0, config.MORGAN_SIZE), np.uint8)
    return meta, desc, morgan, failed


def main(force=False):
    config.make_dirs()
    supplier = open_sdf()
    n_records = len(supplier)
    print(f"SDF records: {n_records:,}")

    # ---- 1. chunks (resumable) ----
    t0 = time.time()
    for start in range(0, n_records, CHUNK):
        stop = min(start + CHUNK, n_records)
        tag = f"{start:06d}"
        files = [config.CHUNK_DIR / f"{tag}_{kind}" for kind in ("meta.parquet", "desc.parquet",
                                                                  "morgan.npy", "failed.json")]
        if all(f.is_file() for f in files) and not force:
            continue
        meta, desc, morgan, failed = process_chunk(supplier, start, stop)
        meta.to_parquet(files[0], index=False)
        desc.to_parquet(files[1], index=False)
        np.save(files[2], morgan)
        files[3].write_text(json.dumps(failed))
        print(f"records {start:>7,}-{stop:>7,} done | {(time.time() - t0) / 60:5.1f} min")

    # ---- 2. join chunks ----
    tags = [f"{s:06d}" for s in range(0, n_records, CHUNK)]
    meta = pd.concat([pd.read_parquet(config.CHUNK_DIR / f"{t}_meta.parquet") for t in tags],
                     ignore_index=True)
    desc = pd.concat([pd.read_parquet(config.CHUNK_DIR / f"{t}_desc.parquet") for t in tags],
                     ignore_index=True)
    morgan = np.vstack([np.load(config.CHUNK_DIR / f"{t}_morgan.npy") for t in tags])
    failed = sum((json.loads((config.CHUNK_DIR / f"{t}_failed.json").read_text()) for t in tags), [])

    # ---- 3. add the target (Hartree -> eV); keep only molecules that have one ----
    props = pd.read_csv(config.PROPERTIES_CSV, usecols=["mol_id", "gap"])
    props["mol_id"] = props["mol_id"].map(clean_id)
    props[config.TARGET] = props["gap"] * config.HARTREE_TO_EV
    meta = meta.merge(props[["mol_id", config.TARGET]], on="mol_id", how="left", validate="one_to_one")
    has_target = meta[config.TARGET].notna().to_numpy()

    # ---- 3b. flag molecules that RDKit read as radicals (wrong graph from the SDF) ----
    meta["radical_electrons"] = [count_radical_electrons(s) for s in meta["smiles"]]
    is_radical = (meta["radical_electrons"] != 0).to_numpy()
    radicals = meta.loc[is_radical & has_target, ["mol_id", "smiles", "radical_electrons", config.TARGET]]
    radicals.to_csv(config.PROC_DIR / "excluded_radicals.csv", index=False)

    keep = has_target & ~(is_radical if config.EXCLUDE_RADICALS else np.zeros_like(is_radical))
    meta, desc, morgan = meta[keep].reset_index(drop=True), desc[keep].reset_index(drop=True), morgan[keep]

    # ---- 4. drop descriptors that are constant over the whole dataset (uses no target info) ----
    constant = [c for c in desc.columns if desc[c].nunique(dropna=False) <= 1]
    desc = desc.drop(columns=constant)
    desc.insert(0, "mol_id", meta["mol_id"].to_numpy())

    meta.to_parquet(config.MOLECULES, index=False)
    desc.to_parquet(config.DESCRIPTORS, index=False)
    np.save(config.MORGAN, morgan)

    report = {
        "sdf_records": n_records,
        "rejected_by_rdkit": len(failed),
        "without_target": int((~has_target).sum()),
        "radicals_found": len(radicals),
        "radicals_excluded": len(radicals) if config.EXCLUDE_RADICALS else 0,
        "molecules_kept": len(meta),
        "descriptors_computed": len(DESCRIPTOR_NAMES),
        "constant_descriptors_removed": len(constant),
        "descriptors_kept": desc.shape[1] - 1,
        "descriptors_with_missing_values": int((desc.drop(columns="mol_id").isna().sum() > 0).sum()),
        "morgan_bits": morgan.shape[1],
        "unique_scaffolds": int(meta["scaffold"].nunique()),
        "acyclic_molecules": int((meta["scaffold"] == "ACYCLIC").sum()),
        "gap_eV_mean": round(float(meta[config.TARGET].mean()), 4),
        "gap_eV_std": round(float(meta[config.TARGET].std()), 4),
    }
    config.PREP_REPORT.write_text(json.dumps({**report, "constant_descriptors": constant,
                                              "rejected_ids": failed}, indent=2))
    print(pd.Series(report).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="recompute all chunks")
    main(parser.parse_args().force)
