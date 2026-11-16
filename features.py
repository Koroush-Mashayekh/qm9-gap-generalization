"""Load a molecular representation as a feature matrix aligned with molecules.parquet."""
import numpy as np
import pandas as pd

from . import config

REPRESENTATIONS = ("descriptors", "morgan", "combined")


def load_molecules():
    return pd.read_parquet(config.MOLECULES)


def load_representation(name, molecules):
    """Return (X as float32 array, feature names) in the row order of `molecules`."""
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{name}'. Choose from {REPRESENTATIONS}.")

    parts, names = [], []
    if name in ("descriptors", "combined"):
        desc = pd.read_parquet(config.DESCRIPTORS)
        if not desc["mol_id"].equals(molecules["mol_id"]):
            raise ValueError("descriptors.parquet is not aligned with molecules.parquet; rerun step 1")
        cols = [c for c in desc.columns if c != "mol_id"]
        parts.append(desc[cols].to_numpy(dtype=np.float32))
        names += cols
    if name in ("morgan", "combined"):
        morgan = np.load(config.MORGAN)
        if len(morgan) != len(molecules):
            raise ValueError("morgan_counts.npy is not aligned with molecules.parquet; rerun step 1")
        parts.append(morgan.astype(np.float32))
        names += [f"morgan_{i}" for i in range(morgan.shape[1])]
    return np.hstack(parts) if len(parts) > 1 else parts[0], names
