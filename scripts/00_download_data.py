# Step 0: download QM9 from the official source (Ramakrishnan et al., Sci. Data 2014, figshare)
#
# The archive has one .xyz file per molecule. Besides the 3D coordinates, each file contains
#   line 2:      the computed properties (index, ..., HOMO, LUMO, gap in Hartree, ...)
#   line n + 4:  the GDB-17 SMILES (the intended structure) and the SMILES of the DFT geometry
# I collect these into one table, qm9.csv, and also save uncharacterized.txt, the list of the
# 3,054 molecules whose structure changed during the DFT optimization.

import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

# let Python find our own code in src/qm9gap (the project is not installed as a package)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qm9gap import config

raw_dir = config.RAW_DIR
raw_dir.mkdir(parents=True, exist_ok=True)

if config.QM9_TABLE.exists() and config.UNCHARACTERIZED.exists():
    print("QM9 is already downloaded.")
    sys.exit()

print("Downloading QM9 (about 80 MB) ...")
archive = raw_dir / "dsgdb9nsd.xyz.tar.bz2"
urllib.request.urlretrieve(config.XYZ_URL, archive)
urllib.request.urlretrieve(config.UNCHARACTERIZED_URL, config.UNCHARACTERIZED)

print("Reading the .xyz files ...")
rows = []
with tarfile.open(archive, "r:bz2") as tar:
    for member in tar:
        if not member.isfile():
            continue
        lines = tar.extractfile(member).read().decode().splitlines()
        n_atoms = int(lines[0])
        props = lines[1].split()          # "gdb", index, A, B, C, mu, alpha, homo, lumo, gap, ...
        smiles_gdb17, smiles_dft = lines[n_atoms + 3].split()[:2]
        rows.append({"index": int(props[1]),
                     "mol_id": f"gdb_{props[1]}",
                     "smiles_gdb17": smiles_gdb17,
                     "smiles_dft": smiles_dft,
                     "homo": float(props[7]),
                     "lumo": float(props[8]),
                     "gap": float(props[9])})   # Hartree

table = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
table.to_csv(config.QM9_TABLE, index=False)
archive.unlink()
print(f"Done: {len(table):,} molecules saved in {config.QM9_TABLE}")
