"""Step 0 - download QM9 (MoleculeNet / DeepChem version) into the raw-data folder.

Source: the GDB-9 archive distributed by DeepChem for MoleculeNet (Wu et al., Chem. Sci. 2018), which
contains the QM9 molecules (Ramakrishnan et al., Sci. Data 2014) as an SDF file plus a CSV with the
DFT properties (B3LYP/6-31G(2df,p); energies in Hartree).

The files are saved as qm9.sdf and qm9.sdf.csv in config.RAW_DIR. Nothing is downloaded if they exist.
"""
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import requests

from qm9gap import config

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
FILES = {"gdb9.sdf": config.SDF, "gdb9.sdf.csv": config.PROPERTIES_CSV}


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all(p.is_file() for p in FILES.values()):
        print(f"QM9 already present in {config.RAW_DIR}; nothing to do.")
        return

    archive = config.RAW_DIR / "gdb9.tar.gz"
    if not archive.is_file():
        print(f"Downloading {URL} ...")
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in FILES:
                with tar.extractfile(member) as src, open(FILES[name], "wb") as dst:
                    dst.write(src.read())

    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} was not found in {archive}")
        print(f"{path.name:14s} {path.stat().st_size / 2**20:7.1f} MB")
    archive.unlink()
    print(f"Saved in {config.RAW_DIR}")


if __name__ == "__main__":
    main()
