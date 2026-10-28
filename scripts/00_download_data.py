# Step 0: download QM9 (MoleculeNet version from DeepChem)
# The archive has gdb9.sdf (structures) and gdb9.sdf.csv (DFT properties, in Hartree).
# I save them as qm9.sdf and qm9.sdf.csv in the raw data folder.

import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qm9gap import config

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
raw_dir = config.RAW_DIR
raw_dir.mkdir(parents=True, exist_ok=True)

if config.SDF.exists() and config.PROPERTIES_CSV.exists():
    print("QM9 is already downloaded.")
else:
    archive = raw_dir / "gdb9.tar.gz"
    print("Downloading QM9 ...")
    urllib.request.urlretrieve(URL, archive)

    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir)

    (raw_dir / "gdb9.sdf").rename(config.SDF)
    (raw_dir / "gdb9.sdf.csv").rename(config.PROPERTIES_CSV)
    archive.unlink()
    print("Done, files saved in", raw_dir)
