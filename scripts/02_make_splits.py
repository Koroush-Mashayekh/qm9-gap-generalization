"""Step 2 - make the train/validation/test splits once and save them.

Every later experiment reads splits.parquet, so all representations and models are compared on
exactly the same molecules. Columns: mol_id, random_s0..s2, scaffold_s0..s2 (values train/val/test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import pandas as pd

from qm9gap import config
from qm9gap.splits import check_no_shared_scaffolds, random_split, scaffold_split, split_column


def main():
    mols = pd.read_parquet(config.MOLECULES)
    splits = pd.DataFrame({"mol_id": mols["mol_id"]})
    rows = []
    for split_type in config.SPLIT_TYPES:
        for seed in config.SEEDS:
            col = split_column(split_type, seed)
            if split_type == "random":
                labels = random_split(len(mols), config.FRACTIONS, seed)
            else:
                labels = scaffold_split(mols["scaffold"], config.FRACTIONS, seed)
                check_no_shared_scaffolds(labels, mols["scaffold"])
            splits[col] = labels
            for part in ("train", "val", "test"):
                mask = labels == part
                rows.append({"split": col, "part": part, "molecules": int(mask.sum()),
                             "scaffolds": int(mols.loc[mask, "scaffold"].nunique()),
                             "gap_eV_mean": round(float(mols.loc[mask, config.TARGET].mean()), 3)})
    splits.to_parquet(config.SPLITS, index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "split_summary.csv", index=False)
    print(summary.pivot(index="split", columns="part", values="molecules").to_string())
    print("\nScaffolds per part:")
    print(summary.pivot(index="split", columns="part", values="scaffolds").to_string())
    print("\nNo scaffold is shared between train, val and test in any scaffold split.")


if __name__ == "__main__":
    config.make_dirs()
    main()
