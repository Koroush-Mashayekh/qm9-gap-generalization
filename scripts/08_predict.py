# Step 8: predict the HOMO-LUMO gap of new molecules from their SMILES.
#
# Usage (on Colab, after steps 1-4):
#   !python scripts/08_predict.py "c1ccccc1O" "CC(=O)C" "OCC#N"
#   !python scripts/08_predict.py my_molecules.txt        (one SMILES per line)
#
# The first run trains the final model and saves it (about 15 min on Colab): LightGBM with the
# settings chosen in step 3, on RDKit descriptors + Morgan fingerprints (the best representation in
# step 4), trained on all 126,559 molecules. Later runs only load the saved model and predict.
#
# For every molecule the script also reports its similarity to the most similar training molecule
# (as in step 6) and warns when the molecule is unlike QM9, because the prediction is then much
# less reliable.

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# let Python find our own code in src/qm9gap (the project is not installed as a package)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qm9gap import chem, config
from qm9gap.features import load_molecules, load_representation
from qm9gap.models import build

# harmless message of newer LightGBM + scikit-learn versions; it does not change the predictions
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def train_final_model():
    """Train the final model on all molecules and save it."""
    selected = json.loads(config.SELECTED_MODEL.read_text())  # model and settings from step 3

    # number of trees: early stopping chose it in step 4; we use the average over all its runs
    run_files = sorted(config.RUNS_DIR.glob(f"{selected['model']}__{config.FINAL_REPRESENTATION}__*.json"))
    if not run_files:
        raise FileNotFoundError("No step-4 runs found - run scripts/04_compare_representations.py first")
    n_trees = round(np.mean([json.loads(f.read_text())["boosting_rounds"] for f in run_files]))

    molecules = load_molecules()
    X, names = load_representation(config.FINAL_REPRESENTATION, molecules)
    y = molecules[config.TARGET].to_numpy()
    print(f"Training {selected['model']} on {len(y):,} molecules, {X.shape[1]:,} features, {n_trees} trees ...")

    model = build(selected["model"], selected["params"], seed=0)
    model.set_params(n_estimators=n_trees)
    model.fit(X, y)

    bundle = {"model": model,
              "descriptor_names": [n for n in names if not n.startswith("morgan_")],
              "model_name": selected["model"], "params": selected["params"],
              "n_trees": n_trees, "n_molecules": len(y)}
    joblib.dump(bundle, config.FINAL_MODEL)
    print("Saved", config.FINAL_MODEL)
    return bundle


def features_of(mol, descriptor_names):
    """The same features as in training: the kept RDKit descriptors, then the Morgan counts."""
    all_descriptors = dict(zip(chem.DESCRIPTOR_NAMES, chem.descriptors_of(mol)))
    descriptors = np.array([all_descriptors[name] for name in descriptor_names], dtype=np.float32)
    return np.concatenate([descriptors, chem.morgan_counts_of(mol).astype(np.float32)])


def nearest_training_molecule(query_bits, train_bits, chunk=20_000):
    """Tanimoto similarity to, and index of, the most similar training molecule (for every query)."""
    Q = query_bits.astype(np.float32)
    best_sim = np.full(len(Q), -1.0)
    best_idx = np.zeros(len(Q), dtype=int)
    for start in range(0, len(train_bits), chunk):
        B = train_bits[start:start + chunk].astype(np.float32)
        common = B @ Q.T                                        # bits on in both
        union = B.sum(1)[:, None] + Q.sum(1)[None, :] - common  # bits on in either
        sim = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
        k = sim.argmax(0)
        better = sim[k, np.arange(len(Q))] > best_sim
        best_sim[better] = sim[k, np.arange(len(Q))][better]
        best_idx[better] = start + k[better]
    return best_sim, best_idx


def similarity_level(s):
    # typical test errors per similarity range, from step 6 (random / scaffold split)
    if s >= 0.6:
        return "high"     # ~0.12-0.15 eV
    if s >= 0.4:
        return "medium"   # ~0.14-0.20 eV
    return "low"          # ~0.25 eV or more


# ---- read the SMILES (from the command line or from a text file) ----
inputs = sys.argv[1:]
if len(inputs) == 1 and Path(inputs[0]).is_file():
    inputs = [line.strip() for line in Path(inputs[0]).read_text().splitlines() if line.strip()]
if not inputs:
    sys.exit('Give one or more SMILES, e.g.  python scripts/08_predict.py "c1ccccc1O" "CC(=O)C"')

config.make_dirs()
bundle = joblib.load(config.FINAL_MODEL) if config.FINAL_MODEL.is_file() else train_final_model()

# ---- turn every SMILES into features ----
rows, features, bits = [], [], []
for smiles in inputs:
    try:
        mol = chem.mol_from_smiles(smiles)
    except ValueError:
        rows.append({"input": smiles, "warning": "invalid SMILES"})
        continue
    rows.append({"input": smiles, "smiles": chem.smiles_of(mol),
                 "warning": "; ".join(chem.outside_qm9(mol))})
    features.append(features_of(mol, bundle["descriptor_names"]))
    bits.append(chem.morgan_counts_of(mol) > 0)
valid = [i for i, r in enumerate(rows) if "smiles" in r]

# ---- predict and compare with the training molecules ----
if valid:
    predictions = bundle["model"].predict(np.vstack(features))
    molecules = load_molecules()
    train_bits = np.load(config.MORGAN) > 0
    sim, idx = nearest_training_molecule(np.vstack(bits), train_bits)
    for i, pred, s, j in zip(valid, predictions, sim, idx):
        rows[i].update({"predicted_gap_eV": round(float(pred), 2),
                        "similarity": round(float(s), 2),
                        "reliability": similarity_level(s),
                        "most_similar_QM9": molecules.at[j, "smiles"],
                        "its_DFT_gap_eV": round(float(molecules.at[j, config.TARGET]), 2)})

table = pd.DataFrame(rows).fillna("")
columns = ["input", "predicted_gap_eV", "similarity", "reliability", "most_similar_QM9",
           "its_DFT_gap_eV", "warning"]
print()
print(table[[c for c in columns if c in table.columns]].to_string(index=False))
print("\nsimilarity: Tanimoto similarity (Morgan bits) to the most similar of the "
      f"{bundle['n_molecules']:,} training molecules.")
print("Typical errors in step 6: high (>= 0.6) ~0.12-0.15 eV, medium (0.4-0.6) ~0.14-0.20 eV, "
      "low (< 0.4) ~0.25 eV or more.")
print("A molecule that is itself in QM9 has similarity 1.0; it was in the training set, so its "
      "error is smaller than for a new molecule.")
