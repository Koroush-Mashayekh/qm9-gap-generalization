"""All settings of the project in one place.

Paths can be changed without touching code through two environment variables:
    QM9_RAW_DIR   folder with qm9.sdf and qm9.sdf.csv
    QM9_WORK_DIR  folder where this project writes processed data and results
"""
import os
from pathlib import Path

# ---------------- paths ----------------
_DRIVE = Path("/content/drive/MyDrive/QM9_Molecular_Property_Prediction")
RAW_DIR = Path(os.environ.get("QM9_RAW_DIR", _DRIVE / "data/raw"))
WORK_DIR = Path(os.environ.get("QM9_WORK_DIR", _DRIVE / "v2"))

SDF = RAW_DIR / "qm9.sdf"
PROPERTIES_CSV = RAW_DIR / "qm9.sdf.csv"

PROC_DIR = WORK_DIR / "processed"
CHUNK_DIR = PROC_DIR / "chunks"
RESULTS_DIR = WORK_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIG_DIR = RESULTS_DIR / "figures"

MOLECULES = PROC_DIR / "molecules.parquet"      # mol_id, smiles, scaffold, n_heavy, gap_eV
DESCRIPTORS = PROC_DIR / "descriptors.parquet"  # mol_id + RDKit 2D descriptors
MORGAN = PROC_DIR / "morgan_counts.npy"         # rows aligned with MOLECULES
SPLITS = PROC_DIR / "splits.parquet"            # mol_id + one column per split/seed
PREP_REPORT = PROC_DIR / "prepare_report.json"

# ---------------- data quality ----------------
# All QM9 molecules are neutral closed-shell species. Some records come out of the SDF with unpaired
# electrons ([CH], [C], [N] in the SMILES), most likely because charges or bond orders are not fully
# encoded in this SDF file; their RDKit graph - and so every descriptor - does not describe the real
# molecule. They are excluded and listed in processed/excluded_radicals.csv.
EXCLUDE_RADICALS = True

# ---------------- physics ----------------
HARTREE_TO_EV = 27.211386245988  # CODATA 2018
TARGET = "gap_eV"

# ---------------- representations ----------------
MORGAN_RADIUS = 2
MORGAN_SIZE = 2048

# ---------------- splits ----------------
SPLIT_TYPES = ("random", "scaffold")
SEEDS = (0, 1, 2)
FRACTIONS = (0.8, 0.1, 0.1)  # train / validation / test

# ---------------- models ----------------
FOREST_TREES = 300            # trees in Random Forest / Extra Trees
MAX_BOOSTING_ROUNDS = 5000    # upper limit for LightGBM / XGBoost; early stopping picks the number
EARLY_STOPPING_ROUNDS = 100

# ---------------- model selection (step 3) ----------------
# Protocol, fixed BEFORE looking at any result:
#  * representation: RDKit descriptors; split: scaffold_s0 (the harder, main question)
#  * every candidate gets SEARCH_N_ITER random hyperparameter configurations, each trained on the
#    same SEARCH_TRAIN_SIZE training molecules and scored on the VALIDATION set
#  * the best configuration of each candidate is retrained on the full training set and scored on
#    validation for scaffold_s0 and random_s0
#  * the test set is never used in this step
#  * DECISION RULE: lowest validation MAE on scaffold_s0; any candidate within SELECTION_TOLERANCE_EV
#    of it counts as equally accurate, and among those the one that trains fastest is chosen.
SELECTION_REPRESENTATION = "descriptors"
SELECTION_SPLIT = "scaffold_s0"
SELECTION_CHECK_SPLIT = "random_s0"
SEARCH_N_ITER = 8
SEARCH_TRAIN_SIZE = 20_000
STOPPING_FRACTION = 0.1       # share of the training data that boosters hold out for early stopping
SELECTION_TOLERANCE_EV = 0.01
SELECTION_DIR = RESULTS_DIR / "model_selection"
SELECTED_MODEL = SELECTION_DIR / "selected_model.json"

# ---------------- learning curves (step 5) ----------------
# Training-set sizes; the full training set (from step 4) is always added as the last point.
# Smaller sets are nested random subsets of the training set (a different subset for each seed);
# validation (early stopping) and test sets are the same as in step 4.
LC_SIZES = (1_000, 3_000, 10_000, 30_000)
LC_DIR = RESULTS_DIR / "learning_curves"

# ---------------- applicability domain (step 6) ----------------
# For every test molecule: Tanimoto similarity (binary Morgan fingerprint, radius 2, 2,048 bits)
# to its MOST SIMILAR training molecule, compared with the error of the step-4 prediction.
AD_REPRESENTATION = "combined"                           # predictions of the best representation
AD_BINS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001)   # similarity bins for the error table
AD_THRESHOLDS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)  # "trust only if similarity >= t"
AD_DIR = RESULTS_DIR / "applicability"

# ---------------- chemical interpretation (step 7) ----------------
# Part A (which descriptors matter): the selected model is trained on the 190 RDKit descriptors, whose
#   names have a chemical meaning (Morgan bits do not), on the main split; SHAP values on its test set.
# Part B/C (where are the errors): the step-4 test predictions of the best representation.
CHEM_SHAP_SPLIT = "scaffold_s0"
CHEM_SHAP_SAMPLE = 2_000      # random test molecules used for SHAP (TreeSHAP over thousands of trees is slow)
CHEM_GAP_BINS = (0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0)   # eV, for systematic over/under-prediction
CHEM_N_WORST = 20                                          # molecules drawn in the "worst predictions" figure
CHEM_DIR = RESULTS_DIR / "chemistry"


def make_dirs():
    for d in [PROC_DIR, CHUNK_DIR, RESULTS_DIR, RUNS_DIR, FIG_DIR, SELECTION_DIR, LC_DIR, AD_DIR, CHEM_DIR]:
        d.mkdir(parents=True, exist_ok=True)
