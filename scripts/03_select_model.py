"""Step 3 - choose the model with a fixed, fair protocol (see config.py, "model selection").

1. Search: each candidate tries SEARCH_N_ITER random hyperparameter configurations, trained on the
   same SEARCH_TRAIN_SIZE training molecules of the scaffold split and scored on its validation set.
2. Final check: the best configuration of each candidate is retrained on the FULL training set and
   scored on validation, for the scaffold split and the random split. Training time, prediction
   time and model size are recorded too.
3. Decision: the pre-registered rule in config.py picks the model. The test set is never touched.

Outputs (results/model_selection/):
    search_<model>.csv            every configuration tried
    final_<model>__<split>.json   full-training-set result of the best configuration
    model_selection_table.csv     one row per candidate
    selected_model.json           the chosen model, its hyperparameters and the rule used
    ../figures/fig0_model_selection.png

Everything is cached: if the run is interrupted, run it again and it continues.
First full run on Colab: roughly 1.5-2 hours (mostly Random Forest and kNN).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from qm9gap import config
from qm9gap.evaluate import regression_metrics
from qm9gap.features import load_molecules, load_representation
from qm9gap.models import BOOSTERS, CANDIDATES, SEARCH_SPACES, available, build, fit
from qm9gap.splits import invalidate_if_changed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLIT_COLORS = {"scaffold": "#eb6834", "random": "#2a78d6"}
SEED = 0


def hold_out_for_stopping(idx, seed):
    """Boosters: keep STOPPING_FRACTION of the training molecules aside to decide when to stop."""
    idx = np.random.default_rng(seed).permutation(idx)
    n_stop = int(round(config.STOPPING_FRACTION * len(idx)))
    return idx[n_stop:], idx[:n_stop]


def train_and_score(name, params, X, y, train_idx, val_idx):
    model = build(name, params, SEED)
    stop_idx = None
    if name in BOOSTERS:
        train_idx, stop_idx = hold_out_for_stopping(train_idx, SEED)
    start = time.time()
    rounds = fit(name, model, X[train_idx], y[train_idx],
                 None if stop_idx is None else X[stop_idx], None if stop_idx is None else y[stop_idx])
    fit_min = (time.time() - start) / 60
    start = time.time()
    pred = model.predict(X[val_idx])
    predict_ms = (time.time() - start) / len(val_idx) * 1e6   # ms per 1000 molecules
    return model, {**regression_metrics(y[val_idx], pred), "fit_minutes": round(fit_min, 2),
                   "predict_ms_per_1000": round(predict_ms, 1), "boosting_rounds": rounds}


def search(name, X, y, split):
    """Try SEARCH_N_ITER configurations; resumable (one CSV row per finished configuration)."""
    out = config.SELECTION_DIR / f"search_{name}.csv"
    configs = list(ParameterSampler(SEARCH_SPACES[name], n_iter=config.SEARCH_N_ITER, random_state=SEED))
    done = pd.read_csv(out) if out.is_file() else pd.DataFrame()

    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")
    sub = np.sort(np.random.default_rng(SEED).choice(train, size=min(config.SEARCH_TRAIN_SIZE, len(train)),
                                                     replace=False))
    for k, params in enumerate(configs):
        if k < len(done):
            continue
        _, res = train_and_score(name, params, X, y, sub, val)
        row = pd.DataFrame([{"config": k, "params": json.dumps(params), **res}])
        done = pd.concat([done, row], ignore_index=True)
        done.to_csv(out, index=False)
        print(f"    config {k + 1}/{len(configs)}  val MAE {res['MAE_eV']:.4f} eV  "
              f"({res['fit_minutes']:.1f} min)  {params}")
    best = done.sort_values("MAE_eV").iloc[0]
    return json.loads(best["params"]), float(best["MAE_eV"])


def final_check(name, params, X, y, splits, split_col):
    out = config.SELECTION_DIR / f"final_{name}__{split_col}.json"
    if out.is_file():
        return json.loads(out.read_text())
    split = splits[split_col].to_numpy()
    model, res = train_and_score(name, params, X, y, np.flatnonzero(split == "train"),
                                 np.flatnonzero(split == "val"))
    with tempfile.TemporaryDirectory() as tmp:          # measure size on local disk, not Drive
        path = Path(tmp) / "m.joblib"
        joblib.dump(model, path)
        res["model_MB"] = round(path.stat().st_size / 1024**2, 1)
    out.write_text(json.dumps(res, indent=2))
    return res


def decide(table):
    best_mae = table["val_MAE_scaffold"].min()
    tied = table[table["val_MAE_scaffold"] <= best_mae + config.SELECTION_TOLERANCE_EV]
    return tied.sort_values("fit_minutes_scaffold").iloc[0], tied


def plot(table, selected, path):
    t = table.sort_values("val_MAE_scaffold", ascending=False)
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.65 * len(t) + 1.6))
    for k, (split, col) in enumerate([("scaffold", "val_MAE_scaffold"), ("random", "val_MAE_random")]):
        bars = ax.barh(y + (0.5 - k) * h, t[col], h * 0.95, color=SPLIT_COLORS[split],
                       label=f"{split} split (validation)")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    labels = [f"{m}  ← selected" if m == selected else m for m in t["model"]]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, t[["val_MAE_scaffold", "val_MAE_random"]].max().max() * 1.2)
    ax.set_xlabel("Validation MAE (eV) — lower is better")
    ax.set_title("Model selection on RDKit descriptors (test set not used)", pad=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(models):
    config.make_dirs()
    mols = load_molecules()
    splits = pd.read_parquet(config.SPLITS)
    if not splits["mol_id"].equals(mols["mol_id"]):
        raise ValueError("splits.parquet is not aligned with molecules.parquet; rerun step 2")
    invalidate_if_changed(config.SELECTION_DIR, splits, ["search_*.csv", "final_*.json"])
    X, _ = load_representation(config.SELECTION_REPRESENTATION, mols)
    y = mols[config.TARGET].to_numpy()

    rows = []
    for name in models:
        if not available(name):
            print(f"\n{name}: library not installed - skipped")
            continue
        print(f"\n{name}: searching {config.SEARCH_N_ITER} configurations")
        params, search_mae = search(name, X, y, splits[config.SELECTION_SPLIT].to_numpy())
        print(f"  best configuration: {params}")
        res_s = final_check(name, params, X, y, splits, config.SELECTION_SPLIT)
        res_r = final_check(name, params, X, y, splits, config.SELECTION_CHECK_SPLIT)
        print(f"  full training set -> val MAE scaffold {res_s['MAE_eV']:.4f} | random {res_r['MAE_eV']:.4f} eV "
              f"| fit {res_s['fit_minutes']:.1f} min")
        rows.append({"model": name, "best_params": json.dumps(params), "search_val_MAE": search_mae,
                     "val_MAE_scaffold": res_s["MAE_eV"], "val_R2_scaffold": res_s["R2"],
                     "val_MAE_random": res_r["MAE_eV"], "val_R2_random": res_r["R2"],
                     "fit_minutes_scaffold": res_s["fit_minutes"],
                     "predict_ms_per_1000": res_s["predict_ms_per_1000"], "model_MB": res_s["model_MB"],
                     "boosting_rounds": res_s["boosting_rounds"]})

    table = pd.DataFrame(rows).sort_values("val_MAE_scaffold").reset_index(drop=True)
    table.to_csv(config.SELECTION_DIR / "model_selection_table.csv", index=False)
    selected, tied = decide(table)
    rule = (f"lowest validation MAE on {config.SELECTION_SPLIT}; candidates within "
            f"{config.SELECTION_TOLERANCE_EV} eV of the best are equally accurate, and the fastest to train is chosen")
    config.SELECTED_MODEL.write_text(json.dumps({
        "model": selected["model"], "params": json.loads(selected["best_params"]),
        "rule": rule, "equally_accurate": tied["model"].tolist(),
        "representation": config.SELECTION_REPRESENTATION}, indent=2))
    plot(table, selected["model"], config.FIG_DIR / "fig0_model_selection.png")

    show = table[["model", "val_MAE_scaffold", "val_MAE_random", "val_R2_scaffold",
                  "fit_minutes_scaffold", "predict_ms_per_1000", "model_MB"]]
    print("\n" + show.round(4).to_string(index=False))
    print(f"\nRule: {rule}")
    print(f"Equally accurate: {', '.join(tied['model'])}")
    print(f"SELECTED: {selected['model']}  {selected['best_params']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=CANDIDATES)
    main(p.parse_args().models)
