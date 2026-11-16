"""Step 4 - compare molecular representations with the model chosen in step 3.

For every representation x split type x seed: train the selected model (results/model_selection/
selected_model.json) on `train` (boosters stop on `val`), score on `test`. A mean-value baseline is scored on the same test sets. Each run is saved as soon as it
finishes (results/runs/), so an interrupted run continues where it stopped.

Outputs:
    results/runs/<model>__<rep>__<split>.json          metrics of one run
    results/runs/<model>__<rep>__<split>__pred.parquet test predictions (for later error analysis)
    results/representations_runs.csv         all runs
    results/representations_summary.csv     mean ± std over seeds
    results/figures/fig1_representations.png

Examples:
    python scripts/04_compare_representations.py                 # everything (about 1-2 h)
    python scripts/04_compare_representations.py --seeds 0       # quick first look
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd

from qm9gap import config
from qm9gap.evaluate import fmt, regression_metrics, summarize
from qm9gap.features import REPRESENTATIONS, load_molecules, load_representation
from qm9gap.models import BOOSTERS, build, fit
from qm9gap.splits import invalidate_if_changed, split_column

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLIT_COLORS = {"random": "#2a78d6", "scaffold": "#eb6834"}


def load_selected():
    if not config.SELECTED_MODEL.is_file():
        raise FileNotFoundError("No selected model yet - run scripts/03_select_model.py first")
    return json.loads(config.SELECTED_MODEL.read_text())


def run_file(model, rep, col, kind="json"):
    suffix = ".json" if kind == "json" else "__pred.parquet"
    name = "mean_baseline" if rep == "mean_baseline" else model
    return config.RUNS_DIR / f"{name}__{rep}__{col}{suffix}"


def run_one(sel, rep, split_type, seed, X, y, mols, splits, force):
    col = split_column(split_type, seed)
    out = run_file(sel["model"], rep, col)
    if out.is_file() and not force:
        return json.loads(out.read_text()), "cached"

    part = splits[col].to_numpy()
    tr, va, te = part == "train", part == "val", part == "test"
    start = time.time()
    if rep == "mean_baseline":
        pred, n_trees = np.full(te.sum(), y[tr].mean()), 0
    else:
        model = build(sel["model"], sel["params"], seed)
        if sel["model"] in BOOSTERS:
            n_trees = fit(sel["model"], model, X[tr], y[tr], X[va], y[va])
        else:
            n_trees = fit(sel["model"], model, X[tr], y[tr])
        pred = model.predict(X[te])
    minutes = (time.time() - start) / 60

    result = {"model": "mean_baseline" if rep == "mean_baseline" else sel["model"], "representation": rep, "split_type": split_type, "seed": seed,
              "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
              **regression_metrics(y[te], pred), "boosting_rounds": n_trees,
              "train_minutes": round(minutes, 2)}
    out.write_text(json.dumps(result, indent=2))
    pd.DataFrame({"mol_id": mols.loc[te, "mol_id"].to_numpy(), "actual_eV": y[te],
                  "predicted_eV": pred}).to_parquet(run_file(sel["model"], rep, col, "pred"), index=False)
    return result, f"trained in {minutes:.1f} min"


def plot(summary, path, model_name):
    reps = [r for r in REPRESENTATIONS if r in set(summary["representation"])]
    x = np.arange(len(reps))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, split_type in enumerate(config.SPLIT_TYPES):
        s = summary[summary.split_type == split_type].set_index("representation").reindex(reps)
        bars = ax.bar(x + (k - 0.5) * width, s["MAE_eV_mean"], width * 0.95,
                      yerr=s["MAE_eV_std"], capsize=3, color=SPLIT_COLORS[split_type],
                      label=f"{split_type} split")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    base = summary[summary.representation == "mean_baseline"]
    if len(base):
        ax.text(0.99, 0.97, "mean baseline: " + ", ".join(
            f"{r.split_type} {r.MAE_eV_mean:.2f}" for r in base.itertuples()) + " eV",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35")
    top = summary.loc[summary.representation.isin(reps), "MAE_eV_mean"].max()
    ax.set_ylim(0, top * 1.35)
    ax.set_xticks(x, reps)
    ax.set_ylabel("Test MAE (eV), mean ± std over seeds")
    ax.set_title(f"HOMO–LUMO gap: representation vs. split ({model_name})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(reps, split_types, seeds, force):
    config.make_dirs()
    sel = load_selected()
    print(f"Model (from step 3): {sel['model']}  {sel['params']}")
    mols = load_molecules()
    splits = pd.read_parquet(config.SPLITS)
    if not splits["mol_id"].equals(mols["mol_id"]):
        raise ValueError("splits.parquet is not aligned with molecules.parquet; rerun step 2")
    invalidate_if_changed(config.RUNS_DIR, splits, ["*.json", "*.parquet"])
    y = mols[config.TARGET].to_numpy()

    results = []
    for rep in ["mean_baseline", *reps]:
        X = None if rep == "mean_baseline" else load_representation(rep, mols)[0]
        if X is not None:
            print(f"\n{rep}: {X.shape[1]:,} features")
        for split_type in split_types:
            for seed in seeds:
                res, status = run_one(sel, rep, split_type, seed, X, y, mols, splits, force)
                results.append(res)
                print(f"  {rep:14s} {split_type:9s} seed {seed} | MAE {res['MAE_eV']:.3f} eV | {status}")
        del X

    runs = pd.DataFrame(results)
    runs.to_csv(config.RESULTS_DIR / "representations_runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(config.RESULTS_DIR / "representations_summary.csv", index=False)
    plot(summary, config.FIG_DIR / "fig1_representations.png", sel["model"])

    table = summary.assign(MAE=[fmt(m, s) for m, s in zip(summary.MAE_eV_mean, summary.MAE_eV_std)],
                           R2=[fmt(m, s) for m, s in zip(summary.R2_mean, summary.R2_std)])
    print("\n" + table.pivot(index="representation", columns="split_type",
                             values=["MAE", "R2"]).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", nargs="+", default=list(REPRESENTATIONS), choices=REPRESENTATIONS)
    p.add_argument("--splits", nargs="+", default=list(config.SPLIT_TYPES), choices=config.SPLIT_TYPES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    p.add_argument("--force", action="store_true", help="retrain even if a run is cached")
    a = p.parse_args()
    main(a.reps, a.splits, a.seeds, a.force)
