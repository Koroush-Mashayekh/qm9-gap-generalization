"""Step 5 - learning curves: how does the test error fall as the training set grows?

For every representation x split type x seed, the selected model (step 3) is trained on nested random
subsets of the training set (config.LC_SIZES molecules) and scored on the same test set as in step 4.
The full-training-set point is taken from step 4 (trained again only if it is missing).

For each curve a power law  MAE = a * n^(-b)  is fitted on log-log axes. The exponent b says how fast
the error falls with more data: if b is smaller for the scaffold split, more data helps less for
molecules with unseen ring systems.

Outputs:
    results/learning_curves/<model>__<rep>__<split>__n<size>.json   one run
    results/learning_curve_runs.csv       all runs
    results/learning_curve_summary.csv    mean ± std over seeds per size
    results/learning_curve_slopes.csv     fitted exponent b per representation and split
    results/figures/fig2_learning_curves.png

Examples:
    python scripts/05_learning_curves.py               # everything (about 1.5 h on Colab)
    python scripts/05_learning_curves.py --seeds 0     # quick first look
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import argparse
import json
import time

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

from qm9gap import config
from qm9gap.evaluate import regression_metrics
from qm9gap.features import REPRESENTATIONS, load_molecules, load_representation
from qm9gap.models import BOOSTERS, build, fit
from qm9gap.splits import invalidate_if_changed, split_column

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REP_COLORS = {"combined": "#2a78d6", "descriptors": "#eb6834", "morgan": "#1baf7a"}
REP_LABELS = {"combined": "Descriptors + Morgan", "descriptors": "RDKit descriptors", "morgan": "Morgan fingerprint"}


def load_selected():
    if not config.SELECTED_MODEL.is_file():
        raise FileNotFoundError("No selected model yet - run scripts/03_select_model.py first")
    return json.loads(config.SELECTED_MODEL.read_text())


def train_subset(n_train, seed, train_idx):
    """First n_train molecules of a seed-specific shuffle of the training set (so subsets are nested)."""
    order = np.random.default_rng(1000 + seed).permutation(train_idx)
    return np.sort(order[:n_train])


def run_one(sel, rep, split_type, seed, n_train, X, y, splits, force):
    col = split_column(split_type, seed)
    part = splits[col].to_numpy()
    train_idx = np.flatnonzero(part == "train")
    full = n_train >= len(train_idx)

    # the full training set was already trained in step 4: reuse it
    step4 = config.RUNS_DIR / f"{sel['model']}__{rep}__{col}.json"
    if full and step4.is_file() and not force:
        r = json.loads(step4.read_text())
        return {"representation": rep, "split_type": split_type, "seed": seed, "point": "full",
                "n_train": len(train_idx),
                "MAE_eV": r["MAE_eV"], "RMSE_eV": r["RMSE_eV"], "R2": r["R2"],
                "train_minutes": r["train_minutes"]}, "from step 4"

    out = config.LC_DIR / f"{sel['model']}__{rep}__{col}__n{n_train}.json"
    if out.is_file() and not force:
        return json.loads(out.read_text()), "cached"

    tr = train_idx if full else train_subset(n_train, seed, train_idx)
    va, te = part == "val", part == "test"
    start = time.time()
    model = build(sel["model"], sel["params"], seed)
    if sel["model"] in BOOSTERS:
        fit(sel["model"], model, X[tr], y[tr], X[va], y[va])
    else:
        fit(sel["model"], model, X[tr], y[tr])
    pred = model.predict(X[te])
    minutes = (time.time() - start) / 60
    result = {"representation": rep, "split_type": split_type, "seed": seed,
              "point": "full" if full else str(n_train), "n_train": len(tr),
              **regression_metrics(y[te], pred), "train_minutes": round(minutes, 2)}
    out.write_text(json.dumps(result, indent=2))
    return result, f"trained in {minutes:.1f} min"


def fit_power_law(summary):
    """log(MAE) = log(a) - b log(n), fitted on the mean MAE of each size."""
    rows = []
    for (rep, split_type), g in summary.groupby(["representation", "split_type"]):
        if len(g) < 2:
            continue
        slope, intercept = np.polyfit(np.log10(g["n_train"]), np.log10(g["MAE_mean"]), 1)
        rows.append({"representation": rep, "split_type": split_type, "exponent_b": round(-slope, 3),
                     "MAE_at_1000": round(10 ** (intercept + slope * 3), 3),
                     "MAE_at_full": round(float(g.sort_values("n_train")["MAE_mean"].iloc[-1]), 3)})
    return pd.DataFrame(rows)


def plot(summary, path):
    split_types = [s for s in config.SPLIT_TYPES if s in set(summary["split_type"])]
    fig, axes = plt.subplots(1, len(split_types), figsize=(5.6 * len(split_types), 4.4),
                             sharey=True, squeeze=False)
    for ax, split_type in zip(axes[0], split_types):
        for rep in REPRESENTATIONS:
            g = summary[(summary.split_type == split_type) & (summary.representation == rep)].sort_values("n_train")
            if g.empty:
                continue
            ax.errorbar(g["n_train"], g["MAE_mean"], yerr=g["MAE_std"].fillna(0), marker="o", ms=5, lw=2,
                        capsize=3, color=REP_COLORS[rep], label=REP_LABELS[rep])
        ax.set_xscale("log")
        ax.set_yscale("log")
        sizes = sorted(set(summary.loc[summary.split_type == split_type, "n_train"]))
        ax.set_xticks(sizes, [f"{n / 1000:.0f}k" if n >= 1000 else str(n) for n in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("Training molecules")
        ax.set_title(f"{split_type} split")
        ax.grid(True, which="both", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Test MAE (eV), mean ± std over seeds")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Learning curves: HOMO–LUMO gap", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(reps, split_types, seeds, force):
    config.make_dirs()
    sel = load_selected()
    print(f"Model (from step 3): {sel['model']}  {sel['params']}")
    mols = load_molecules()
    splits = pd.read_parquet(config.SPLITS)
    if not splits["mol_id"].equals(mols["mol_id"]):
        raise ValueError("splits.parquet is not aligned with molecules.parquet; rerun step 2")
    invalidate_if_changed(config.LC_DIR, splits, ["*.json"])
    y = mols[config.TARGET].to_numpy()

    results = []
    for rep in reps:
        X, _ = load_representation(rep, mols)
        print(f"\n{rep}: {X.shape[1]:,} features")
        for split_type in split_types:
            for seed in seeds:
                n_full = int((splits[split_column(split_type, seed)] == "train").sum())
                for n in [s for s in config.LC_SIZES if s < n_full] + [n_full]:
                    res, status = run_one(sel, rep, split_type, seed, n, X, y, splits, force)
                    results.append(res)
                    print(f"  {rep:12s} {split_type:9s} seed {seed}  n={res['n_train']:>7,} | "
                          f"MAE {res['MAE_eV']:.3f} eV | {status}")
        del X

    runs = pd.DataFrame(results)
    runs.to_csv(config.RESULTS_DIR / "learning_curve_runs.csv", index=False)
    # group by nominal size ("full" may differ by a molecule or two between scaffold seeds)
    summary = (runs.groupby(["representation", "split_type", "point"])
               .agg(n_train=("n_train", "mean"), MAE_mean=("MAE_eV", "mean"), MAE_std=("MAE_eV", "std"),
                    n_seeds=("MAE_eV", "size")).reset_index())
    summary["n_train"] = summary["n_train"].round().astype(int)
    summary = summary.sort_values(["representation", "split_type", "n_train"]).reset_index(drop=True)
    summary.to_csv(config.RESULTS_DIR / "learning_curve_summary.csv", index=False)
    slopes = fit_power_law(summary)
    slopes.to_csv(config.RESULTS_DIR / "learning_curve_slopes.csv", index=False)
    plot(summary, config.FIG_DIR / "fig2_learning_curves.png")

    table = summary.assign(MAE=[f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"
                                for m, s in zip(summary.MAE_mean, summary.MAE_std)])
    print("\nTest MAE (eV) by training-set size:")
    print(table.pivot_table(index="n_train", columns=["split_type", "representation"], values="MAE",
                            aggfunc="first").to_string())
    print("\nPower-law fit  MAE = a * n^(-b):")
    print(slopes.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", nargs="+", default=list(REPRESENTATIONS), choices=REPRESENTATIONS)
    p.add_argument("--splits", nargs="+", default=list(config.SPLIT_TYPES), choices=config.SPLIT_TYPES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    p.add_argument("--force", action="store_true", help="retrain even if a run is cached")
    a = p.parse_args()
    main(a.reps, a.splits, a.seeds, a.force)
