"""Step 7 - chemical interpretation: what drives the predictions, and where does the model fail?

Part A - Which descriptors matter?
    The selected model is trained on the 190 RDKit descriptors (their names have a chemical meaning;
    Morgan bits do not) on config.CHEM_SHAP_SPLIT. SHAP values on its test set show how much each
    descriptor moves the predicted gap, and in which direction.
Part B - Which kinds of molecules are hard?
    The step-4 test predictions of the best representation (all splits and seeds) are grouped by
    structural class (aromatic ring, 3-membered ring, carbonyl, ...) and by the size of the true gap.
Part C - The worst predictions
    The molecules with the largest errors on the main split are listed and drawn, for inspection by eye.

Outputs (results/chemistry/ and results/):
    chemistry/<model>__descriptors__<split>.joblib   descriptor model used for SHAP (cached)
    chemistry/structural_classes.parquet             class membership of every molecule (cached)
    chemistry_shap_importance.csv      mean |SHAP| and direction per descriptor
    chemistry_error_by_class.csv       MAE per structural class, random vs scaffold split
    chemistry_error_by_gap.csv         mean signed error and MAE per true-gap range
    chemistry_worst_molecules.csv      the CHEM_N_WORST largest errors (main split)
    figures/fig4_shap_importance.png, fig5_error_by_chemistry.png, fig6_worst_molecules.png

Runs in about 10-15 minutes on Colab (one model training of about 5 minutes, SHAP, then analysis).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # find qm9gap without installing

import argparse
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

from qm9gap import config
from qm9gap.evaluate import regression_metrics
from qm9gap.features import load_molecules, load_representation
from qm9gap.models import BOOSTERS, build, fit
from qm9gap.splits import split_column

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLIT_COLORS = {"random": "#2a78d6", "scaffold": "#eb6834"}
RAISES, LOWERS = "#2a78d6", "#eb6834"      # SHAP direction colours


# --------------------------------------------------------------------------- Part A
def shap_importance(sel, mols, splits, force):
    out_csv = config.RESULTS_DIR / "chemistry_shap_importance.csv"
    if sel["model"] != "lightgbm":
        print(f"  SHAP values are computed with LightGBM's built-in TreeSHAP; selected model is "
              f"{sel['model']} - Part A skipped.")
        return None
    if out_csv.is_file() and not force:
        print("  using cached SHAP importance")
        return pd.read_csv(out_csv)

    X, names = load_representation("descriptors", mols)
    y = mols[config.TARGET].to_numpy()
    part = splits[config.CHEM_SHAP_SPLIT].to_numpy()
    tr, va, te = part == "train", part == "val", part == "test"

    model_path = config.CHEM_DIR / f"{sel['model']}__descriptors__{config.CHEM_SHAP_SPLIT}.joblib"
    if model_path.is_file() and not force:
        model = joblib.load(model_path)
    else:
        model = build(sel["model"], sel["params"], seed=0)
        print("  training LightGBM on descriptors (about 5 min) ...", flush=True)
        fit(sel["model"], model, X[tr], y[tr], X[va], y[va])
        joblib.dump(model, model_path)
    print(f"  descriptor model, {config.CHEM_SHAP_SPLIT} test MAE: "
          f"{regression_metrics(y[te], model.predict(X[te]))['MAE_eV']:.3f} eV")

    # TreeSHAP on a random sample of test molecules (one column per feature + a bias column)
    test_idx = np.flatnonzero(te)
    sample = np.sort(np.random.default_rng(0).choice(test_idx, size=min(config.CHEM_SHAP_SAMPLE, len(test_idx)),
                                                     replace=False))
    parts, start = [], time.time()
    for k in range(0, len(sample), 250):
        parts.append(model.predict(X[sample[k:k + 250]], pred_contrib=True))
        done = min(k + 250, len(sample))
        print(f"    SHAP: {done:,}/{len(sample):,} molecules | {(time.time() - start) / 60:.1f} min", flush=True)
    shap = np.vstack(parts)[:, :-1]
    Xs = X[sample]
    imp = pd.DataFrame({"descriptor": names, "mean_abs_shap_eV": np.abs(shap).mean(axis=0)})
    # direction: does a HIGHER descriptor value raise (+) or lower (-) the predicted gap?
    imp["direction"] = [pd.Series(Xs[:, j]).corr(pd.Series(shap[:, j]), method="spearman")
                        for j in range(len(names))]
    imp = imp.sort_values("mean_abs_shap_eV", ascending=False).reset_index(drop=True)
    imp.to_csv(out_csv, index=False)
    return imp


def plot_shap(imp, path, top=15):
    t = imp.head(top).iloc[::-1]
    colors = [RAISES if d > 0 else LOWERS for d in t["direction"].fillna(0)]
    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.4))
    ax.barh(t["descriptor"], t["mean_abs_shap_eV"], color=colors)
    ax.set_xlabel(f"Mean |SHAP value| (eV): average shift of the predicted gap\n"
                  f"({config.CHEM_SHAP_SAMPLE:,} test molecules of {config.CHEM_SHAP_SPLIT})")
    ax.set_title(f"Top {top} descriptors (LightGBM on RDKit descriptors)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RAISES), plt.Rectangle((0, 0), 1, 1, color=LOWERS)],
              labels=["higher value → larger gap", "higher value → smaller gap"], frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- Part B
def load_all_predictions(sel):
    frames = []
    for split_type in config.SPLIT_TYPES:
        for seed in config.SEEDS:
            col = split_column(split_type, seed)
            path = config.RUNS_DIR / f"{sel['model']}__{config.AD_REPRESENTATION}__{col}__pred.parquet"
            if not path.is_file():
                print(f"  missing {path.name} - skipped")
                continue
            df = pd.read_parquet(path)
            df["split_type"], df["seed"] = split_type, seed
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["error"] = data["predicted_eV"] - data["actual_eV"]          # > 0: gap over-predicted
    data["abs_error"] = data["error"].abs()
    return data


def structural_table(mols, force):
    from qm9gap import chem
    path = config.CHEM_DIR / "structural_classes.parquet"
    if path.is_file() and not force:
        return pd.read_parquet(path)
    rows = [chem.structural_classes(s) for s in mols["smiles"]]
    table = pd.DataFrame(rows, columns=list(chem.STRUCTURAL_CLASSES))
    table.insert(0, "mol_id", mols["mol_id"].to_numpy())
    table.to_parquet(path, index=False)
    return table


def error_by_class(data, classes):
    names = [c for c in classes.columns if c != "mol_id"]
    d = data.merge(classes, on="mol_id", how="left")
    rows = []
    for (split_type, seed), g in d.groupby(["split_type", "seed"]):
        rows.append({"class": "all molecules", "split_type": split_type, "seed": seed,
                     "n": len(g), "share": 1.0, "MAE": g["abs_error"].mean()})
        for name in names:
            sub = g[g[name]]
            rows.append({"class": name, "split_type": split_type, "seed": seed, "n": len(sub),
                         "share": len(sub) / len(g), "MAE": sub["abs_error"].mean() if len(sub) else np.nan})
    per_seed = pd.DataFrame(rows)
    table = (per_seed.groupby(["class", "split_type"], sort=False)
             .agg(n=("n", "mean"), share=("share", "mean"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
             .reset_index())
    wide = table.pivot(index="class", columns="split_type", values=["share", "MAE_mean", "MAE_std"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reindex(["all molecules"] + names).reset_index()
    if {"MAE_mean_random", "MAE_mean_scaffold"} <= set(wide.columns):
        wide["scaffold_penalty_eV"] = wide["MAE_mean_scaffold"] - wide["MAE_mean_random"]
    return wide


def error_by_gap(data):
    d = data.assign(gap_range=pd.cut(data["actual_eV"], bins=list(config.CHEM_GAP_BINS), right=False))
    per_seed = (d.groupby(["split_type", "seed", "gap_range"], observed=True)
                .agg(n=("error", "size"), mean_signed_error=("error", "mean"), MAE=("abs_error", "mean"))
                .reset_index())
    table = (per_seed.groupby(["split_type", "gap_range"], observed=True)
             .agg(n=("n", "mean"), mean_signed_error=("mean_signed_error", "mean"), MAE=("MAE", "mean"))
             .reset_index())
    table["gap_range"] = table["gap_range"].astype(str)
    return table


def plot_chemistry(by_class, by_gap, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    t = by_class[by_class["class"] != "all molecules"].dropna(subset=["MAE_mean_scaffold"])
    t = t.sort_values("MAE_mean_scaffold")
    y = np.arange(len(t))
    h = 0.4
    for k, s in enumerate(["random", "scaffold"]):
        if f"MAE_mean_{s}" in t:
            ax.barh(y + (k - 0.5) * h, t[f"MAE_mean_{s}"], h * 0.95, color=SPLIT_COLORS[s], label=f"{s} split")
    labels = [f"{c}  ({100 * sh:.0f}%)" for c, sh in zip(t["class"], t["share_scaffold"])]
    ax.set_yticks(y, labels)
    overall = by_class.loc[by_class["class"] == "all molecules", "MAE_mean_scaffold"].iloc[0]
    ax.axvline(overall, color="0.3", ls="--", lw=1, label="all molecules, scaffold split")
    ax.set_xlabel("Test MAE (eV)")
    ax.set_title("(a) Error by structural class  (% = share of scaffold test set)", pad=34)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)

    ax = axes[1]
    for s in ["random", "scaffold"]:
        g = by_gap[by_gap.split_type == s]
        g = g[g["n"] >= 30]
        ax.plot(g["gap_range"], g["mean_signed_error"], marker="o", lw=2, color=SPLIT_COLORS[s], label=f"{s} split")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("True HOMO–LUMO gap (eV)")
    ax.set_ylabel("Mean (predicted − true) gap (eV)")
    ax.set_title("(b) Systematic bias: > 0 over-predicted, < 0 under-predicted", pad=34)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- Part C
def worst_molecules(data, mols, split_col):
    split_type, seed = split_col.rsplit("_s", 1)
    d = data[(data.split_type == split_type) & (data.seed == int(seed))]
    worst = d.nlargest(config.CHEM_N_WORST, "abs_error").merge(mols[["mol_id", "smiles", "scaffold"]], on="mol_id")
    sim_path = config.AD_DIR / f"nn_similarity__{split_col}.parquet"
    if sim_path.is_file():
        worst = worst.merge(pd.read_parquet(sim_path), on="mol_id", how="left")
    cols = ["mol_id", "smiles", "actual_eV", "predicted_eV", "error"] + \
           (["nn_similarity"] if "nn_similarity" in worst else [])
    return worst.sort_values("abs_error", ascending=False)[cols].reset_index(drop=True)


def main(force):
    config.make_dirs()
    sel = json.loads(config.SELECTED_MODEL.read_text())
    mols = load_molecules()
    splits = pd.read_parquet(config.SPLITS)
    if not splits["mol_id"].equals(mols["mol_id"]):
        raise ValueError("splits.parquet is not aligned with molecules.parquet; rerun step 2")
    pd.set_option("display.width", 170)

    print("Part A - which descriptors matter (SHAP)")
    imp = shap_importance(sel, mols, splits, force)
    if imp is not None:
        plot_shap(imp, config.FIG_DIR / "fig4_shap_importance.png")
        show = imp.head(15).assign(effect=np.where(imp.head(15)["direction"] > 0,
                                                   "higher -> larger gap", "higher -> smaller gap"))
        print(show[["descriptor", "mean_abs_shap_eV", "effect"]].round(4).to_string(index=False))

    print(f"\nPart B - where are the errors ({sel['model']}, '{config.AD_REPRESENTATION}', all splits and seeds)")
    data = load_all_predictions(sel)
    classes = structural_table(mols, force)
    by_class, by_gap = error_by_class(data, classes), error_by_gap(data)
    by_class.to_csv(config.RESULTS_DIR / "chemistry_error_by_class.csv", index=False)
    by_gap.to_csv(config.RESULTS_DIR / "chemistry_error_by_gap.csv", index=False)
    plot_chemistry(by_class, by_gap, config.FIG_DIR / "fig5_error_by_chemistry.png")

    show = by_class.copy()
    for s in ["random", "scaffold"]:
        if f"share_{s}" in show:
            show[f"share_{s} %"] = (100 * show.pop(f"share_{s}")).round(1)
    print("\nMAE (eV) by structural class (mean over seeds; share = % of test molecules in the class):")
    print(show.round(3).to_string(index=False))
    print("\nBy true gap (mean signed error = predicted - true; > 0 means the gap is over-predicted):")
    print(by_gap.round(3).to_string(index=False))

    scaffold = data[data.split_type == "scaffold"]
    top = scaffold.groupby("seed")["abs_error"].apply(
        lambda e: e.nlargest(max(1, len(e) // 20)).sum() / e.sum()).mean()
    print(f"\nOn the scaffold split, the 5% worst-predicted molecules carry {100 * top:.0f}% of the total absolute error.")

    print(f"\nPart C - the {config.CHEM_N_WORST} worst predictions ({config.CHEM_SHAP_SPLIT})")
    worst = worst_molecules(data, mols, config.CHEM_SHAP_SPLIT)
    worst.to_csv(config.RESULTS_DIR / "chemistry_worst_molecules.csv", index=False)
    print(worst.round(3).to_string(index=False))
    from qm9gap import chem
    legends = [f"true {a:.2f} / pred {p:.2f} eV" for a, p in zip(worst.actual_eV, worst.predicted_eV)]
    chem.draw_grid(worst["smiles"].tolist(), legends, config.FIG_DIR / "fig6_worst_molecules.png")
    print(f"\nFigures: fig4_shap_importance.png, fig5_error_by_chemistry.png, fig6_worst_molecules.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="recompute everything even if cached")
    main(p.parse_args().force)
