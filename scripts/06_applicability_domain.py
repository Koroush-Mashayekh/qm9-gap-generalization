"""Step 6 - applicability domain: is the error larger for molecules that are unlike the training set?

For every test molecule of step 4 (best representation, selected model) we compute its Tanimoto
similarity to the MOST SIMILAR training molecule (binary Morgan fingerprint, radius 2, 2,048 bits),
and compare it with the absolute error of its prediction. No model is trained in this step.

Questions:
  1. Are scaffold-split test molecules less similar to the training set than random-split ones?
  2. Does the error grow as similarity falls?
  3. At the SAME similarity, are the two splits equally hard? If yes, the scaffold penalty of step 4
     is explained by lower similarity, and similarity is a usable warning signal.
  4. If we trust only predictions with similarity >= t, how many molecules are kept and how accurate
     are they?  (a simple applicability-domain rule)

Outputs (results/applicability/ and results/):
    applicability/nn_similarity__<split>.parquet   similarity of every test molecule (cached)
    applicability_molecules.csv     mol_id, split, seed, similarity, actual, predicted, abs error
    applicability_bins.csv          MAE per similarity bin (mean ± std over seeds)
    applicability_thresholds.csv    coverage and MAE when keeping only similarity >= t
    applicability_summary.csv       median similarity and Spearman correlation per split type
    figures/fig3_applicability.png

Runs in about 15-25 minutes on Colab (similarity of 12,656 test x 101,247 training molecules, 6 times).
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
from qm9gap.features import load_molecules
from qm9gap.splits import invalidate_if_changed, split_column

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLIT_COLORS = {"random": "#2a78d6", "scaffold": "#eb6834"}


def nearest_train_similarity(bits, train_idx, test_idx, chunk=256):
    """Max Tanimoto similarity of each test molecule to all training molecules.
    Tanimoto(a, b) = common bits / (bits in a + bits in b - common bits)."""
    B = bits[train_idx].astype(np.float32)          # training fingerprints (0/1)
    b_count = B.sum(axis=1)
    out = np.empty(len(test_idx), dtype=np.float32)
    for start in range(0, len(test_idx), chunk):
        A = bits[test_idx[start:start + chunk]].astype(np.float32)
        common = A @ B.T                              # common bits, for every test x train pair
        union = A.sum(axis=1)[:, None] + b_count[None, :] - common
        out[start:start + chunk] = (common / np.maximum(union, 1)).max(axis=1)
    return out


def load_predictions(model, col):
    path = config.RUNS_DIR / f"{model}__{config.AD_REPRESENTATION}__{col}__pred.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found - run step 4 for this split and seed first")
    return pd.read_parquet(path)


def similarity_for_split(col, mols, splits, bits, force):
    out = config.AD_DIR / f"nn_similarity__{col}.parquet"
    if out.is_file() and not force:
        return pd.read_parquet(out), "cached"
    part = splits[col].to_numpy()
    train_idx, test_idx = np.flatnonzero(part == "train"), np.flatnonzero(part == "test")
    start = time.time()
    sim = nearest_train_similarity(bits, train_idx, test_idx)
    df = pd.DataFrame({"mol_id": mols["mol_id"].to_numpy()[test_idx], "nn_similarity": sim})
    df.to_parquet(out, index=False)
    return df, f"computed in {(time.time() - start) / 60:.1f} min"


def bin_table(data):
    data = data.assign(bin=pd.cut(data["nn_similarity"], bins=list(config.AD_BINS), right=False))
    per_seed = (data.groupby(["split_type", "seed", "bin"], observed=True)
                .agg(n=("abs_error", "size"), MAE=("abs_error", "mean")).reset_index())
    table = (per_seed.groupby(["split_type", "bin"], observed=True)
             .agg(n_mean=("n", "mean"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")).reset_index())
    table["bin"] = table["bin"].astype(str).str.replace("1.0001", "1.0", regex=False)
    return table


def threshold_table(data):
    rows = []
    for (split_type, seed), g in data.groupby(["split_type", "seed"]):
        for t in config.AD_THRESHOLDS:
            kept = g[g["nn_similarity"] >= t]
            rows.append({"split_type": split_type, "seed": seed, "threshold": t,
                         "coverage": len(kept) / len(g),
                         "MAE_kept": kept["abs_error"].mean() if len(kept) else np.nan,
                         "MAE_rejected": g.loc[g["nn_similarity"] < t, "abs_error"].mean()})
    per_seed = pd.DataFrame(rows)
    return (per_seed.groupby(["split_type", "threshold"])
            .agg(coverage=("coverage", "mean"), MAE_kept=("MAE_kept", "mean"),
                 MAE_rejected=("MAE_rejected", "mean")).reset_index())


def summary_table(data):
    rows = []
    for (split_type, seed), g in data.groupby(["split_type", "seed"]):
        rows.append({"split_type": split_type, "seed": seed,
                     "median_similarity": g["nn_similarity"].median(),
                     "share_below_0.5": (g["nn_similarity"] < 0.5).mean(),
                     "spearman_similarity_vs_error": g["nn_similarity"].corr(g["abs_error"], method="spearman"),
                     "MAE": g["abs_error"].mean()})
    per_seed = pd.DataFrame(rows)
    out = per_seed.drop(columns="seed").groupby("split_type").agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def plot(data, bins, thresholds, path):
    split_types = [s for s in config.SPLIT_TYPES if s in set(data["split_type"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]                                       # (a) how similar are test molecules?
    edges = np.linspace(0, 1, 41)
    for s in split_types:
        ax.hist(data.loc[data.split_type == s, "nn_similarity"], bins=edges, density=True, alpha=0.55,
                color=SPLIT_COLORS[s], label=f"{s} split")
    ax.set(xlabel="Similarity to nearest training molecule (Tanimoto)", ylabel="Density",
           title="(a) How similar are test molecules to the training set?")
    ax.legend(frameon=False)

    ax = axes[1]                                       # (b) error vs similarity
    bins = bins[bins["n_mean"] >= 30]                  # too few molecules -> unreliable MAE
    labels = list(dict.fromkeys(bins["bin"]))
    labels = sorted(labels, key=lambda lab: float(lab.strip("[)").split(",")[0]))
    x = np.arange(len(labels))
    for s in split_types:
        b = bins[bins.split_type == s].set_index("bin").reindex(labels)
        ax.errorbar(x, b["MAE_mean"], yerr=b["MAE_std"].fillna(0), marker="o", lw=2, capsize=3,
                    color=SPLIT_COLORS[s], label=f"{s} split")
    ax.set_xticks(x, [lab.strip("[)").replace(", ", "–") for lab in labels], rotation=30)
    ax.set(xlabel="Similarity bin (bins with < 30 molecules hidden)", ylabel="Test MAE (eV)",
           title="(b) Error vs. similarity")
    ax.legend(frameon=False)

    ax = axes[2]                                       # (c) trust only similar molecules
    for s in split_types:
        t = thresholds[(thresholds.split_type == s) & (thresholds.coverage >= 0.01)]
        if t.empty:
            continue
        ax.plot(t["coverage"] * 100, t["MAE_kept"], marker="o", lw=2, color=SPLIT_COLORS[s], label=f"{s} split")
        for _, r in t.iloc[[0, len(t) // 2, -1]].iterrows():
            ax.annotate(f"≥{r.threshold:.2f}", (r.coverage * 100, r.MAE_kept), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color="0.3")
    ax.invert_xaxis()
    ax.set(xlabel="Test molecules kept (%)", ylabel="MAE of kept molecules (eV)",
           title="(c) Trust only predictions above a similarity threshold")
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(split_types, seeds, force):
    config.make_dirs()
    sel = json.loads(config.SELECTED_MODEL.read_text())
    print(f"Predictions: {sel['model']} on '{config.AD_REPRESENTATION}' from step 4")
    mols = load_molecules()
    splits = pd.read_parquet(config.SPLITS)
    if not splits["mol_id"].equals(mols["mol_id"]):
        raise ValueError("splits.parquet is not aligned with molecules.parquet; rerun step 2")
    invalidate_if_changed(config.AD_DIR, splits, ["nn_similarity__*.parquet"])
    bits = np.load(config.MORGAN) > 0                  # count fingerprint -> on/off bits

    frames = []
    for split_type in split_types:
        for seed in seeds:
            col = split_column(split_type, seed)
            sim, status = similarity_for_split(col, mols, splits, bits, force)
            pred = load_predictions(sel["model"], col)
            df = pred.merge(sim, on="mol_id", how="inner", validate="one_to_one")
            df["abs_error"] = (df["actual_eV"] - df["predicted_eV"]).abs()
            df["split_type"], df["seed"] = split_type, seed
            frames.append(df)
            print(f"  {col:12s} | {len(df):,} test molecules | median similarity "
                  f"{df['nn_similarity'].median():.2f} | {status}")

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(config.RESULTS_DIR / "applicability_molecules.csv", index=False)
    bins, thresholds, summary = bin_table(data), threshold_table(data), summary_table(data)
    bins.to_csv(config.RESULTS_DIR / "applicability_bins.csv", index=False)
    thresholds.to_csv(config.RESULTS_DIR / "applicability_thresholds.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "applicability_summary.csv", index=False)
    plot(data, bins, thresholds, config.FIG_DIR / "fig3_applicability.png")

    pd.set_option("display.width", 160)
    print("\n1) Similarity of test molecules to the training set, and its link to the error (mean ± std over seeds):")
    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"
    compact = pd.DataFrame({
        "split": summary["split_type"],
        "median similarity": [fmt(m, s) for m, s in zip(summary.median_similarity_mean, summary.median_similarity_std)],
        "share < 0.5": [fmt(m, s) for m, s in zip(summary["share_below_0.5_mean"], summary["share_below_0.5_std"])],
        "Spearman(sim, |error|)": [fmt(m, s) for m, s in zip(summary.spearman_similarity_vs_error_mean,
                                                              summary.spearman_similarity_vs_error_std)],
        "MAE (eV)": [fmt(m, s) for m, s in zip(summary.MAE_mean, summary.MAE_std)]})
    print(compact.to_string(index=False))
    print("\n2) Test MAE (eV) and number of molecules per similarity bin (mean over seeds):")
    show = bins.assign(MAE=[f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"
                            for m, s in zip(bins.MAE_mean, bins.MAE_std)],
                       n=bins.n_mean.round().astype(int))
    print(show.pivot(index="bin", columns="split_type", values=["MAE", "n"]).to_string())
    print("\n3) Keep only molecules with similarity >= threshold:")
    print(thresholds.assign(coverage=(thresholds.coverage * 100).round(1)).round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--splits", nargs="+", default=list(config.SPLIT_TYPES), choices=config.SPLIT_TYPES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    p.add_argument("--force", action="store_true", help="recompute similarities even if cached")
    a = p.parse_args()
    main(a.splits, a.seeds, a.force)
