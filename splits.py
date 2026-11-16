"""Train / validation / test splits.

random    molecules shuffled at random: measures interpolation between similar molecules.
scaffold  whole Bemis-Murcko scaffold groups go to one side ("balanced scaffold split", as in
          Chemprop): groups larger than half of the validation or test size are placed first
          (so the big families end up in training), the remaining small groups are shuffled
          and fill the rest. Validation and test molecules therefore have ring systems that
          never appear in training. Different seeds give different scaffold splits.
"""
import numpy as np
import pandas as pd


def random_split(n, fractions, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    return labels


def scaffold_split(scaffolds, fractions, seed):
    scaffolds = pd.Series(scaffolds).reset_index(drop=True)
    n = len(scaffolds)
    train_size, val_size = fractions[0] * n, fractions[1] * n
    test_size = n - train_size - val_size

    groups = list(scaffolds.groupby(scaffolds).indices.values())
    big = [g for g in groups if len(g) > val_size / 2 or len(g) > test_size / 2]
    small = [g for g in groups if not (len(g) > val_size / 2 or len(g) > test_size / 2)]
    rng = np.random.default_rng(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    labels = np.empty(n, dtype=object)
    n_train = n_val = 0
    for g in big + small:
        if n_train + len(g) <= train_size:
            labels[g] = "train"
            n_train += len(g)
        elif n_val + len(g) <= val_size:
            labels[g] = "val"
            n_val += len(g)
        else:
            labels[g] = "test"
    return labels


def fingerprint(splits):
    """Short hash of the split table; changes whenever molecules or split assignments change."""
    import hashlib
    return hashlib.md5(pd.util.hash_pandas_object(splits, index=False).values.tobytes()).hexdigest()[:12]


def invalidate_if_changed(folder, splits, patterns):
    """Delete cached results in `folder` that were made with different splits."""
    from pathlib import Path
    folder = Path(folder)
    stamp = folder / "splits_fingerprint.txt"
    fp = fingerprint(splits)
    if stamp.is_file() and stamp.read_text().strip() != fp:
        removed = 0
        for pattern in patterns:
            for f in folder.glob(pattern):
                f.unlink()
                removed += 1
        print(f"Splits changed since the cached results were made: removed {removed} cached files in {folder.name}/")
    stamp.write_text(fp)


def split_column(split_type, seed):
    return f"{split_type}_s{seed}"


def check_no_shared_scaffolds(labels, scaffolds):
    """Raise if a scaffold appears in more than one of train / val / test."""
    df = pd.DataFrame({"part": labels, "scaffold": np.asarray(scaffolds)})
    parts_per_scaffold = df.groupby("scaffold")["part"].nunique()
    shared = int((parts_per_scaffold > 1).sum())
    if shared:
        raise AssertionError(f"{shared} scaffolds are shared between parts")
