"""Train and tune the MLP with a subject-wise split; save out-of-fold predictions.

Run from the repo root:  python models/train_mlp.py
Output: evaluation/predictions/mlp.csv  (name, subject, y_true, y_proba, fold)
"""
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- settings ----------
DATA_FILE = (glob.glob("data/*.data") + glob.glob("data/*.csv"))[0]  
OUT_PATH = Path("evaluation/predictions/mlp.csv")
SPLIT_SEED = 42
N_FOLDS = 5
MODEL_SEEDS = [0, 1, 2, 3, 4]
HIDDEN_SIZES = [(16,), (32,), (64,), (128,), (16, 8)]
ALPHAS = [1e-4, 1e-2, 1.0]


# ---------- data ----------
def load_data():
    df = pd.read_csv(DATA_FILE)
    # phon_R01_S01_1 -> S01
    df["subject"] = df["name"].str.extract(r"_(S\d+)_")[0]
    assert df["subject"].notna().all(), "Could not parse subject ID from name"
    y = df["status"].to_numpy()
    X = df.drop(columns=["name", "subject", "status"])
    return df[["name", "subject"]], X, y



def get_folds(X, y, groups):
    """Return a list of (train_idx, test_idx). TEMPORARY: replace with shared split."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    return list(cv.split(X, y, groups))


# ---------- model ----------
def make_model(hidden, alpha, seed):
    return Pipeline([
        ("scale", StandardScaler()),  # fitted on the training part of each fold only
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden, alpha=alpha,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, max_iter=2000, random_state=seed)),
    ])


def oof_proba(X, y, folds, hidden, alpha, seed):
    proba = np.zeros(len(y))
    for tr, te in folds:
        model = make_model(hidden, alpha, seed)
        model.fit(X.iloc[tr], y[tr])
        proba[te] = model.predict_proba(X.iloc[te])[:, 1]
    return proba


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, p),
            "acc": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0)}


def main():
    meta, X, y = load_data()
    groups = meta["subject"].to_numpy()
    folds = get_folds(X, y, groups)

    # sanity check: no subject appears in both train and test
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "Subject leakage!"
    print(f"{len(y)} recordings, {len(set(groups))} subjects, {len(folds)} folds, no leakage")

    # ---------- light tuning: average over seeds ----------
    results = []
    for hidden, alpha in itertools.product(HIDDEN_SIZES, ALPHAS):
        per_seed = [metrics(y, oof_proba(X, y, folds, hidden, alpha, s))
                    for s in MODEL_SEEDS]
        avg = {k: np.mean([m[k] for m in per_seed]) for k in per_seed[0]}
        std_auc = np.std([m["auc"] for m in per_seed])
        results.append({"hidden": hidden, "alpha": alpha, **avg, "auc_std": std_auc})
        print(f"hidden={str(hidden):8s} alpha={alpha:<7g} "
              f"AUC={avg['auc']:.3f}±{std_auc:.3f}  acc={avg['acc']:.3f}  f1={avg['f1']:.3f}")

    res = pd.DataFrame(results).sort_values("auc", ascending=False)
    res.to_csv("evaluation/mlp_tuning.csv", index=False)
    best = res.iloc[0]
    hidden, alpha = best["hidden"], best["alpha"]
    print(f"\nBest: hidden={hidden}, alpha={alpha}  (mean AUC {best['auc']:.3f})")

    # ---------- final predictions: average probabilities over seeds ----------
    probas = np.mean([oof_proba(X, y, folds, hidden, alpha, s)
                      for s in MODEL_SEEDS], axis=0)
    fold_id = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(folds):
        fold_id[te] = i

    out = meta.copy()
    out["y_true"] = y
    out["y_proba"] = probas
    out["fold"] = fold_id
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("Final (seed-averaged):", {k: round(v, 3) for k, v in metrics(y, probas).items()})
    print("Saved", OUT_PATH)


if __name__ == "__main__":
    main()