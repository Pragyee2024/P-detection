import glob
from pathlib import Path
import argparse, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, balanced_accuracy_score)



ap = argparse.ArgumentParser()
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--pred-dir", default="evaluation/predictions")
ap.add_argument("--out-dir", default="evaluation/results")
args = ap.parse_args()

PRED_DIR = args.pred_dir
OUT_DIR = Path(args.out_dir)
THRESHOLD = args.threshold
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = {"subject", "y_true", "y_proba"}


def load_predictions(path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)}")
    if df["y_proba"].isna().any() or not df["y_proba"].between(0, 1).all():
        sys.exit(f"{path}: y_proba must be in [0, 1] with no blanks")
    return df


def score(y_true, y_proba):
    y_pred = (y_proba >= THRESHOLD).astype(int)
    auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec,
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def to_subject_level(df):
    g = df.groupby("subject").agg(y_true=("y_true", "first"),
                                  y_proba=("y_proba", "mean"))
    return g.reset_index()


files = sorted(glob.glob(f"{PRED_DIR}/*.csv"))
if not files:
    sys.exit(f"No CSVs found in {PRED_DIR}")

rows, preds = [], {}
for path in files:
    model = Path(path).stem
    df = load_predictions(path)
    preds[model] = df

    rec = score(df["y_true"].to_numpy(), df["y_proba"].to_numpy())
    sub_df = to_subject_level(df)
    sub = score(sub_df["y_true"].to_numpy(), sub_df["y_proba"].to_numpy())

    rows.append({"model": model, "level": "recording",
                 "n": len(df), **rec})
    rows.append({"model": model, "level": "subject",
                 "n": len(sub_df), **sub})

table = pd.DataFrame(rows).round(3)
ref_name, ref_df = next(iter(preds.items()))
for m, d in preds.items():
    if len(d) != len(ref_df) or set(d["subject"]) != set(ref_df["subject"]):
        print(f"WARNING: {m} does not cover the same subjects/rows as "
              f"{ref_name}. Comparison may be unfair (different split?).")

table.to_csv(OUT_DIR / "comparison.csv", index=False)
with open(OUT_DIR / "comparison.md", "w") as f:
    f.write(table.to_markdown(index=False))
print(table.to_string(index=False))

# --- Chart 1: ROC curves (recording level) ---
plt.figure(figsize=(6, 5))
for model, df in preds.items():
    if df["y_true"].nunique() > 1:
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_proba"])
        auc = roc_auc_score(df["y_true"], df["y_proba"])
        plt.plot(fpr, tpr, label=f"{model} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves (recording level)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png", dpi=150)
plt.close()

# --- Chart 2: confusion matrices (recording level) ---
n_models = len(preds)
fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5),
                         squeeze=False)
for ax, (model, df) in zip(axes[0], preds.items()):
    y_pred = (df["y_proba"] >= THRESHOLD).astype(int)
    cm = confusion_matrix(df["y_true"], y_pred, labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Healthy", "PD"])
    ax.set_yticks([0, 1], ["Healthy", "PD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(model, fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
plt.close()

# --- Chart 3: metric bars (recording level) ---
rec_table = table[table["level"] == "recording"].set_index("model")
metrics = ["accuracy", "precision", "recall", "specificity",
           "balanced_acc", "f1", "roc_auc"]
rec_table[metrics].plot(kind="bar", figsize=(9, 5))
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model comparison (recording level)")
plt.xticks(rotation=30, ha="right")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "metric_comparison.png", dpi=150)
plt.close()

print("\nSaved table and charts to", OUT_DIR)