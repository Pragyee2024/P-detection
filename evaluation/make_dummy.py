import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

df = pd.read_csv("data/parkinsons.data")
df["subject"] = df["name"].str.extract(r"(S\d+)")

# one row per subject with its label
subj = df.groupby("subject")["status"].first()

# fake test set: 2 healthy + 6 Parkinson's subjects
healthy = rng.choice(subj[subj == 0].index, 2, replace=False)
pd_subj = rng.choice(subj[subj == 1].index, 6, replace=False)
test = df[df["subject"].isin(list(healthy) + list(pd_subj))].copy()

out = pd.DataFrame({
    "name": test["name"],
    "subject": test["subject"],
    "y_true": test["status"],
})

n = len(out)
y = out["y_true"].to_numpy()

models = {
    # random guesses
    "dummy_random": rng.uniform(0, 1, n),
    # always predicts Parkinson's
    "dummy_always_pd": np.full(n, 0.95),
    # decent model: signal plus noise
    "dummy_good": np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.2, n), 0, 1),
}

Path("evaluation/predictions").mkdir(parents=True, exist_ok=True)
for name, proba in models.items():
    res = out.copy()
    res["y_proba"] = proba.round(4)
    res.to_csv(f"evaluation/predictions/{name}.csv", index=False)
    print("saved", name, "-", n, "rows")