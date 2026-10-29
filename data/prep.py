import json
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv("data/parkinsons.data")

# 2. Get subject ID from name (e.g. phon_R01_S01_1 -> S01)
df["subject"] = df["name"].str.extract(r"(S\d+)")

# 3. Separate features and label
features = [c for c in df.columns if c not in ["name", "status", "subject"]]
X, y, groups = df[features], df["status"], df["subject"]

# 4. Subject-wise split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
train, test = df.iloc[train_idx], df.iloc[test_idx]

# 5. Check no subject is in both sets
assert set(train["subject"]).isdisjoint(set(test["subject"]))
print("Train:", train.shape, "Test:", test.shape)
print("Train status ratio:\n", train["status"].value_counts(normalize=True))
print("Test status ratio:\n", test["status"].value_counts(normalize=True))

# 6. Save split files
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

# 7. Fit scaler on TRAIN only and save it
scaler = StandardScaler().fit(train[features])
joblib.dump(scaler, "models/scaler.pkl")

# 8. Save feature order
with open("models/features.json", "w") as f:
    json.dump(features, f)

print("Done.")