import json
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load split files and feature list
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
features = json.load(open("models/features.json"))
X_train, y_train, groups = train[features], train["status"], train["subject"]
X_test, y_test = test[features], test["status"]

# 2. Tune with subject-grouped cross-validation (scaler refit inside each fold)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
])
grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"]}
search = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=3), scoring="roc_auc")
search.fit(X_train, y_train, groups=groups)
print("Best params:", search.best_params_)
print("CV ROC-AUC:", round(search.best_score_, 3))

# 3. Final model: best params, trained on data scaled with the shared scaler
scaler = joblib.load("models/scaler.pkl")
best = LogisticRegression(
    C=search.best_params_["clf__C"],
    penalty=search.best_params_["clf__penalty"],
    max_iter=5000, class_weight="balanced", solver="liblinear",
)
best.fit(scaler.transform(X_train), y_train)

# 4. Evaluate once on the test set
X_test_s = scaler.transform(X_test)
pred = best.predict(X_test_s)
prob = best.predict_proba(X_test_s)[:, 1]
print(classification_report(y_test, pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, prob), 3))

# 5. Save model and predictions
os.makedirs("results", exist_ok=True)
joblib.dump(best, "models/logreg.pkl")
pd.DataFrame({
    "subject": test["subject"], "y_true": y_test, "y_pred": pred, "y_prob": prob,
}).to_csv("results/logreg_preds.csv", index=False)
print("Done.")