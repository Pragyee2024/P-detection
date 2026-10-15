import pandas as pd

df = pd.read_csv("data/parkinsons.data")

df["subject"] = df["name"].str.extract(r"(S\d+)")

print("Subjects:", df["subject"].nunique())
print("Recordings per subject:")
print(df.groupby("subject").size().describe())
print("Label counts (recordings):")
print(df["status"].value_counts())
print("Label counts (subjects):")
print(df.groupby("subject")["status"].first().value_counts())