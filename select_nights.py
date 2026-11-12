"""Pick 30 first-night recordings from Sleep-EDF Expanded (Sleep Cassette): 10 per age group."""
import os
import mne
import pandas as pd

csv = os.path.join(os.path.dirname(mne.__file__), "datasets", "sleep_physionet", "age_records.csv")
d = pd.read_csv(csv)
d = d[(d["record type"] == "PSG") & (d["night"] == 1)].copy()
groups = {"25 to 34": (0, 40), "50 to 60": (41, 64), "66 to 101": (65, 120)}
parts = []
for name, (lo, hi) in groups.items():
    g = d[(d["age"] >= lo) & (d["age"] <= hi)].sort_values("subject")
    step = max(1, len(g) // 10)
    g = g.iloc[::step].head(10).copy()
    g["age_group"] = name
    parts.append(g)
pick = pd.concat(parts)
pick[["subject", "age", "sex", "age_group"]].to_csv("data/nights.csv", index=False)
print(pick.groupby("age_group").size().to_dict())
