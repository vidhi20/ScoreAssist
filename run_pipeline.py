"""Download (first run only), score 30 real nights, and measure how much technologist review the AI saves.

Usage: python run_pipeline.py
"""

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from mne.datasets.sleep_physionet.age import fetch_data

from scoreassist.analyze import summarize
from scoreassist.score import score_night

warnings.filterwarnings("ignore")
CACHE = Path("results/nights")


def run_one(row):
    out = CACHE / f"subject_{row['subject']:02d}.json"
    if out.exists():
        return json.loads(out.read_text())
    psg, hyp = fetch_data(subjects=[row["subject"]], recording=[1], verbose="error")[0]
    res = score_night(psg, hyp, row["age"], row["sex"] == "male")
    res.update({"subject": int(row["subject"]), "age": int(row["age"]),
                "sex": row["sex"], "age_group": row["age_group"]})
    out.write_text(json.dumps(res))
    return res


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    nights = pd.read_csv("data/nights.csv").to_dict("records")
    with ProcessPoolExecutor(4) as ex:
        results = list(ex.map(run_one, nights))
    metrics, dashboard = summarize(results)
    Path("results/metrics.json").write_text(json.dumps(metrics, indent=2))
    tpl = Path("docs/template.html").read_text()
    Path("docs/index.html").write_text(tpl.replace("/*DATA*/null", json.dumps(dashboard)))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
