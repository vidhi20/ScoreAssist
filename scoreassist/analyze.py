"""Turn per-night scoring into product metrics: agreement, where the AI is unsure, and review savings."""

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .score import STAGES

REVIEW_SHARES = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TARGET_SHARE = 0.20  # product default: technologist reviews the least-confident 20% of the night


def review_curve(expert, model, conf):
    """Agreement if a technologist fixes the least-confident share of epochs (pooled across nights)."""
    order = np.argsort(conf)
    errors = expert != model
    out = []
    for share in REVIEW_SHARES:
        k = int(round(len(conf) * share))
        fixed = np.zeros(len(conf), bool)
        fixed[order[:k]] = True
        agree = float(((expert == model) | fixed).mean())
        caught = float(errors[fixed].sum() / max(errors.sum(), 1))
        out.append({"review_share": share, "agreement": round(agree, 4), "errors_caught": round(caught, 4)})
    return out


def summarize(results):
    expert = np.concatenate([r["expert"] for r in results])
    model = np.concatenate([r["model"] for r in results])
    conf = np.concatenate([r["confidence"] for r in results])

    # One confidence cutoff for every night, set so the pooled review load is about 20%.
    cutoff = float(np.quantile(conf, TARGET_SHARE))
    flagged = conf < cutoff

    per_night = []
    for r in results:
        e, m, c = np.array(r["expert"]), np.array(r["model"]), np.array(r["confidence"])
        f = c < cutoff
        per_night.append({
            "subject": r["subject"], "age": r["age"], "sex": r["sex"], "age_group": r["age_group"],
            "epochs": int(len(e)),
            "agreement": round(float((e == m).mean()), 4),
            "kappa": round(float(cohen_kappa_score(e, m)), 4),
            "flagged_share": round(float(f.mean()), 4),
            "agreement_after_review": round(float(((e == m) | f).mean()), 4),
        })

    by_age = {}
    for g in sorted({p["age_group"] for p in per_night}):
        rows = [p for p in per_night if p["age_group"] == g]
        idx = np.concatenate([[p["age_group"] == g] * p["epochs"] for p in per_night])
        by_age[g] = {
            "nights": len(rows),
            "agreement": round(float((expert[idx] == model[idx]).mean()), 4),
            "kappa": round(float(cohen_kappa_score(expert[idx], model[idx])), 4),
            "flagged_share": round(float(flagged[idx].mean()), 4),
            "agreement_after_review": round(float(((expert[idx] == model[idx]) | flagged[idx]).mean()), 4),
        }

    f1 = f1_score(expert, model, labels=STAGES, average=None)
    cm = confusion_matrix(expert, model, labels=STAGES)
    errors = expert != model
    metrics = {
        "dataset": "Sleep-EDF Expanded, Sleep Cassette, first night, 10 nights per age group",
        "nights": len(results),
        "epochs_scored": int(len(expert)),
        "ai_only": {
            "agreement": round(float((expert == model).mean()), 4),
            "kappa": round(float(cohen_kappa_score(expert, model)), 4),
            "median_night_kappa": round(float(np.median([p["kappa"] for p in per_night])), 4),
            "f1_by_stage": {s: round(float(v), 4) for s, v in zip(STAGES, f1)},
        },
        "with_review": {
            "confidence_cutoff": round(cutoff, 4),
            "share_of_epochs_reviewed": round(float(flagged.mean()), 4),
            "errors_caught_by_review": round(float(errors[flagged].sum() / errors.sum()), 4),
            "agreement_after_review": round(float(((expert == model) | flagged).mean()), 4),
            "error_rate_in_flagged": round(float(errors[flagged].mean()), 4),
            "error_rate_in_unflagged": round(float(errors[~flagged].mean()), 4),
        },
        "by_age_group": by_age,
        "review_curve": review_curve(expert, model, conf),
    }
    # One example night (closest to the median agreement) for the hypnogram view.
    med = np.median([p["agreement"] for p in per_night])
    ex = min(results, key=lambda r: abs(np.mean(np.array(r["expert"]) == np.array(r["model"])) - med))
    sample = {"subject": ex["subject"], "age": ex["age"], "expert": ex["expert"], "model": ex["model"],
              "flagged": [bool(c < cutoff) for c in ex["confidence"]]}
    dashboard = {"metrics": metrics, "nights": per_night, "stages": STAGES,
                 "confusion": cm.tolist(), "sample": sample}
    return metrics, dashboard
