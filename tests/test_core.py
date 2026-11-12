import numpy as np
import mne

from scoreassist.analyze import review_curve
from scoreassist.score import expert_epochs


def test_expert_epochs_alignment_and_stage_merge():
    ann = mne.Annotations(onset=[0, 60, 120, 150], duration=[60, 60, 30, 30],
                          description=["Sleep stage W", "Sleep stage 4", "Movement time", "Sleep stage R"])
    labels = expert_epochs(ann, start=0, n_epochs=6)
    assert labels == ["W", "W", "N3", "N3", None, "R"]


def test_expert_epochs_respects_crop_start():
    ann = mne.Annotations(onset=[0, 90], duration=[90, 60], description=["Sleep stage W", "Sleep stage 2"])
    assert expert_epochs(ann, start=60, n_epochs=3) == ["W", "N2", "N2"]


def test_review_curve_is_monotonic_and_reaches_full_agreement():
    rng = np.random.default_rng(0)
    expert = rng.choice(["W", "N2", "R"], 500)
    model = expert.copy()
    wrong = rng.random(500) < 0.2
    model[wrong] = "N1"
    conf = np.where(wrong, rng.uniform(0.3, 0.6, 500), rng.uniform(0.5, 1.0, 500))
    curve = review_curve(expert, model, conf)
    agree = [p["agreement"] for p in curve]
    assert agree == sorted(agree)
    assert curve[0]["agreement"] == round(float((expert == model).mean()), 4)
    assert curve[-1]["errors_caught"] > 0.9
