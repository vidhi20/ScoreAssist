"""Score one Sleep-EDF night with a pretrained sleep-staging model and line it up with the expert's scoring.

Model: YASA's published LightGBM sleep-staging classifier (Vallat and Walker, eLife 2021), bundled with
the `yasa` package and run fully offline. Inputs: one EEG channel (Fpz-Cz), one EOG channel, age and sex.
"""

import mne
import numpy as np
import yasa

STAGES = ["W", "N1", "N2", "N3", "R"]
EXPERT_MAP = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "R",
}
STAGE_ALIAS = {"WAKE": "W", "REM": "R", "N1": "N1", "N2": "N2", "N3": "N3"}
EPOCH = 30.0
WAKE_PAD = 30 * 60  # keep 30 minutes of wake before and after sleep, the standard practice for this dataset


def expert_epochs(annotations, start, n_epochs):
    """Expert stage for each 30-second epoch from `start`; None where unscored or movement."""
    labels = [None] * n_epochs
    for onset, dur, desc in zip(annotations.onset, annotations.duration, annotations.description):
        stage = EXPERT_MAP.get(desc)
        if stage is None:
            continue
        first = int(round((onset - start) / EPOCH))
        for i in range(first, first + int(round(dur / EPOCH))):
            if 0 <= i < n_epochs:
                labels[i] = stage
    return labels


def score_night(psg_path, hyp_path, age, male):
    raw = mne.io.read_raw_edf(psg_path, include=["EEG Fpz-Cz", "EOG horizontal"], preload=True, verbose="error")
    ann = mne.read_annotations(hyp_path)

    # Crop to the sleep period plus 30 minutes of wake on each side.
    sleep = [(o, o + d) for o, d, s in zip(ann.onset, ann.duration, ann.description)
             if EXPERT_MAP.get(s) not in (None, "W")]
    tmin = max(0.0, sleep[0][0] - WAKE_PAD)
    tmax = min(raw.times[-1], sleep[-1][1] + WAKE_PAD)
    tmin = np.floor(tmin / EPOCH) * EPOCH
    raw.crop(tmin=tmin, tmax=tmax)

    sls = yasa.SleepStaging(raw, eeg_name="EEG Fpz-Cz", eog_name="EOG horizontal",
                            metadata={"age": int(age), "male": bool(male)})
    hyp = sls.predict()
    pred_stages = [STAGE_ALIAS.get(str(s), str(s)) for s in hyp.hypno.to_numpy()]
    conf = hyp.proba.max(axis=1).to_numpy()

    expert = expert_epochs(ann, tmin, len(pred_stages))
    keep = [i for i, e in enumerate(expert) if e is not None]
    return {
        "expert": [expert[i] for i in keep],
        "model": [pred_stages[i] for i in keep],
        "confidence": conf[keep].round(4).tolist(),
    }
