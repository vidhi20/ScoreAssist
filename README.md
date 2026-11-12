# ScoreAssist: AI Pre-Scoring for Sleep Studies

**An AI model scores the whole sleep study first. The sleep technologist reviews only the parts it is unsure about.**

Tested on **30 real nights scored by sleep experts** (36,839 thirty-second windows) from the public Sleep-EDF dataset. Runs fully offline, with no API key.

Dashboard: open `docs/index.html` (or the GitHub Pages link for this repo).

## The problem

After an overnight sleep study, a sleep technologist scores the recording by hand: every 30-second window gets a stage (wake, light sleep N1 and N2, deep sleep N3, or REM). A typical night is around 900 windows, and scoring takes a trained technologist a large part of a shift. That manual step slows down how fast patients get a diagnosis.

Fully automatic scoring is not the answer either. Clinics need a person accountable for the result, and models still make mistakes in predictable places.

**User:** sleep technologists and the physicians who sign off on their scoring.
**Job to be done:** "Score this study accurately without going through all 900 windows by hand."

## What ScoreAssist does

1. **Pre-scores every window** with a published, pretrained sleep-staging model (YASA, Vallat and Walker, eLife 2021), using one brain-wave (EEG) channel, one eye-movement (EOG) channel, age and sex.
2. **Scores its own confidence** on every window.
3. **Routes only low-confidence windows to the technologist.** One confidence cutoff for every night, set so about 20% of windows go to review.
4. **Shows the tradeoff** between review effort and accuracy, so a lab can pick its own setting.

## Results (30 nights, 10 per age group)

| Metric | Result |
|---|---|
| AI alone: agreement with expert scoring | **79.7%** (Cohen's kappa 0.72, "substantial") |
| Share of windows sent to the technologist | **20%** |
| Agreement after the technologist reviews those windows* | **89.4%** |
| Error rate in flagged windows vs unflagged | **48.6% vs 13.2%** (3.7x) |
| Model errors caught by reviewing 20% of windows | 47.9% |
| Review 30% / 40% / 50% of windows | 92.7% / 94.9% / 96.7% agreement |

\*Assumes the technologist corrects each reviewed window to the expert's stage.

**Where the model struggles**

| Stage | F1 score |
|---|---|
| Wake | 0.90 |
| N1 (light sleep) | **0.33** |
| N2 | 0.82 |
| N3 (deep sleep) | 0.69 |
| REM | 0.83 |

N1 is also the stage human scorers disagree on most, so it is the right place to keep a person in the loop.

**By patient age**

| Ages | AI alone | Flagged for review | After review |
|---|---|---|---|
| 25 to 34 | 83.5% | 18% | 92.1% |
| 50 to 60 | 79.2% | 19% | 88.5% |
| 66 to 101 | 77.4% | 22% | 88.3% |

The model is weaker on older patients, whose sleep is more fragmented. Confidence routing sends them more review automatically, which narrows the gap from 6.1 points to 3.7.

All numbers are written to `results/metrics.json` on every run.

## Product decisions

- **Assist, don't replace.** The technologist stays accountable. The AI's job is to shrink the pile, not sign the report.
- **Route by confidence, not by stage.** Errors cluster in low-confidence windows (48.6% wrong vs 13.2%), so confidence is a better signal than "always review N1".
- **Let the lab choose the dial.** Some labs will accept 20% review, others will want 40%. The review curve makes that tradeoff explicit.
- **Watch the hardest patients.** Older patients get lower AI accuracy. Showing results by age keeps that visible instead of hiding it in an average.

**Success metrics for a real pilot:** technologist minutes per study, time from study to signed report, agreement with a second scorer on a weekly audit sample, override rate on unflagged windows, and the agreement gap between age groups.

See `docs/PRD.md` for the one-page product brief.

## Run it

```bash
pip install -r requirements.txt
python select_nights.py   # picks 30 first-night recordings, 10 per age group
python run_pipeline.py    # downloads the nights from PhysioNet (~1.4 GB, first run only), scores them, rebuilds the dashboard
pytest -q                 # tests for epoch alignment and the review curve
```

## How it works

```
select_nights.py        picks 30 nights across three age groups from Sleep-EDF
scoreassist/score.py    loads each night, crops to sleep plus 30 min of wake on each side,
                        runs the pretrained model, lines AI output up with expert scoring
scoreassist/analyze.py  agreement, kappa, F1 by stage, confidence routing, review curve, age breakdown
run_pipeline.py         runs everything and writes results/ and docs/index.html
```

## Limits and next steps

- **Different sensor placement.** The model was trained on central EEG, and Sleep-EDF uses a forehead channel (Fpz-Cz). That likely lowers accuracy here. Next step: fine-tune on this montage.
- **Review is simulated.** The "after review" numbers assume perfect correction. A real pilot would time technologists and measure their corrections.
- **Sleep staging only.** Breathing events (apneas and hypopneas) drive the sleep apnea diagnosis. Extending confidence routing to event scoring is the next feature.
- Research prototype, not a medical device.

## Data and model credits

- Sleep-EDF Expanded: Kemp B. et al., IEEE Trans Biomed Eng 2000; Goldberger A. et al., PhysioNet, Circulation 2000.
- YASA: Vallat R., Walker M.P., "An open-source, high-performance tool for automated sleep staging", eLife 2021.

## Author

Vidhi Parekh · [LinkedIn](https://www.linkedin.com/in/vidhi-parekh) · [GitHub](https://github.com/vidhi20) · [Portfolio](https://vidhiparekh.com)
