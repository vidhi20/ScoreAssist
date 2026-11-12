# ScoreAssist: one-page product brief

**Problem.** Sleep technologists score every sleep study by hand, about 900 thirty-second windows a night. It slows time to diagnosis, and fully automatic scoring is not trusted enough to sign.

**User.** Sleep technologists, and the sleep physicians who sign off on scored studies.

**Desired outcome.** Technologists spend their time only where judgment is needed, studies are signed faster, and accuracy matches or beats manual scoring.

**Assumptions tested in this prototype.**
1. A pretrained model gets most windows right. *Result: 79.7% agreement, kappa 0.72.*
2. Model confidence tells us where the mistakes are. *Result: flagged windows are wrong 3.7x as often as unflagged ones.*
3. Reviewing a small share closes most of the gap. *Result: reviewing 20% lifts agreement to 89.4%.*
4. Performance is even across patients. *Result: not true. Older patients score lower, so they need more review.*

**What ships in v1.** AI pre-scoring, confidence-based review queue, lab-adjustable review setting, results by patient group.

**Success measures.**
- Technologist minutes per study
- Time from study to signed report
- Agreement with a second scorer on a weekly audit sample
- Override rate on windows the AI did not flag
- Accuracy gap between age groups

**Out of scope for v1.** Breathing-event scoring, EHR integration, billing.

**Risks.** Sensor differences between labs, automation bias (technologists trusting unflagged windows too much), and regulatory requirements for clinical decision-support software. Any real deployment needs HIPAA-compliant data handling.
