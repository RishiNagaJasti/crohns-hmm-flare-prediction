# Implementation and Verification Summary

## Release identity

- Source commit: recorded in `release_metadata.json`
- Local release tag: `time-to-flare-study-v2.1.0`
- Public archive DOI: `PENDING_AUTHOR_DEPOSIT`
- Manuscript: `Crohns_HMM_Time_to_Flare_Study.pdf`
- Exact source snapshot: `release_source.bundle`

A public DOI is intentionally not fabricated. The author must deposit this immutable release in an appropriate archive and replace the pending field before submission.

## Reviewer requirements implemented

1. The manuscript now distinguishes the unconditional waiting-time law, including the day-zero flare mass, from the law conditional on a transient current state.
2. The simulator, equations, and HMM likelihood use the same distributions: fresh patient-day wearable noise, jointly observed log-normal laboratory panels, and no laboratory-density term when no panel is drawn.
3. Primary HMM fitting uses latent-state Baum-Welch expectation maximization rather than simulator labels, with three observable-data starts and archived likelihood traces.
4. Parameter variation is propagated through 16 responsibility-conditioned draws plus the EM estimate for each HMM; the event-history model uses 16 patient-bootstrap refits plus its base fit.
5. No-flare patients remain in the censored evaluation, finite PMFs retain the `>180` tail, and uncertainty resamples simulation seeds and then patients.
6. The draw, no-draw, draw-stratified-emission, and event-history models all produce the same time-to-flare PMF schema and are compared with proper time-to-event scores.
7. The event-history comparator initializes pre-panel laboratory history from the training cohort, not from a generating parameter, and includes an explicit prior-panel indicator.
8. Complete PMFs, fitted HMM ensembles, event-history coefficients/scalers/bootstrap members, prediction summaries, and patient-level metrics are archived.
9. A non-current-flare sensitivity analysis is reported because 31.0% of all-landmark proposed-model rows are current-flare landmarks.
10. The directly relevant latent-Markov, informative-dropout, and shared-parameter literature requested by the reviewers is integrated into the argument.

## Final verification completed

- `pytest`: **16 passed**.
- Independent release gate: **passed**.
- Canonical-table regeneration: **passed**.
- Generated TeX byte-for-byte comparison: **passed**.
- Complete PMF normalization and day-zero-mass checks: **passed**.
- Archived model-object and ensemble checks: **passed**.
- Event-history training-reference and portability checks: **passed**.
- Manifest checksum verification: **passed**.
- PDF compilation and page-by-page visual inspection: **passed**.

The release verifier reports:

> Release verification passed: complete PMFs, model objects, canonical tables, TeX, PDF, and release identity agree.

## Principal result

The updated all-landmark analysis gives the draw-indicator HMM a censored log score of 2.261 (95% CI 2.162-2.370), four-horizon mean Brier score of 0.1199 (0.1118-0.1278), and nominal-90% interval coverage of 94.6% (92.9%-96.0%). Relative to the otherwise identical no-draw HMM, the paired differences are +0.0010 in log score (-0.0001 to +0.0021) and +0.00010 in mean Brier score (-0.00011 to +0.00031), with positive values favoring the draw model. The primary conclusion is therefore cautious: the first-passage interface is coherent, but this well-separated simulation does not establish a practically important general gain from the draw indicator.

The non-current-flare sensitivity analysis finds a small positive log-score difference but an uncertain Brier-score difference. It is reported separately because it answers a different estimand question.
