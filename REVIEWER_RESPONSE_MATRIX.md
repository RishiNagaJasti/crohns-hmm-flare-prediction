# Reviewer-to-Correction Matrix

| Reviewer concern | Implemented correction | Auditable evidence |
|---|---|---|
| Equation (7) conditions on transient states, yet Figure 2 approaches zero. | Defines `H_d`; separates the unconditional law with `P(H_d=0)=pi_d(F)` from the transient-conditional law; proves the conditional bound; plots both. | Equations, Figure 2, PMF and bound tests. |
| Calibration code grouped by true simulator state rather than the stated predictor. | Uses predicted 30-day risks, quantile calibration curves, and logistic calibration intercept/slope. | `risk_30`, `y_30`, calibration tables, generated figure. |
| Forward-filled absent labs entered the HMM likelihood. | Laboratory density contributes only when `L_d=1`; absent stored values are mathematically irrelevant. | Exact invariance unit test. |
| Code used raw Gaussian labs despite a log-normal equation. | Simulator and likelihood use Gaussian log-laboratory values, including the log-normal Jacobian. | Shared simulator/likelihood family and observed-value test. |
| Exact table-generating code was absent. | One pipeline writes canonical predictions, complete PMFs, model objects, tables, figures, and generated TeX. | `make reproduce`, independent table regeneration, byte-identical TeX check. |
| Coverage discarded patients without a flare and renormalized at 120 days. | Uses the censoring survival probability and preserves the `>180` tail. | Finite censored log scores; complete tail outcome archive. |
| Thousands of correlated patient-days were treated as independent. | Resamples seeds, then patients, preserving all landmarks within a selected patient. | Hierarchical bootstrap code and deterministic bootstrap counts. |
| The no-lambda model was not tested on the primary output. | Draw and no-draw HMMs both yield full first-passage PMFs and are compared by proper time-to-event scores. | Paired proper-score table and PMF archive. |
| AUROC did not evaluate the paper's principal claim. | AUROC is secondary; censored log score, Brier score, calibration, coverage, and entropy are primary. | Main results table and discussion. |
| Training used true simulator states. | Primary HMMs use multi-start Baum-Welch EM with observable-severity label ordering. | Oracle-access sentinel test; archived EM traces and selected/rejected starts. |
| Parameter uncertainty was omitted. | Each HMM uses 16 responsibility-conditioned parameter draws plus the base fit; every prediction is averaged after re-filtering. | 17-member serialized ensembles and PMFs. |
| Event-history comparator used a generating remission laboratory value before the first panel. | Pre-panel log-lab reference is the training-cohort median; an explicit `prior_panel_observed` feature identifies whether actual history exists. | Stored reference, feature names, deterministic reference check; source contains no generator initialization. |
| Event-history uncertainty was a single plug-in fit. | Adds 16 patient-bootstrap refits plus the base logistic current-state/hazard fit. | 17 serialized coefficient/scaler objects and bootstrap patient IDs. |
| “Pattern-mixture HMM” overstated the implemented comparator. | Renames it `Draw-stratified-emission HMM` and states its narrow selection-pattern scope. | Manuscript terminology and code model label. |
| Full predictive distributions and model objects were not archived. | Stores per-seed PMF/state tensors and all fitted HMM/hazard objects, traces, scalers, coefficients, references, and resampling IDs. | `prediction_distributions/`, `model_artifacts/`, SHA-256 indices. |
| Printed confidence intervals did not reproduce from the included code. | Canonical files are reread before aggregation; the release verifier independently regenerates all inferential tables and generated TeX. | `verify_release.py` exact numerical and byte comparisons. |
| The primary landmark population included current-flare rows without a sensitivity analysis. | Declares the all-landmark unconditional estimand and separately excludes current-flare landmarks. | Table IV and `noncurrent_flare_*` outputs. |
| Environment and release identity were incomplete. | Adds exact package lock, Dockerfile, environment/BLAS manifest, Git source bundle, release tag, source commit, and checksums. | `requirements.lock.txt`, `Dockerfile`, `release_metadata.json`, `MANIFEST.sha256`. |
| Requested latent-Markov/dropout literature was missing. | Discusses Bartolucci-Farcomeni (2015, 2019) and Marino-Alfo (2020), distinguishing intermittent observation from terminal dropout. | Related-work section and bibliography. |
| Real-data section overstated validation. | Defers confirmatory IBDMDB claims until cadence, clustering, independent outcome, and prospective validation conditions are satisfied. | Deferred external-validation subsection. |
| Reviewer marked some references inappropriate without identifying them. | Does not guess or silently delete citations; response letter should ask the editor to identify the disputed references. | Response-letter instruction in the reproducibility guide. |

## Updated conclusion

The analysis validates the mathematical first-passage interface. Whether the explicit draw indicator improves practical prediction is an empirical question; the main and non-current-flare analyses are reported without suppressing null or small effects.
