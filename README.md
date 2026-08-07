# Updated Crohn's HMM Reference Reconstruction

This package is an audited reference implementation of the rejected Crohn's disease time-to-flare manuscript. It preserves the IEEE two-column article format while making the equations, simulator, fitted models, prediction targets, uncertainty analysis, figures, tables, and archived code describe the same analysis.

The paper is a reproducibility target for Rishi, not a substitute for his own verification and authorship decisions.

## Central corrections

The release now:

1. distinguishes the unconditional waiting-time law, including a day-zero flare mass, from the transient-conditional law and its convex-combination bound;
2. uses fresh patient-day wearable noise, jointly observed log-normal laboratory panels, and no forward-filled laboratory value in the HMM likelihood;
3. fits latent states by multi-start Baum-Welch EM without simulator labels;
4. propagates HMM parameter variation through a 17-member responsibility-conditioned empirical-Bayes mixture for each HMM;
5. evaluates a 17-member patient-bootstrap event-history ensemble whose pre-panel laboratory reference is estimated only from training data and is accompanied by an explicit panel-history indicator;
6. retains no-flare patients, the unrenormalized survival tail, and the complete `0,...,180,>180` predictive distribution;
7. compares all four models on the same time-to-flare output with censored log scores, Brier scores, calibration, coverage, entropy, and secondary AUROC;
8. uses a hierarchical seed-then-patient bootstrap and reports a separate sensitivity analysis excluding landmarks already in flare;
9. archives complete PMF tensors, state posteriors, all HMM mixture members, every event-history bootstrap model, scalers, coefficients, EM traces, resampling identifiers, and canonical tables;
10. verifies that the generated manuscript agrees with independently regenerated tables to displayed precision.

## Main files

- `Crohns_HMM_Time_to_Flare_Study.pdf` - IEEE-style reference manuscript.
- `Crohns_HMM_Time_to_Flare_Study.tex` - generated LaTeX source.
- `manuscript_template.tex` - numerical template filled only from canonical outputs.
- `crohns_hmm_pipeline.py` - simulation, fitting, prediction, evaluation, and archiving.
- `finalize_outputs.py` and `generate_figures_from_outputs.py` - deterministic finalization workers.
- `tests/test_pipeline.py` - mathematical and implementation tests.
- `verify_release.py` - independent release gate.
- `final_outputs/` - canonical predictions, PMFs, model objects, tables, and figures.
- `REPRODUCIBILITY_GUIDE.md` - staged instructions for Rishi.
- `REVIEWER_RESPONSE_MATRIX.md` - reviewer demand mapped to correction and evidence.
- `Dockerfile`, `requirements.lock.txt`, `environment_manifest.json` - reproducible environment materials.
- `release_metadata.json`, `release_source.bundle`, and `MANIFEST.sha256` - immutable local source identity and file checksums.

## Exact manuscript configuration

```text
simulation seeds                    10
patients per seed                   120
training / validation / test        65 / 15 / 40
observed days                       120
latent continuation                 240 days
finite PMF grid                     0,...,180 plus >180 tail
HMM parameter draws                 16 plus base EM estimate
hazard patient-bootstrap refits     16 plus base fit
primary EM initializations          3
maximum EM iterations               35
performance bootstrap replicates    2000
calibration bootstrap replicates    500
base seed                           1729
```

## Reproduce

A locked local environment:

```bash
make reproduce
```

A containerized reproduction:

```bash
docker build -t crohns-hmm-study .
docker run --rm -v "$PWD":/work crohns-hmm-study
```

A reduced smoke run:

```bash
make quick
```

The quick run is a structural test only; its values are not manuscript results.

## Release identity

`release_metadata.json` records the source commit, local release tag, Git bundle checksum, pipeline checksum, template checksum, generated TeX checksum, generated PDF checksum, and output-configuration checksum. The public archive DOI is deliberately marked `PENDING_AUTHOR_DEPOSIT`; a DOI must be assigned by an actual repository and must never be invented.

## Scientific scope

This remains a controlled simulation study. The state emissions are well separated, the principal HMM is correctly specified, the empirical-Bayes mixture conditions on final EM responsibilities, and the real-data IBDMDB illustration has been deferred rather than presented as clinical validation. The paper's claims are intentionally narrower than those in the rejected version.
