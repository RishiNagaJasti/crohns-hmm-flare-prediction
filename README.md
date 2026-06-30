# crohns-hmm-flare-prediction

Bayesian hidden Markov framework for time-to-flare estimation in Crohn's disease under endogenous laboratory sampling. **Simulation study.**

This repository contains the code accompanying the paper *"A Bayesian Hidden Markov Framework for Time-to-Flare Estimation in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study"* (Jasti, 2026).

> **All results in this repository are computational and use synthetic data only.** No real patient records are used anywhere in the simulation portion of this work. The IBDMDB grounding in Section IX-I uses publicly available data from the NIH Integrative Human Microbiome Project. Real-data validation, comparison against established missing-data methodology on real cohorts, and prospective clinical study under IRB oversight are essential before any deployment.

## What this code does

The repository is split into two layers: the canonical simulation pipeline (`crohns_hmm.py`) that reproduces Tables II and III, and a small set of analysis scripts that operate on top of the simulation to reproduce Figures 3 through 6.

### Canonical simulation: `crohns_hmm.py`

This file implements the experimental pipeline behind Tables II and III:

- **Joint observation model.** State-dependent wearable emissions (Gaussian), a state-dependent laboratory-draw mechanism (Bernoulli with rate `lambda(i)` for state `i`), and laboratory biomarker emissions (log-normal) when drawn.
- **Forward filter with informative-missingness factor.** When no laboratory measurement is observed on a given day, the filter multiplies in `(1 - lambda(i))` for each state, so laboratory absence enters the posterior update as evidence rather than as ignorable missingness.
- **Hitting-time output.** Posterior over time until flare, derived analytically from the absorbing-Markov-chain fundamental matrix `(I - Q)^(-1)` and the filtering distribution.
- **Held-out evaluation.** 60/20/40 train/validation/test split, with classification thresholds tuned on the validation split and applied unchanged on the test split. No test-set leakage.
- **Five baselines.** Naive threshold on CRP, logistic regression on per-day features, random forest with lag features, a standard pattern-mixture MNAR HMM, and a no-hidden-state naive-Bayes condition.
- **Ablations and sensitivity sweep.** Component ablations (no `lambda` factor, no per-patient normalization, no endogenous sampling, no hidden state) and a sensitivity sweep across lab-draw-rate regimes.

### Analysis scripts (Sections IX-F through IX-I)

The following four scripts import the canonical simulator and produce the calibration, coverage, misspecification, and IBDMDB results reported in the paper:

- `calibration_analysis.py`  -- Figure 3 (state-conditional calibration of the time-to-flare posterior; Section IX-F).
- `coverage_analysis.py`     -- Figure 4 (full predictive distribution and coverage diagnostic; Section IX-G).
- `misspec_analysis.py`      -- Figure 5 (heavy-tailed emission misspecification check; Section IX-H).
- `ibdmdb_analysis.py`       -- Figure 6 (empirical MNAR grounding on the IBDMDB cohort; Section IX-I).
- `gen_figures.py`           -- convenience orchestrator that runs all four.

## Headline result

| Method                              | AUROC             | Mean Lead Time (d) |
|-------------------------------------|-------------------|--------------------|
| Naive Threshold (CRP)               | 0.915 ± 0.018     | 6.56 ± 0.20        |
| Logistic Regression                 | 0.920 ± 0.021     | 6.52 ± 0.24        |
| Random Forest                       | 0.915 ± 0.022     | 6.33 ± 0.27        |
| **Pattern-Mixture HMM (MNAR)**      | **0.939 ± 0.017** | **6.55 ± 0.24**    |
| HMM (no λ factor)                   | 0.912 ± 0.024     | 6.59 ± 0.20        |
| HMM (with λ factor, proposed)       | 0.916 ± 0.023     | 6.60 ± 0.21        |

Mean ± std over 10 independent seeds, 120 patients × 120 days each. Lab-draw rates `λ = (0.02, 0.20, 0.40)` for Remission, Mild, Flare states respectively.

The proposed framework does not improve classification AUROC over established baselines. The pattern-mixture MNAR baseline outperforms it. The framework's distinguishing contribution is its time-to-flare posterior, derived analytically from the absorbing-chain dynamics, which the per-day baselines and pattern-mixture model do not produce as a primary output.

## Requirements

- Python 3.9 or newer
- See `requirements.txt`

## Usage

```bash
git clone https://github.com/RishiNagaJasti/crohns-hmm-flare-prediction.git
cd crohns-hmm-flare-prediction
pip install -r requirements.txt

# Reproduce Tables II and III (~3 minutes)
python crohns_hmm.py > results/results.txt

# Reproduce Figures 3-6 (~5 minutes)
python gen_figures.py
```

Outputs of `gen_figures.py` are written to `./figures/`.

## Reproducibility note

The analysis scripts in this repository re-run the canonical simulation in `crohns_hmm.py` and re-derive the calibration, coverage, and misspecification results on the resulting day-patient pairs. The headline AUROC and lead-time numbers in Tables II and III reproduce bit-for-bit across runs, as the per-seed RNG initialization is fully deterministic and the evaluation only depends on whether scores cross a fixed validation-tuned threshold.

The numbers reported in Sections IX-F, IX-G, and IX-H (state-conditional mean realized leads, empirical coverage at nominal levels, pre-flare day-patient pair counts) are approximate to within roughly one to two units across re-runs. Three sources contribute to this:

1. The per-patient transition matrix is drawn from `Dirichlet(12 * P_pop + 1e-3)` once per patient. This Dirichlet step is sensitive to floating-point ordering, and small differences in NumPy or SciPy versions can shift individual patient trajectories by a few days, changing the total number of pre-flare day-patient pairs by tens to low hundreds out of roughly eight thousand.
2. The calibration analysis groups pre-flare pairs by the simulator's ground-truth latent state (the quantity being calibrated against), which the paper text refers to as "filter argmax" for brevity. The qualitative conclusion (state-conditional means within roughly five days of the population predictions) reproduces robustly under either grouping; the precise pair counts and the M-state mean differ slightly between the two.
3. The misspecification analysis re-simulates the test split with Student's *t* noise. The marginal sample variance of *t*-distributed noise has a small finite-sample bias, which produces minor per-run variation in the pre-flare pair count between the Gaussian and *t*-distributed conditions.

The IBDMDB analysis in Section IX-I uses a static published metadata file and reproduces exactly. The Tables II and III simulation results reproduce exactly.

## Repository structure

```
crohns-hmm-flare-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── crohns_hmm.py              # Full simulation pipeline; reproduces Tables II, III
├── calibration_analysis.py    # Figure 3 (Section IX-F)
├── coverage_analysis.py       # Figure 4 (Section IX-G)
├── misspec_analysis.py        # Figure 5 (Section IX-H)
├── ibdmdb_analysis.py         # Figure 6 (Section IX-I); downloads IBDMDB metadata
├── gen_figures.py             # Runs all four analysis scripts in sequence
├── figures/                   # Output directory for analysis-script figures
└── results/
    └── results.txt            # Reference output from a 10-seed crohns_hmm.py run
```

## Reproducibility notes (simulation)

- All seeds are deterministic. `seed = run * 53 + 11` for run `i` in the headline experiment; `seed = run * 41 + 7` for the ablation. Setting `numpy` and `scipy` versions per `requirements.txt` should give bit-identical Tables II and III.
- The simulation generates 120 patients per seed; per-seed data is split 60/20/40 into train/validation/test.
- Threshold for each baseline is chosen on the validation split to maximize Youden's index, then fixed and applied unchanged on the test split. This eliminates the test-set threshold leakage present in earlier development versions of the code.

## Honest scope statement

This is methods-and-simulation work. No clinical efficacy claim is made, and no real patient data is used in the simulation portion. The IBDMDB grounding in Section IX-I uses publicly available data from a peer-reviewed published cohort. The framework is presented as a candidate decision-support tool that produces a posterior distribution over time-to-flare; whether this output translates to improved patient outcomes is an empirical question that real-data prospective studies under IRB oversight would need to answer.

## License

MIT. See `LICENSE`.

## Citation

If you use this code or framework in your work, please cite the accompanying paper:

```
@misc{Jasti2026crohnshmm,
  author       = {Rishi Jasti},
  title        = {A Bayesian Hidden Markov Framework for Time-to-Flare Estimation in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study},
  year         = {2026},
  howpublished = {Manuscript},
  note         = {Code: \url{https://github.com/RishiNagaJasti/crohns-hmm-flare-prediction}}
}
```

## Contact

Rishi Jasti -- jasti27r@ncssm.edu

North Carolina School of Science and Mathematics, Durham, NC, USA
