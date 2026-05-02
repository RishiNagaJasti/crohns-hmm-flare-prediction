# crohns-hmm-flare-prediction

Bayesian hidden Markov framework for time-to-flare estimation in Crohn's disease under endogenous laboratory sampling. **Simulation study.**

This repository contains the code accompanying the paper *"A Bayesian Hidden Markov Framework for Time-to-Flare Estimation in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study"* (Jasti, 2026).

> **All results in this repository are computational and use synthetic data only.** No real patient records are used anywhere in this work. Real-data validation, comparison against established missing-data methodology on real cohorts, and prospective clinical study under IRB oversight are essential before any deployment.

## What this code does

`crohns_hmm.py` implements the full experimental pipeline described in the paper:

- **Joint observation model.** State-dependent wearable emissions (Gaussian), a state-dependent laboratory-draw mechanism (Bernoulli with rate `lambda(i)` for state `i`), and laboratory biomarker emissions (log-normal) when drawn.
- **Forward filter with informative-missingness factor.** When no laboratory measurement is observed on a given day, the filter multiplies in `(1 - lambda(i))` for each state, so laboratory absence enters the posterior update as evidence rather than as ignorable missingness.
- **Hitting-time output.** Posterior over time until flare, derived analytically from the absorbing-Markov-chain fundamental matrix `(I - Q)^(-1)` and the filtering distribution.
- **Held-out evaluation.** 60/20/40 train/validation/test split, with classification thresholds tuned on the validation split and applied unchanged on the test split. No test-set leakage.
- **Five baselines.** Naive threshold on CRP, logistic regression on per-day features, random forest with lag features, a standard pattern-mixture MNAR HMM, and a no-hidden-state naive-Bayes condition.
- **Ablations and sensitivity sweep.** Component ablations (no `lambda` factor, no per-patient normalization, no endogenous sampling, no hidden state) and a sensitivity sweep across lab-draw-rate regimes.

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
git clone https://github.com/<your-username>/crohns-hmm-flare-prediction.git
cd crohns-hmm-flare-prediction
pip install -r requirements.txt
python crohns_hmm.py > results/results.txt
```

Runs in approximately 2–3 minutes on a modern laptop. Output reproduces Table II (discrimination and lead time) and Table III (ablation) of the paper, plus an end-of-run summary.

## Repository structure

```
crohns-hmm-flare-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── crohns_hmm.py          # Full pipeline: simulation, filter, baselines, ablation
├── figures/
│   ├── hmm_graphical_model.png
│   └── posterior_heatmap.png
└── results/
    └── results.txt        # Reference output from a 10-seed run
```

## Reproducibility notes

- All seeds are deterministic. `seed = run * 53 + 11` for run `i` in the headline experiment; `seed = run * 41 + 7` for the ablation. Setting `numpy` and `scipy` versions per `requirements.txt` should give bit-identical results.
- The simulation generates 120 patients per seed; per-seed data is split 60/20/40 into train/validation/test.
- Threshold for each baseline is chosen on the validation split to maximize Youden's index, then fixed and applied unchanged on the test split. This eliminates the test-set threshold leakage present in earlier development versions of the code.

## Honest scope statement

This is methods-and-simulation work. No clinical efficacy claim is made, and no real patient data is used. The framework is presented as a candidate decision-support tool that produces a posterior distribution over time-to-flare; whether this output translates to improved patient outcomes is an empirical question that real-data prospective studies under IRB oversight would need to answer.

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
  note         = {Code: \url{https://github.com/<your-username>/crohns-hmm-flare-prediction}}
}
```

## Contact

Rishi Jasti — jasti27r@ncssm.edu

North Carolina School of Science and Mathematics, Durham, NC, USA
