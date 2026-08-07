# Rishi's Step-by-Step Reproducibility Guide

## Purpose

The supplied paper is a reference implementation of the analysis. Rishi's task is not to edit the rejected paper until it resembles this PDF. His task is to rebuild the scientific pipeline so that the equations, code, generated data, fitting procedure, predictions, evaluation, uncertainty estimates, tables, captions, and conclusions all refer to the same model.

Each stage below has a **deliverable** and an **acceptance test**. Do not proceed to the next stage until the current acceptance test passes.

---

## Stage 0 - Preserve the rejected record

1. Tag the exact code used for the rejected submission.
2. Archive the submitted PDF, supplementary files, environment information, and every reviewer message.
3. Do not overwrite the old release or silently replace its outputs.
4. Create a new branch, for example `reviewer-reproducibility`, and a new results directory.

**Deliverable:** immutable rejected-submission tag and a clean analysis branch.

**Acceptance test:** another person can retrieve both the original paper and the exact code claimed to have generated it.

---

## Stage 1 - Build a reviewer audit ledger

Create a table with these columns:

- reviewer and comment number;
- exact quoted concern;
- affected equation, figure, table, prose paragraph, and code function;
- proposed correction;
- unit or integration test;
- old result;
- updated result;
- manuscript location;
- response-letter location;
- status.

Enter every concern separately. In particular, do not combine the Equation (7) contradiction, code-model discrepancies, censoring, clustering, comparator-output problem, and oracle fitting into one generic “methods revised” row.

**Deliverable:** `reviewer_audit.csv` or equivalent.

**Acceptance test:** every reviewer sentence has an owner, code location, manuscript location, and objective completion criterion.

---

## Stage 2 - Lock the computational environment

1. Create a fresh virtual environment.
2. Install exact versions from `requirements.lock.txt`, or build the supplied Docker image pinned to Python 3.13.5.
3. Set numerical-library threads to one:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

4. Record the Python version, operating system, BLAS implementation, and package versions.
5. Run the unit tests before touching the manuscript.

**Deliverable:** exact dependency lock, Dockerfile, Python/OS/BLAS environment manifest, and thread settings.

**Acceptance test:** `pytest -q` passes in a clean environment and a second machine can recreate it.

---

## Stage 3 - Repair the simulator before fitting anything

### 3.1 Fresh wearable noise

For patient `p`, day `d`, and state `i`, generate a new vector

```text
Y_w[p,d] ~ Normal(mu_w[i], diag(sd_w[i]^2))
```

at every patient-day. Do not draw one vector per state and reuse it on every day spent in that state.

### 3.2 Joint laboratory panels

Generate one binary panel indicator `L[p,d]`. If it is zero, every laboratory channel is absent. If it is one, generate the complete positive panel. Do not permute laboratory channels independently in an ablation.

### 3.3 Log-normal laboratory values

When `L[p,d] = 1`, generate

```text
log Y_l[p,d] ~ Normal(nu[i], diag(tau[i]^2)).
```

The fitting density must include the corresponding log-normal Jacobian. The Jacobian cancels in state posterior ratios on a fixed day, but retaining it keeps the implemented density equal to the written density.

### 3.4 Separate physiological and first-passage chains

The physiological flare state may permit recovery. Make flare absorbing only inside the landmark-specific first-passage transformation. Do not alter the simulated physiological trajectory merely to calculate time to flare.

**Deliverable:** updated simulator with a deterministic seed interface.

**Acceptance tests:** repeated days in one state have nonzero wearable variance; labs are all present or all absent as a panel; absent labs are stored as missing; the same seed exactly reproduces the same cohort.

---

## Stage 4 - Implement one coherent HMM likelihood

For state `i`, use

```text
b_i(O_d)
 = Gaussian(Y_w[d] | mu_w[i], sd_w[i])
   * lambda_i^L[d] * (1-lambda_i)^(1-L[d])
   * LogNormal(Y_l[d] | nu[i], tau[i])^L[d].
```

Consequences:

1. On `L[d] = 0`, the laboratory-value density is exactly one.
2. The no-draw HMM omits the Bernoulli factor but still evaluates observed laboratory values on draw days.
3. A causal last-observation-carried-forward laboratory value may be a comparator feature, but it must never enter the proposed HMM likelihood on a no-lab day.
4. The simulator and likelihood must use the same distributional family in the well-specified experiment.

**Deliverable:** one emission function called by filtering, smoothing, EM, and evaluation.

**Acceptance test:** replacing a stored lab value on a day with `L=0` leaves every state likelihood unchanged; replacing a value on `L=1` changes the likelihood.

---

## Stage 5 - Fit latent states rather than oracle labels

### 5.1 Initialization

- Standardize training wearables.
- Initialize three clusters without using simulator states.
- Order labels by a prespecified severity score based only on fitted observable means.

### 5.2 E-step

For every training patient, calculate

```text
gamma[p,d,i] = P(S[p,d]=i | all observed data)
xi[p,d,i,j]  = P(S[p,d]=i, S[p,d+1]=j | all observed data).
```

### 5.3 M-step

Update initial probabilities, transition rows, wearable means and variances, observed-log-lab means and variances, and state-specific draw probabilities from responsibility-weighted sufficient statistics. Use weak, declared pseudocounts or priors to avoid zero probabilities.

### 5.4 Convergence and label alignment

- Track observed-data log likelihood.
- Use fixed tolerance and iteration limits.
- Retain diagnostic logs.
- Reorder states only by observable severity, never by true simulator labels.
- In the final study, use more than one initialization or document why a single start is retained as a computational compromise.

**Deliverable:** Baum-Welch/EM fitting function.

**Acceptance tests:** fitting succeeds when the patient object raises an exception on any attempt to access `states_full` or `states_obs`; transition rows sum to one; fitted variances and draw probabilities remain in valid ranges.

---

## Stage 6 - Propagate parameter uncertainty honestly

The reference uses 16 responsibility-conditioned empirical-Bayes parameter draws plus the base EM estimate for each HMM:

- Dirichlet draws for initial and transition probabilities;
- beta draws for state-specific laboratory-draw rates;
- weakly regularized location/scale draws for Gaussian and log-Gaussian emissions;
- filtering and first-passage recomputed for every draw;
- equal-weight mixture of the EM estimate and parameter draws.

This is not full posterior sampling because the final EM responsibilities are held fixed. Rishi may reproduce this approximation or replace it with a patient bootstrap refit or full Bayesian model, but the manuscript must name the method accurately.

**Deliverable:** parameter ensemble and manifest of every draw.

**Acceptance test:** uncertainty in transitions, emissions, initial state, and draw probabilities changes the final PMF, not merely a table of standard errors.

---

## Stage 7 - Define the prediction target without ambiguity

At every landmark `d`, define

```text
H_d = inf{n >= 0 : S[d+n] = Flare}.
```

Let `q_d = (pi_d(Remission), pi_d(Mild))`, let `Q` be the transient block after flare is made absorbing, and let `r` be the transient-to-flare column.

### 7.1 Unconditional law

```text
P(H_d=0 | F_d) = pi_d(Flare)
P(H_d=n | F_d) = q_d' Q^(n-1) r, n>=1
P(H_d>h | F_d) = q_d' Q^h 1.
```

Its mean is

```text
E(H_d | F_d) = q_d' (I-Q)^(-1) 1.
```

This mean may approach zero.

### 7.2 Transient-conditional law

When transient mass is positive,

```text
E(H_d | F_d, S_d in {R,M})
 = q_d' (I-Q)^(-1) 1 / (q_d'1).
```

This is a convex combination of the two state-specific transient means and must remain between them. If transient mass is zero, display the quantity as undefined, not as zero.

### 7.3 Parameter-mixture conditioning

First marginalize over parameter draws, then condition on transient mass. Do not average draw-specific conditional means with equal weights unless their transient masses are identical.

**Deliverable:** PMF with columns `0,1,...,H,>H`, unconditional mean, and transient-conditional mean.

**Acceptance tests:** every PMF sums to one including its tail; day-zero mass equals the flare posterior; the conditional mean obeys its bound; the mixture conditional mean equals mixture unconditional mean divided by mixture transient mass.

---

## Stage 8 - Give every comparator the same output

Implement and evaluate:

1. HMM with the draw-indicator factor.
2. Otherwise identical HMM without that factor.
3. Draw-stratified-emission HMM with draw-stratum-specific wearable emissions and one transition matrix. This is a narrow comparator, not a universal pattern-mixture implementation.
4. A 17-member discrete-time event-history ensemble using only causal features. Before the first panel, its lab reference must be estimated from training data and paired with an explicit panel-history indicator; it must never use a simulator generating value.

If a comparator has one filtered state posterior and one transition matrix, apply the same first-passage transform. Do not exempt the strongest classifier from time-to-event evaluation merely because the original manuscript did not implement the transform.

For the event-history comparator, construct the day-zero flare probability and future discrete hazards, then combine them into one PMF with an explicit tail.

**Deliverable:** identical PMF schema for all models, complete serialized HMM ensembles, and complete serialized event-history coefficient/scaler ensembles.

**Acceptance test:** every model yields the same set of scoreable fields at every test landmark.

---

## Stage 9 - Evaluate censoring correctly

### 9.1 Censored logarithmic score

If an event is observed at waiting time `t`, score `-log p(t)`. If follow-up ends at `c` with no event, score `-log P(H_d>c)`. Do not remove no-flare patients and do not renormalize the PMF at day 120.

### 9.2 Fixed-horizon Brier scores

For administrative censoring, use landmarks with complete follow-up at 7, 14, 30, and 60 days. If random censoring is later introduced, use inverse-probability-of-censoring weighting or an explicitly justified alternative.

### 9.3 Coverage

The simulation supplies enough latent continuation to classify the outcome as `0,...,180` or `>180`. A no-flare outcome in the finite grid is the observed `>180` category, not a missing value. Equal-tailed intervals must be evaluated against that category.

### 9.4 Calibration and sharpness

Report:

- calibration intercept and slope at clinically relevant horizons;
- censored log score;
- horizon-specific and mean Brier scores;
- interval coverage;
- predictive entropy;
- current-state AUROC only as a secondary result.

**Deliverable:** landmark-level and patient-level metric files.

**Acceptance tests:** no `event_full` field is missing; censored landmarks have finite scores; every no-flare patient remains in the NLL analysis; AUROC is not presented as proof of time-to-flare performance.

---

## Stage 10 - Preserve the dependence structure in uncertainty estimates

For each bootstrap replicate:

1. sample simulation seeds with replacement;
2. within every selected seed, sample patients with replacement;
3. retain all repeated landmark rows for each selected patient;
4. calculate paired model differences on matched seed-patient units.

Use the same hierarchy for calibration slope and intercept. Never bootstrap 7,000 or more day-patient rows as though they were independent.

**Deliverable:** reproducible bootstrap seeds and confidence intervals.

**Acceptance test:** the bootstrap code samples seed and patient identifiers, not individual landmark rows.

---

## Stage 11 - Generate tables and figures from machine-readable outputs

1. Save exact fitted base objects, all HMM parameter-mixture members, all event-history bootstrap members, EM likelihood traces, feature names, scalers, coefficients, training-derived references, and resampling patient IDs for every seed.
2. Save complete landmark PMF tensors and state posteriors in addition to landmark summaries and patient metrics.
3. Generate every figure from those saved outputs.
4. Generate the manuscript tables from CSV/JSON files.
5. Fill the TeX template programmatically.
6. Do not type updated numbers directly into the manuscript.

The updated Figure 2 must show both means and make the distinction visible. The calibration figure must group by predicted risks, not true simulator states mislabeled as filter beliefs.

**Deliverable:** generated figures, tables, TeX, and PDF.

**Acceptance test:** deleting the output directory and running the build recreates every displayed number and image; an independent verifier regenerates every inferential table from canonical patient-level files and produces byte-identical TeX.

---

## Stage 12 - Reframe the empirical claims

The reference result supports the **validity of the first-passage interface**. The practical effect of the draw indicator must be stated exactly as the regenerated intervals show. The all-landmark unconditional estimand and the sensitivity analysis excluding current-flare landmarks must both be visible and must not be conflated.

Rishi must report that result directly. Sensitivity studies with overlapping emissions, more extreme draw contrasts, heterogeneity, heavy tails, or semi-Markov dynamics should be prespecified and reported as separate experiments, not used selectively to recover a desired conclusion.

**Deliverable:** cautious Results, Discussion, Abstract, and Conclusion.

**Acceptance test:** every claim is no stronger than the uncertainty interval and every null result is visible.

---

## Stage 12A - Audit the landmark estimand

The unconditional PMF is defined at every landmark, including days when the current state is already Flare. Report the fraction of such landmarks and add a separately labeled analysis excluding them. Do not replace the primary estimand after seeing the result.

**Deliverable:** `noncurrent_flare_performance.csv`, paired differences, calibration output, and manuscript table.

**Acceptance test:** both analyses are regenerated from the same canonical landmark file, and true states are used only to define the sensitivity subset after prediction.

---

## Stage 12B - Archive complete predictions and model objects

For every seed, archive:

- the complete PMF tensor over `0,...,180,>180` for every model, patient, and landmark;
- state posterior tensors and both time-to-flare means;
- base HMM fits and all 16 parameter draws;
- all three EM starts and their likelihood traces;
- base event-history fit and 16 patient-bootstrap refits;
- feature names, scalers, coefficients, training references, and sampled patient IDs.

**Acceptance test:** PMFs normalize, day-zero masses equal flare posteriors, and all archive checksums match the index.

---

## Stage 13 - Restore external data only after it can answer the question

Do not use the former IBDMDB section as validation unless all of the following are true:

1. every count is independently reproducible from individual-level data;
2. repeated visits are modeled with patient clustering or random effects;
3. the transition model is estimated on the actual visit cadence;
4. scheduled study draws are not described as symptom-triggered clinical ordering without evidence;
5. the flare outcome is independent of the principal predictor;
6. wearable data or a defensible substitute are available;
7. time-to-flare calibration is evaluated prospectively.

Until then, external data may motivate the model but cannot validate its PMF.

**Deliverable:** either a new defensible real-data analysis or no confirmatory real-data section.

**Acceptance test:** the paper does not claim clinical validation from a cadence-mismatched, circular, or unclustered analysis.

---

## Stage 14 - Integrate the missing literature

Discuss, rather than merely list:

- Bartolucci and Farcomeni (2015), discrete-time event-history modeling of informative dropout in mixed latent Markov models;
- Bartolucci and Farcomeni (2019), shared-parameter continuous-time hidden Markov and survival modeling;
- Marino and Alfò (2020), finite mixtures of HMMs for longitudinal responses subject to dropout.

State the contribution narrowly: the paper combines intermittent state-dependent laboratory observation with a first-passage clinical interface. It does not invent informative missingness, HMM filtering, latent Markov event-history modeling, or absorbing-chain mathematics.

**Deliverable:** revised related-work section and comparison narrative.

**Acceptance test:** a reader can identify how the method differs from each directly adjacent model class.

---

## Stage 15 - Prepare the response letter

For each reviewer point, include:

1. the comment;
2. agreement or a tightly reasoned partial disagreement;
3. the exact correction;
4. the new numerical result;
5. manuscript page, equation, figure, or table;
6. code file and function;
7. release tag and commit hash;
8. any conclusion that changed.

Use language such as “the submitted caption was incorrect” when it was incorrect. Do not call substantive code-paper discrepancies “clarifications for brevity.”

For the unexplained “inappropriate references” checkbox, ask the editor to identify the disputed references rather than guessing which citations to delete.

**Deliverable:** point-by-point response letter.

**Acceptance test:** every response is auditable against the new release.

---

## Final release gate

Run:

```bash
make reproduce
```

Then confirm:

- all tests pass;
- the exact ten-seed configuration is used;
- every output file is regenerated;
- every manuscript placeholder is filled from tables;
- the PDF compiles without errors;
- every page is visually inspected for clipping, overlaps, broken equations, or misleading captions;
- the release contains source, tests, lock file, Dockerfile, environment/BLAS manifest, configuration, complete PMFs, all model objects, patient metrics, tables, figures, TeX, PDF, Git source bundle, commit hash, release tag, and checksums;
- the public archive DOI is either a real assigned DOI or is explicitly marked as pending author deposit, never fabricated.

Only after all of these checks pass should Rishi prepare a resubmission.
