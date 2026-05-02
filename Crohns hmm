"""
crohns_hmm.py
=============

Bayesian hidden Markov framework for time-to-flare estimation in
Crohn's disease under endogenous laboratory sampling. Simulation
study.

This file implements the full experimental pipeline described in the
companion paper:

  - Joint observation model with state-dependent wearable emissions,
    state-dependent laboratory-draw mechanism, and laboratory
    biomarker emissions when drawn (paper Eq. 2).
  - Forward filter that uses the laboratory-draw indicator L_d as
    evidence, including the (1 - lambda(i)) factor on lab-absent days
    (paper Proposition 1).
  - Maximum-likelihood parameter estimation with Laplace smoothing on
    lab-draw counts.
  - 60/20/40 train/validation/test split with thresholds tuned on
    validation, applied unchanged on test (no test-set leakage).
  - Per-patient median/MAD normalization variant for ablation.
  - Five baselines: naive threshold on CRP, logistic regression,
    random forest, pattern-mixture MNAR HMM, no-hidden-state naive
    Bayes.
  - Sensitivity sweep across lab-draw rate regimes.

Default lambda regime is the lab-sparse-remission setting
lambda = (0.02, 0.20, 0.40), which models the clinical reality that
remission patients are tested rarely (~once per 50 days) while flare
patients are tested frequently (every 2-3 days).

Run:

    python crohns_hmm.py > results/results.txt

Reproduces the headline numbers in Table II and Table III of the
paper (10 independent seeds, 120 patients x 120 days each).

All results are computational and use synthetic data only. No real
patient records are used. Real-data validation under IRB oversight is
essential before any clinical deployment.
"""

import numpy as np
from scipy.stats import norm as sp_norm, dirichlet
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =========================================================================
# CONSTANTS
# =========================================================================

K = 3
N_BIO = 7
LEAD_MIN = 1
LEAD_MAX = 7
STATE_NAMES = ['Remission', 'Mild', 'Moderate/Flare']

PI0 = np.array([0.40, 0.43, 0.17])

P_POP = np.array([
    [0.93, 0.06, 0.01],
    [0.05, 0.87, 0.08],
    [0.01, 0.09, 0.90],
])

POP_MEANS = np.array([
    [37.0, 37.1, 37.3],    # Temperature
    [70.0, 72.0, 76.0],    # Heart Rate
    [ 2.0,  4.0,  7.5],    # Stool frequency
    [ 4.0, 18.0, 55.0],    # CRP
    [10.0, 28.0, 55.0],    # ESR
    [ 6.5,  9.0, 13.0],    # WBC
    [ 0.0, -0.3, -1.1],    # delta-Hb
])

BIO_STDS = np.array([1.0, 20.0, 1.3, 6.0, 12.0, 2.0, 0.40])
PATIENT_SD = np.array([0.8, 15.0, 1.0, 5.0, 9.5, 1.6, 0.35])

# State-direct lab draw probabilities, lab-sparse-remission regime.
# Remission patients tested rarely (1-2x per quarter), Mild patients
# moderately (~1x per week), Flare patients frequently (~3x per week).
# This is closer to actual clinical practice than the baseline
# [0.07, 0.22, 0.38] used in the paper's original simulations.
LAMBDA_STATE = np.array([0.02, 0.20, 0.40])


# =========================================================================
# DATA GENERATION  (state-direct draws — matches paper's model)
# =========================================================================

def simulate_patients(n_patients, n_days, seed,
                      lambda_state=None,
                      P_override=None,
                      lab_missing_frac=None):
    rng    = np.random.RandomState(seed)
    P_use  = P_override if P_override is not None else P_POP
    lam    = LAMBDA_STATE if lambda_state is None else lambda_state
    pats   = []

    for _ in range(n_patients):
        P_n = np.vstack([
            dirichlet.rvs(12 * P_use[k] + 1e-3, random_state=rng)[0]
            for k in range(K)
        ])
        offset = rng.randn(N_BIO) * PATIENT_SD

        s = rng.choice(K, p=PI0); states = [s]
        for _ in range(n_days - 1):
            s = rng.choice(K, p=P_n[s]); states.append(s)
        states = np.array(states)

        X      = np.zeros((n_days, N_BIO))
        obs    = np.ones((n_days, N_BIO), dtype=bool)
        L_draw = np.zeros(n_days, dtype=bool)

        # Wearables observed every day
        for b in range(2):
            X[:, b] = rng.normal(POP_MEANS[b] + offset[b], BIO_STDS[b])[states]

        # Lab draws: state-direct (paper's model)
        for t in range(n_days):
            if lab_missing_frac is not None:
                draw = rng.random() > lab_missing_frac
            else:
                draw = rng.random() < lam[states[t]]
            L_draw[t] = draw

            if draw:
                for b in range(2, N_BIO):
                    X[t, b]   = rng.normal(POP_MEANS[b] + offset[b],
                                           BIO_STDS[b])[states[t]]
                    obs[t, b] = True
            else:
                X[t, 2:]   = np.nan
                obs[t, 2:] = False

        pats.append(dict(states=states, X=X, obs=obs,
                         L=L_draw, n_visits=n_days))
    return pats


# =========================================================================
# IMPUTATION
# =========================================================================

def fill_forward(X, obs):
    Xi = X.copy()
    for b in range(N_BIO):
        last = POP_MEANS[b, 0]
        for t in range(len(Xi)):
            if obs[t, b] and not np.isnan(Xi[t, b]):
                last = Xi[t, b]
            else:
                Xi[t, b] = last
    return Xi


def patient_relative_normalize(pats_or_seqs, return_stats=False):
    """Robust per-patient z-score (median / MAD) on each channel.
       This is Section 2.3 of the manuscript.  Operates on filled-forward
       sequences; produces a list of normalized per-patient arrays.

       The normalization is done per channel within each patient's own
       trajectory, so that patient baselines (offsets) cancel and only
       deviations from each patient's own median in MAD-units enter
       downstream estimation.
    """
    out = []
    stats = []
    for seq in pats_or_seqs:
        norm = np.zeros_like(seq)
        per_patient = []
        for b in range(seq.shape[1]):
            ch  = seq[:, b]
            med = np.median(ch)
            mad = np.median(np.abs(ch - med))
            scale = 1.4826 * mad if mad > 1e-9 else max(np.std(ch), 1e-6)
            norm[:, b] = (ch - med) / scale
            per_patient.append((med, scale))
        out.append(norm)
        stats.append(per_patient)
    if return_stats:
        return out, stats
    return out


def denormalize_means_for_filter(pop_means, patient_stats):
    """Given population-level state means in ORIGINAL units and a patient's
       (median, scale) pair per channel, returns the state means transformed
       to the patient's normalized coordinates.

       In normalized coordinates: x_norm = (x - med) / scale
       So the state-mean in normalized space is: (mu_state - med) / scale.
    """
    K_, B = pop_means.shape
    out = np.zeros_like(pop_means)
    for b in range(B):
        med, scale = patient_stats[b]
        out[:, b] = (pop_means[:, b] - med) / scale
    return out


def randomize_lab_timing(pats, seed):
    rng = np.random.RandomState(seed)
    out = []
    for p in pats:
        n  = len(p['states'])
        Xn = p['X'].copy()
        on = np.zeros((n, N_BIO), dtype=bool)
        Ln = np.zeros(n, dtype=bool)
        on[:, :2] = True
        Xn[:, :2] = p['X'][:, :2]
        # wearable values stay; lab draw days are permuted
        for b in range(2, N_BIO):
            drawn = np.where(p['obs'][:, b])[0]
            vals  = p['X'][drawn, b]
            Xn[:, b] = np.nan
            on[:, b] = False
            if len(drawn) > 0:
                new_days = rng.choice(n, size=len(drawn), replace=False)
                for d, v in zip(new_days, vals):
                    Xn[d, b]  = v
                    on[d, b]  = True
        # L_d follows the lab-channel observation flag (any of channels 2-6)
        Ln = on[:, 2:].any(axis=1)
        out.append(dict(states=p['states'], X=Xn, obs=on, L=Ln,
                        n_visits=p['n_visits']))
    return out


# =========================================================================
# MLE PARAMETER ESTIMATION (now also estimates LAMBDA_STATE)
# =========================================================================

def mle_params(tr_pats, seqs_imp):
    # Transition matrix
    tc = np.ones((K, K))
    for p in tr_pats:
        st = p['states']
        for t in range(len(st) - 1):
            tc[st[t], st[t + 1]] += 1
    P = tc / tc.sum(1, keepdims=True)

    # Emission means and stds
    mu  = np.zeros((K, N_BIO))
    sig = np.ones((K, N_BIO)) * 0.5
    for k in range(K):
        rows = [s[p['states'] == k]
                for p, s in zip(tr_pats, seqs_imp)
                if (p['states'] == k).any()]
        if rows:
            data   = np.vstack(rows)
            mu[k]  = np.nanmean(data, 0)
            sig[k] = np.nanstd(data, 0) + 0.05

    # Initial distribution
    pi = np.ones(K)
    for p in tr_pats:
        pi[p['states'][0]] += 1
    pi = pi / pi.sum()

    # Lab-draw probability per state (Laplace-smoothed)
    n_state    = np.zeros(K)
    n_state_L  = np.zeros(K)
    for p in tr_pats:
        st = p['states']
        L  = p['L']
        for k in range(K):
            mask = (st == k)
            n_state[k]   += mask.sum()
            n_state_L[k] += L[mask].sum()
    lam_hat = (n_state_L + 1.0) / (n_state + 2.0)
    return P, mu, sig, pi, lam_hat


# =========================================================================
# HMM CORE — emission_matrix and forward_filter now optionally include
#            the P(L_d | S_d) factor.
# =========================================================================

def emission_matrix(X, mu, sig, L=None, lam=None):
    """
    L   : (T,) bool — lab-draw indicator (optional)
    lam : (K,)     — P(L_d=1 | S_d=k)   (optional)
    If both provided, multiplies emission by P(L_d | S_d=k) factor.
    """
    T = len(X)
    log_E = np.zeros((T, K))
    for k in range(K):
        for b in range(N_BIO):
            valid = ~np.isnan(X[:, b])
            if valid.any():
                log_E[valid, k] += sp_norm.logpdf(
                    X[valid, b], mu[k, b], sig[k, b])

    if L is not None and lam is not None:
        log_lam = np.log(np.clip(lam,       1e-6, 1.0 - 1e-6))
        log_1ml = np.log(np.clip(1.0 - lam, 1e-6, 1.0 - 1e-6))
        # Add per-row factor: log_lam[k] when L_t=1, log_1ml[k] when L_t=0
        factor = np.where(L[:, None], log_lam[None, :], log_1ml[None, :])
        log_E += factor

    log_E -= log_E.max(axis=1, keepdims=True)
    E      = np.exp(log_E)
    E     /= E.sum(axis=1, keepdims=True) + 1e-300
    return E


def forward_filter(seq, P, mu, sig, pi, L=None, lam=None):
    E = emission_matrix(seq, mu, sig, L=L, lam=lam)
    T = len(seq)
    alpha = np.zeros((T, K))
    scale = np.zeros(T)

    alpha[0] = pi * E[0]
    scale[0] = alpha[0].sum() + 1e-300
    alpha[0] /= scale[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ P) * E[t]
        scale[t] = alpha[t].sum() + 1e-300
        alpha[t] /= scale[t]

    return alpha, np.log(scale + 1e-300).sum()


def viterbi(seq, P, mu, sig, pi, L=None, lam=None):
    E = emission_matrix(seq, mu, sig, L=L, lam=lam)
    T = len(seq)
    log_P = np.log(P + 1e-300)

    delta = np.zeros((T, K))
    psi   = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + 1e-300) + np.log(E[0] + 1e-300)

    for t in range(1, T):
        for k in range(K):
            scores      = delta[t - 1] + log_P[:, k]
            psi[t, k]   = np.argmax(scores)
            delta[t, k] = scores[psi[t, k]] + np.log(E[t, k] + 1e-300)

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def hmm_flare_scores(pats, seqs, P, mu, sig, pi, lam=None,
                     use_lab_indicator=True, flare_state=2,
                     patient_relative=False, mu_pop=None):
    """Computes P(S_t = flare | F_t) per day.
       use_lab_indicator=False reproduces the original code's behavior.

       If patient_relative=True, per-patient z-scoring is applied to each
       sequence, the state means are transformed to the patient's normalized
       coordinates, and the filter runs in normalized units. mu_pop should
       be the state means in original units (i.e., what was estimated from
       training data); the per-patient normalization makes those means
       patient-specific without re-estimating.
    """
    out = []
    if patient_relative:
        if mu_pop is None:
            mu_pop = mu
        norm_seqs, stats_list = patient_relative_normalize(
            seqs, return_stats=True)
        for p, seq, stats in zip(pats, norm_seqs, stats_list):
            L_arg = p['L'] if use_lab_indicator else None
            lam_arg = lam if use_lab_indicator else None
            mu_pat = denormalize_means_for_filter(mu_pop, stats)
            sig_pat = np.array([
                sig[b] / max(stats[b][1], 1e-6) for b in range(len(sig))
            ])
            alpha, _ = forward_filter(
                seq, P, mu_pat, sig_pat, pi, L=L_arg, lam=lam_arg)
            out.append(alpha[:, flare_state])
    else:
        for p, seq in zip(pats, seqs):
            L_arg = p['L'] if use_lab_indicator else None
            lam_arg = lam if use_lab_indicator else None
            alpha, _ = forward_filter(
                seq, P, mu, sig, pi, L=L_arg, lam=lam_arg)
            out.append(alpha[:, flare_state])
    return out


# =========================================================================
# EVALUATION METRICS
# =========================================================================

def compute_auroc(sc_lists, pats, flare_state=2):
    y_true = np.concatenate(
        [(p['states'] == flare_state).astype(int) for p in pats])
    y_sc   = np.concatenate(sc_lists)
    if y_true.sum() == 0 or np.std(y_sc) < 1e-9:
        return 0.5
    return roc_auc_score(y_true, y_sc)


def compute_lead_times(sc_lists, pats, thr, flare_state=2):
    lts = []
    for sc, p in zip(sc_lists, pats):
        st = p['states']
        T  = len(st)
        for d in range(1, T):
            if st[d] == flare_state and st[d - 1] != flare_state:
                lo  = max(0, d - LEAD_MAX)
                hi  = max(lo, d - LEAD_MIN + 1)
                hit = next((i for i in range(lo, hi) if sc[i] >= thr), None)
                if hit is not None:
                    lts.append(d - hit)
    return np.array(lts)


def optimal_threshold(sc_lists, pats, flare_state=2):
    y_true = np.concatenate(
        [(p['states'] == flare_state).astype(int) for p in pats])
    y_sc   = np.concatenate(sc_lists)
    fpr, tpr, thrs = roc_curve(y_true, y_sc)
    return thrs[np.argmax(tpr - fpr)]


def evaluate(sc, pats):
    """Original (leaky) evaluation that tunes threshold on the same data."""
    auc = compute_auroc(sc, pats)
    thr = optimal_threshold(sc, pats)
    lt  = compute_lead_times(sc, pats, thr)
    return dict(auc=auc, lt=lt.mean() if len(lt) > 0 else 0.0)


def evaluate_honest(sc_val, pats_val, sc_test, pats_test):
    """Tunes threshold on a separate VALIDATION split, evaluates on TEST.
       Removes the test-set leakage in optimal_threshold().

       sc_val/pats_val: scores and patients on validation set (for thr)
       sc_test/pats_test: scores and patients on test set (for AUROC + LT)
    """
    auc = compute_auroc(sc_test, pats_test)
    thr = optimal_threshold(sc_val, pats_val)
    lt  = compute_lead_times(sc_test, pats_test, thr)
    return dict(auc=auc, lt=lt.mean() if len(lt) > 0 else 0.0, thr=thr)


# =========================================================================
# BASELINES
# =========================================================================

def naive_threshold_scores(pats, channel=3):
    out = []
    for p in pats:
        seq = fill_forward(p['X'], p['obs'])
        crp = seq[:, channel]
        lo, hi = POP_MEANS[channel, 0], POP_MEANS[channel, 2]
        out.append(np.clip((crp - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def logistic_regression_scores(tr_pats, te_pats, seed=0):
    def feats(pats):
        rows, labs = [], []
        for p in pats:
            Xi = fill_forward(p['X'], p['obs'])
            rows.append(Xi)
            labs.extend((p['states'] == 2).astype(int))
        return np.vstack(rows), np.array(labs)
    Xtr, ytr = feats(tr_pats); Xte, _ = feats(te_pats)
    sc_v = StandardScaler()
    lr   = LogisticRegression(max_iter=2000, C=1.0,
                              class_weight='balanced', random_state=seed)
    lr.fit(sc_v.fit_transform(Xtr), ytr)
    raw = lr.predict_proba(sc_v.transform(Xte))[:, 1]
    out = []; idx = 0
    for p in te_pats:
        n = p['n_visits']; out.append(raw[idx:idx + n]); idx += n
    return out


def random_forest_scores(tr_pats, te_pats, seed=0, lag=3):
    def feats(pats):
        rows, labs = [], []
        for p in pats:
            Xi = fill_forward(p['X'], p['obs'])
            T  = len(Xi)
            for t in range(T):
                window = Xi[max(0, t - lag + 1):t + 1]
                pad    = np.zeros((lag - len(window), N_BIO))
                feat   = np.vstack([pad, window]).flatten()
                rows.append(feat)
                labs.append(int(p['states'][t] == 2))
        return np.vstack(rows), np.array(labs)
    Xtr, ytr = feats(tr_pats); Xte, _ = feats(te_pats)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    raw = rf.predict_proba(Xte)[:, 1]
    out = []; idx = 0
    for p in te_pats:
        n = p['n_visits']; out.append(raw[idx:idx + n]); idx += n
    return out


# =========================================================================
# MNAR BASELINE — Pattern-mixture HMM
# =========================================================================
# Standard pattern-mixture missing-not-at-random baseline (Little 1993,
# Daniels & Hogan 2008). The data are stratified by lab-draw indicator
# pattern; separate emission means and variances are estimated for each
# stratum (lab observed vs lab absent), and the day-d posterior is the
# stratum-weighted average. This is the standard alternative to the
# shared-parameter formulation used in our main HMM.

def pattern_mixture_scores(tr_pats, tr_seqs, te_pats, te_seqs):
    """Pattern-mixture HMM. Estimates separate emission means/variances for
       lab-observed (L=1) and lab-absent (L=0) days, fits separate transition
       matrices per stratum, and combines via marginal mixing weights.
    """
    K_ = K
    # Estimate emission parameters per (state, stratum)
    sums   = np.zeros((K_, 2, N_BIO))
    sqsums = np.zeros((K_, 2, N_BIO))
    counts = np.zeros((K_, 2, N_BIO))
    for p, seq in zip(tr_pats, tr_seqs):
        st = p['states']; L = p['L']
        for t in range(len(seq)):
            stratum = int(L[t])
            for b in range(N_BIO):
                if not np.isnan(seq[t, b]):
                    sums[st[t], stratum, b]   += seq[t, b]
                    sqsums[st[t], stratum, b] += seq[t, b] ** 2
                    counts[st[t], stratum, b] += 1.0
    counts = np.maximum(counts, 1.0)
    mu_pm  = sums / counts
    sig_pm = np.sqrt(np.maximum(
        sqsums / counts - mu_pm ** 2, 0.01)) + 0.01

    # Marginal stratum probability per state (for mixing weights at inference)
    n_state_L = np.zeros((K_, 2))
    for p in tr_pats:
        st = p['states']; L = p['L']
        for t in range(len(st)):
            n_state_L[st[t], int(L[t])] += 1.0
    pi_strat = (n_state_L + 1.0) / (n_state_L.sum(axis=1, keepdims=True) + 2.0)

    # Standard transition matrix and initial distribution from training data
    P, _, _, pi, _ = mle_params(tr_pats, tr_seqs)

    out = []
    for p, seq in zip(te_pats, te_seqs):
        T = len(seq); L = p['L']
        log_E = np.zeros((T, K_))
        for t in range(T):
            stratum = int(L[t])
            for k in range(K_):
                # Mixture: use the stratum-specific emission with weight pi_strat
                w0 = pi_strat[k, 0]; w1 = pi_strat[k, 1]
                lp0 = lp1 = 0.0
                for b in range(N_BIO):
                    if not np.isnan(seq[t, b]):
                        lp0 += sp_norm.logpdf(seq[t, b], mu_pm[k, 0, b], sig_pm[k, 0, b])
                        lp1 += sp_norm.logpdf(seq[t, b], mu_pm[k, 1, b], sig_pm[k, 1, b])
                # If we observed L_d, condition on the appropriate stratum
                if stratum == 1:
                    log_E[t, k] = lp1 + np.log(w1 + 1e-300)
                else:
                    log_E[t, k] = lp0 + np.log(w0 + 1e-300)

        # Forward filter using stratum-conditioned emissions
        log_alpha = np.zeros((T, K_))
        log_alpha[0] = np.log(pi + 1e-300) + log_E[0]
        log_alpha[0] -= log_alpha[0].max()
        for t in range(1, T):
            for k in range(K_):
                log_alpha[t, k] = (
                    log_E[t, k] + np.log(np.exp(log_alpha[t-1] -
                                                log_alpha[t-1].max()) @ P[:, k]
                                          + 1e-300) + log_alpha[t-1].max())
            log_alpha[t] -= log_alpha[t].max()
        alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        alpha /= alpha.sum(axis=1, keepdims=True)
        out.append(alpha[:, 2])
    return out


# =========================================================================
# EXPERIMENT 1 — synthetic comparison, including HMM with vs without P(L|S)
# =========================================================================

def run_experiment1(n_runs=10, n_patients=120, n_days=120, seed_base=11):
    """Path B: 3-way split (train/val/test), threshold tuned on validation,
       evaluation on test. Includes Patient-Relative HMM variant."""
    print("=" * 70)
    print("EXPERIMENT 1 — Synthetic Method Comparison (Path B)")
    print(f"  {n_runs} seeds, {n_patients} patients, {n_days} days each")
    print(f"  State-direct lambda = {LAMBDA_STATE}")
    print(f"  3-way split: 60 train / 20 val / 40 test")
    print(f"  Thresholds tuned on val, evaluated on test (no leakage)")
    print("=" * 70)

    methods = ['Naive Threshold', 'Logistic Regression', 'Random Forest',
               'Pattern-Mixture HMM (MNAR)',
               'HMM (no L|S)', 'HMM (with L|S)',
               'HMM (with L|S, pt-relative)']
    results = {m: {'auc': [], 'lt': []} for m in methods}

    n_tr  = 60
    n_val = 20

    for run in range(n_runs):
        seed = run * 53 + seed_base
        pats = simulate_patients(n_patients, n_days, seed)
        tr  = pats[:n_tr]
        val = pats[n_tr:n_tr + n_val]
        te  = pats[n_tr + n_val:]

        tr_ff  = [fill_forward(p['X'], p['obs']) for p in tr]
        val_ff = [fill_forward(p['X'], p['obs']) for p in val]
        te_ff  = [fill_forward(p['X'], p['obs']) for p in te]
        P, mu, sig, pi, lam_hat = mle_params(tr, tr_ff)

        scores_val  = {}
        scores_test = {}

        scores_val['Naive Threshold']  = naive_threshold_scores(val)
        scores_test['Naive Threshold'] = naive_threshold_scores(te)

        scores_val['Logistic Regression']  = logistic_regression_scores(
            tr, val, seed)
        scores_test['Logistic Regression'] = logistic_regression_scores(
            tr, te, seed)

        scores_val['Random Forest']  = random_forest_scores(tr, val, seed)
        scores_test['Random Forest'] = random_forest_scores(tr, te, seed)

        scores_val['Pattern-Mixture HMM (MNAR)'] = pattern_mixture_scores(
            tr, tr_ff, val, val_ff)
        scores_test['Pattern-Mixture HMM (MNAR)'] = pattern_mixture_scores(
            tr, tr_ff, te, te_ff)

        scores_val['HMM (no L|S)']  = hmm_flare_scores(
            val, val_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=False)
        scores_test['HMM (no L|S)'] = hmm_flare_scores(
            te, te_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=False)

        scores_val['HMM (with L|S)']  = hmm_flare_scores(
            val, val_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True)
        scores_test['HMM (with L|S)'] = hmm_flare_scores(
            te, te_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True)

        scores_val['HMM (with L|S, pt-relative)'] = hmm_flare_scores(
            val, val_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True,
            patient_relative=True, mu_pop=mu)
        scores_test['HMM (with L|S, pt-relative)'] = hmm_flare_scores(
            te, te_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True,
            patient_relative=True, mu_pop=mu)

        for m in methods:
            ev = evaluate_honest(scores_val[m], val,
                                 scores_test[m], te)
            results[m]['auc'].append(ev['auc'])
            results[m]['lt'].append(ev['lt'])

        if (run + 1) % 2 == 0:
            row = ' | '.join(f"{m[:8]}:{results[m]['auc'][-1]:.2f}"
                             for m in methods)
            print(f"  Run {run+1:>2}/{n_runs}  {row}")

    print(f"\n  Estimated lambda from last run: {lam_hat}")
    print(f"\n  {'Method':<32} {'AUROC':>16} {'Lead Time (days)':>20}")
    print('  ' + '-' * 70)
    for m in methods:
        a = np.array(results[m]['auc'])
        l = np.array(results[m]['lt'])
        print(f"  {m:<32} {a.mean():.3f} +/- {a.std():.3f}     "
              f"{l.mean():.2f} +/- {l.std():.2f}")
    return results


# =========================================================================
# ABLATION — same data, four conditions
# =========================================================================

def run_ablation(n_runs=10, n_patients=120, n_days=120, seed_base=11):
    """Path B ablation. Honest threshold tuning on validation split.
       'No Patient-Relative' is now a real ablation: same MLE estimates,
       same data, only the patient-relative normalization step is removed."""
    print("\n" + "=" * 70)
    print("ABLATION — Path B")
    print(f"  {n_runs} seeds, {n_patients} patients, {n_days} days each")
    print(f"  3-way split, threshold tuned on val, evaluated on test")
    print("=" * 70)

    conds = ['Full Model (L|S + pt-relative)',
             'No L|S factor',
             'No Patient-Relative',
             'No Endogenous Sampling (data-side)',
             'No Hidden State']
    raw = {c: {'auc': [], 'lt': []} for c in conds}

    n_tr  = 60
    n_val = 20

    for run in range(n_runs):
        seed = run * 41 + 7
        pats = simulate_patients(n_patients, n_days, seed)
        tr  = pats[:n_tr]
        val = pats[n_tr:n_tr + n_val]
        te  = pats[n_tr + n_val:]

        tr_ff  = [fill_forward(p['X'], p['obs']) for p in tr]
        val_ff = [fill_forward(p['X'], p['obs']) for p in val]
        te_ff  = [fill_forward(p['X'], p['obs']) for p in te]
        P, mu, sig, pi, lam_hat = mle_params(tr, tr_ff)

        # Full Model
        sc_full_v = hmm_flare_scores(
            val, val_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True,
            patient_relative=True, mu_pop=mu)
        sc_full_t = hmm_flare_scores(
            te, te_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True,
            patient_relative=True, mu_pop=mu)

        # No L|S factor (still patient-relative)
        sc_nols_v = hmm_flare_scores(
            val, val_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=False,
            patient_relative=True, mu_pop=mu)
        sc_nols_t = hmm_flare_scores(
            te, te_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=False,
            patient_relative=True, mu_pop=mu)

        # No Patient-Relative (still uses L|S factor)
        sc_nopr_v = hmm_flare_scores(
            val, val_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True,
            patient_relative=False)
        sc_nopr_t = hmm_flare_scores(
            te, te_ff, P, mu, sig, pi, lam_hat, use_lab_indicator=True,
            patient_relative=False)

        # No Endogenous Sampling (data-side: randomize lab timing)
        tr_rand  = randomize_lab_timing(tr,  seed + 100)
        val_rand = randomize_lab_timing(val, seed + 150)
        te_rand  = randomize_lab_timing(te,  seed + 200)
        tr_ff_r  = [fill_forward(p['X'], p['obs']) for p in tr_rand]
        val_ff_r = [fill_forward(p['X'], p['obs']) for p in val_rand]
        te_ff_r  = [fill_forward(p['X'], p['obs']) for p in te_rand]
        P_r, mu_r, sig_r, pi_r, lam_r = mle_params(tr_rand, tr_ff_r)
        sc_noendo_v = hmm_flare_scores(
            val_rand, val_ff_r, P_r, mu_r, sig_r, pi_r, lam_r,
            use_lab_indicator=True, patient_relative=True, mu_pop=mu_r)
        sc_noendo_t = hmm_flare_scores(
            te_rand, te_ff_r, P_r, mu_r, sig_r, pi_r, lam_r,
            use_lab_indicator=True, patient_relative=True, mu_pop=mu_r)

        # No Hidden State
        def naive_bayes(seqs):
            out = []
            for seq in seqs:
                E = emission_matrix(seq, mu, sig)
                nb = E * pi[None, :]
                nb /= nb.sum(1, keepdims=True) + 1e-300
                out.append(nb[:, 2])
            return out
        sc_nh_v = naive_bayes(val_ff)
        sc_nh_t = naive_bayes(te_ff)

        # Evaluate each condition with held-out threshold tuning
        cond_data = [
            (sc_full_v,   val,      sc_full_t,   te),
            (sc_nols_v,   val,      sc_nols_t,   te),
            (sc_nopr_v,   val,      sc_nopr_t,   te),
            (sc_noendo_v, val_rand, sc_noendo_t, te_rand),
            (sc_nh_v,     val,      sc_nh_t,     te),
        ]
        for c, (svv, pv, stt, pt) in zip(conds, cond_data):
            ev = evaluate_honest(svv, pv, stt, pt)
            raw[c]['auc'].append(ev['auc'])
            raw[c]['lt'].append(ev['lt'])

    print(f"\n  {'Condition':<40} {'AUROC':>16} {'Lead Time':>16}")
    print('  ' + '-' * 72)
    for c in conds:
        a = np.array(raw[c]['auc'])
        l = np.array(raw[c]['lt'])
        print(f"  {c:<40} {a.mean():.3f} +/- {a.std():.3f}    "
              f"{l.mean():.2f} +/- {l.std():.2f}")
    return raw


if __name__ == '__main__':
    res1 = run_experiment1(n_runs=10)
    abl  = run_ablation(n_runs=10)
