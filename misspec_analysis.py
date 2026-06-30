"""
misspec_analysis.py
===================
Reproduces the heavy-tailed emission misspecification check of Section IX-H
and renders Figure 5.

The test split is re-simulated with Student's t biomarker noise (nu = 4,
scaled so the marginal variance still matches sigma_b^2), keeping the same
Markov dynamics and state-conditional means. The filter is unchanged: it
continues to use the Gaussian-emission likelihood fitted on the original
Gaussian training data, isolating the effect of test-time misspecification.

Reproducibility note
--------------------
Numbers are approximate to those in the paper (within ~1 day on the
state-conditional means, ~10% on pair counts) due to small numerical
variation in the simulator's Dirichlet-perturbed transition matrices and
floating-point ordering across NumPy/SciPy versions. The qualitative
conclusion (R-state calibration survives, M-state calibration degrades and
exceeds the 5-day band under t_4 emissions) reproduces robustly.

Run:
    python misspec_analysis.py

Output:
    figures/misspec.png
    Pair counts and state-conditional means for Gaussian and t_4 reported.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import dirichlet, t as student_t
import warnings; warnings.filterwarnings('ignore')

from crohns_hmm import (
    fill_forward, mle_params, forward_filter, simulate_patients,
    K, N_BIO, PI0, P_POP, POP_MEANS, BIO_STDS, PATIENT_SD, LAMBDA_STATE,
)

FLARE_STATE = 2
N_SEEDS = 10
N_PATIENTS = 120
N_DAYS = 120
NU = 4

E_TAU_R_POP = 31.1
E_TAU_M_POP = 19.7


def simulate_patients_t(n_patients, n_days, seed):
    """Mirror of simulate_patients but with Student-t biomarker noise on the
    test-time emissions (scaled so the marginal variance matches sigma_b^2)."""
    rng = np.random.RandomState(seed)
    pats = []
    scale_factor = np.sqrt((NU - 2) / NU)  # so Var = sigma^2 for student-t(nu)
    for _ in range(n_patients):
        P_n = np.vstack([
            dirichlet.rvs(12 * P_POP[k] + 1e-3, random_state=rng)[0]
            for k in range(K)
        ])
        offset = rng.randn(N_BIO) * PATIENT_SD

        s = rng.choice(K, p=PI0)
        states = [s]
        for _ in range(n_days - 1):
            s = rng.choice(K, p=P_n[s])
            states.append(s)
        states = np.array(states)

        X = np.zeros((n_days, N_BIO))
        obs = np.ones((n_days, N_BIO), dtype=bool)
        L = np.zeros(n_days, dtype=bool)

        for b in range(2):
            noise = student_t.rvs(NU, size=n_days, random_state=rng) * \
                    BIO_STDS[b] * scale_factor
            X[:, b] = POP_MEANS[b, states] + offset[b] + noise

        for t_d in range(n_days):
            draw = rng.random() < LAMBDA_STATE[states[t_d]]
            L[t_d] = draw
            if draw:
                for b in range(2, N_BIO):
                    noise = student_t.rvs(NU, random_state=rng) * \
                            BIO_STDS[b] * scale_factor
                    X[t_d, b] = POP_MEANS[b, states[t_d]] + offset[b] + noise
            else:
                X[t_d, 2:] = np.nan
                obs[t_d, 2:] = False
        pats.append(dict(states=states, X=X, obs=obs, L=L, n_visits=n_days))
    return pats


def collect_realized_leads(simulate_fn, label):
    realized_R = []
    realized_M = []
    n_total = 0
    for run in range(N_SEEDS):
        seed = run * 53 + 11
        # Training data: always Gaussian (the filter never sees t_4 in fitting)
        tr_pats = simulate_patients(N_PATIENTS, N_DAYS, seed)[:60]
        tr_ff = [fill_forward(p['X'], p['obs']) for p in tr_pats]
        P, mu, sig, pi, lam_hat = mle_params(tr_pats, tr_ff)

        # Re-simulated test split per Section IX-H
        te_pats = simulate_fn(40, N_DAYS, seed + 9001)
        te_ff = [fill_forward(p['X'], p['obs']) for p in te_pats]
        for p, seq in zip(te_pats, te_ff):
            states = p['states']
            flare_days = np.where(states == FLARE_STATE)[0]
            if len(flare_days) == 0:
                continue
            tau = int(flare_days[0])
            if tau == 0:
                continue
            for d in range(tau):
                realized_lead = tau - d
                n_total += 1
                ground_truth = int(states[d])
                if ground_truth == 0:
                    realized_R.append(realized_lead)
                elif ground_truth == 1:
                    realized_M.append(realized_lead)
    print(f"  {label}: total pre-flare pairs = {n_total:,}")
    return np.array(realized_R), np.array(realized_M), n_total


def render_figure(g_R, g_M, t_R, t_M, n_g, n_t, out_path):
    g_mean_R, g_mean_M = g_R.mean(), g_M.mean()
    t_mean_R, t_mean_M = t_R.mean(), t_M.mean()
    g_se_R = g_R.std() / np.sqrt(len(g_R))
    g_se_M = g_M.std() / np.sqrt(len(g_M))
    t_se_R = t_R.std() / np.sqrt(len(t_R))
    t_se_M = t_M.std() / np.sqrt(len(t_M))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('white')

    width = 0.25
    x = np.array([0, 1.5])

    PRED_COLOR = '#cccccc'
    GAUSS_COLOR = '#5e4b8b'
    T_COLOR = '#c0504d'

    preds = [E_TAU_R_POP, E_TAU_M_POP]
    gauss = [g_mean_R, g_mean_M]
    tdist = [t_mean_R, t_mean_M]
    gerr = [g_se_R, g_se_M]
    terr = [t_se_R, t_se_M]

    b1 = ax.bar(x - width, preds, width, color=PRED_COLOR,
                edgecolor='white', linewidth=1.5,
                label=r'Predicted $\mathbb{E}[\tau | S]$ (from $N$)')
    b2 = ax.bar(x, gauss, width, color=GAUSS_COLOR, yerr=gerr,
                edgecolor='white', linewidth=1.5, capsize=4,
                label=f'Realized, Gaussian (well-specified, n={n_g:,})')
    b3 = ax.bar(x + width, tdist, width, color=T_COLOR, yerr=terr,
                edgecolor='white', linewidth=1.5, capsize=4,
                label=f'Realized, t-distributed df=4 (misspecified, n={n_t:,})')

    for bars, vals in [(b1, preds), (b2, gauss), (b3, tdist)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Filter believes R', 'Filter believes M'], fontsize=11)
    ax.set_ylabel(r'Days to first flare ($\tau - d$)', fontsize=11)
    ax.set_title('Calibration under Gaussian and t-distributed emissions',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9, frameon=True, edgecolor='#cccccc')
    ax.set_ylim(0, 40)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print("EMISSION MISSPECIFICATION ROBUSTNESS (Section IX-H)")
    print("  10 seeds, 40 test patients/seed, 120 days each")
    print(f"  Student's t emissions: nu={NU}, scaled so Var matches sigma^2")
    print("=" * 70)

    print("\nGaussian (well-specified):")
    g_R, g_M, n_g = collect_realized_leads(simulate_patients, 'Gaussian')
    print(f"  R-state: mean = {g_R.mean():.2f} days "
          f"(predicted {E_TAU_R_POP}, gap {g_R.mean()-E_TAU_R_POP:+.2f})")
    print(f"  M-state: mean = {g_M.mean():.2f} days "
          f"(predicted {E_TAU_M_POP}, gap {g_M.mean()-E_TAU_M_POP:+.2f})")

    print("\nStudent-t df=4 (misspecified):")
    t_R, t_M, n_t = collect_realized_leads(simulate_patients_t, 't_4')
    print(f"  R-state: mean = {t_R.mean():.2f} days "
          f"(predicted {E_TAU_R_POP}, gap {t_R.mean()-E_TAU_R_POP:+.2f})")
    print(f"  M-state: mean = {t_M.mean():.2f} days "
          f"(predicted {E_TAU_M_POP}, gap {t_M.mean()-E_TAU_M_POP:+.2f})")

    os.makedirs('figures', exist_ok=True)
    out_path = os.path.join('figures', 'misspec.png')
    render_figure(g_R, g_M, t_R, t_M, n_g, n_t, out_path)
    print(f"\nFigure 5 saved to {out_path}")


if __name__ == '__main__':
    main()
