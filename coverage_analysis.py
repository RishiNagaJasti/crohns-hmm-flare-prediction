"""
coverage_analysis.py
====================
Reproduces the full-distribution calibration of Section IX-G and renders
Figure 4 (example predictive PMFs plus the coverage diagnostic).

For each pre-flare day-patient pair, the hitting-time PMF
P(tau - d = n | F_d) = pi_tilde^T Q^{n-1} R from (8) is evaluated against the
realized lead at 19 nominal credible-interval levels. The fraction of pairs
falling inside each equal-tailed interval is the empirical coverage.

Reproducibility note
--------------------
Empirical coverage at the 50% and 90% nominal levels and the mean absolute
deviation from the 45-degree line are reproduced to within roughly one
percentage point of the published values across re-runs at this scale, due to
small numerical variation in the simulator's Dirichlet-perturbed transition
matrices and floating-point ordering across NumPy/SciPy versions. The
qualitative conclusion (empirical coverage tracking the 45-degree ideal at
all nominal levels tested) reproduces robustly.

Run:
    python coverage_analysis.py

Output:
    figures/coverage.png
    Coverage at 50%, 90%, and MAD across 19 nominal levels printed to stdout.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')

from crohns_hmm import (
    simulate_patients, fill_forward, mle_params, forward_filter,
    K, P_POP,
)

FLARE_STATE = 2
N_SEEDS = 10
N_PATIENTS = 120
N_DAYS = 120
NOMINAL_LEVELS = np.arange(0.05, 1.0, 0.05)


def predictive_pmf(alpha, P, max_n=120):
    """Hitting-time PMF P(tau - d = n | F_d) from (8) of the paper."""
    Q = P[:2, :2]
    R_col = P[:2, 2]
    pi_tilde = alpha[:2] / (alpha[:2].sum() + 1e-300)
    pmf = np.zeros(max_n)
    Qn = np.eye(2)
    for n in range(max_n):
        pmf[n] = float(pi_tilde @ Qn @ R_col)
        Qn = Qn @ Q
    return pmf / (pmf.sum() + 1e-300)


def equal_tailed_interval(pmf, level):
    cdf = np.cumsum(pmf)
    lo = int(np.searchsorted(cdf, (1 - level) / 2)) + 1
    hi = int(np.searchsorted(cdf, 1 - (1 - level) / 2)) + 1
    return lo, hi


def run_coverage():
    """Aggregate (alpha, P, realized_lead) across the canonical test split."""
    pairs = []  # list of (alpha[d], P, realized_lead)
    for run in range(N_SEEDS):
        seed = run * 53 + 11
        pats = simulate_patients(N_PATIENTS, N_DAYS, seed)
        tr = pats[:60]
        te = pats[80:]
        tr_ff = [fill_forward(p['X'], p['obs']) for p in tr]
        te_ff = [fill_forward(p['X'], p['obs']) for p in te]
        P, mu, sig, pi, lam_hat = mle_params(tr, tr_ff)

        for p, seq in zip(te, te_ff):
            states = p['states']
            flare_days = np.where(states == FLARE_STATE)[0]
            if len(flare_days) == 0:
                continue
            tau = int(flare_days[0])
            if tau == 0:
                continue
            alpha, _ = forward_filter(seq, P, mu, sig, pi,
                                       L=p['L'], lam=lam_hat)
            for d in range(tau):
                pairs.append((alpha[d].copy(), P.copy(), tau - d))
    return pairs


def compute_coverage(pairs):
    hits = np.zeros(len(NOMINAL_LEVELS), dtype=int)
    n_pairs = len(pairs)
    for alpha, P, realized in pairs:
        pmf = predictive_pmf(alpha, P, max_n=120)
        for li, lev in enumerate(NOMINAL_LEVELS):
            lo, hi = equal_tailed_interval(pmf, lev)
            if lo <= realized <= hi:
                hits[li] += 1
    return hits / n_pairs


def render_figure(pairs, emp_coverage, out_path):
    n_pairs = len(pairs)
    mad = float(np.mean(np.abs(emp_coverage - NOMINAL_LEVELS)))

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

    # Panel (a): three example PMFs at distinct filter beliefs
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_facecolor('white')
    Q = P_POP[:2, :2]
    R_col = P_POP[:2, 2]
    COLORS = {'R': '#1f77b4', 'mixed': '#9467bd', 'M': '#d62728'}
    LABELS = {
        'R':     r'$R$-dominated: $\tilde\pi=(0.95, 0.05)$',
        'mixed': r'Mixed: $\tilde\pi=(0.50, 0.50)$',
        'M':     r'$M$-dominated: $\tilde\pi=(0.05, 0.95)$',
    }
    EXAMPLE_REAL = {'R': 27, 'mixed': 49, 'M': 11}
    for key, pi_t in [('R',     np.array([0.95, 0.05])),
                       ('mixed', np.array([0.50, 0.50])),
                       ('M',     np.array([0.05, 0.95]))]:
        pmf = np.zeros(80)
        Qn = np.eye(2)
        for n in range(80):
            pmf[n] = float(pi_t @ Qn @ R_col)
            Qn = Qn @ Q
        pmf /= pmf.sum()
        ax_a.plot(np.arange(1, 81), pmf, color=COLORS[key], lw=1.6,
                  label=LABELS[key])
        ax_a.axvline(EXAMPLE_REAL[key], color=COLORS[key], ls=':', lw=1,
                     alpha=0.6)
    ax_a.set_xlabel(r'Days to first flare, $n=\tau-d$', fontsize=10)
    ax_a.set_ylabel(r'$\mathbb{P}(\tau-d=n \mid \mathcal{F}_d)$', fontsize=10)
    ax_a.set_title('(a) Predictive distributions at three filter beliefs',
                   fontsize=10)
    ax_a.legend(fontsize=8, loc='upper right', frameon=False)
    ax_a.set_xlim(0, 80)
    ax_a.grid(alpha=0.25)

    # Panel (b): empirical vs nominal coverage
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor('white')
    ax_b.plot([0, 1], [0, 1], 'k:', lw=1, label='Ideal (45-degree line)')
    ax_b.plot(NOMINAL_LEVELS, emp_coverage, 'o-', color='#2c3e50',
              lw=1.6, markersize=5, label=f'Empirical (n={n_pairs:,})')
    idx50, idx90 = 9, 17
    ax_b.scatter([NOMINAL_LEVELS[idx50]], [emp_coverage[idx50]], s=80,
                  color='#e74c3c', zorder=5)
    ax_b.annotate(f'50%: {emp_coverage[idx50]*100:.1f}%',
                   xy=(0.50, emp_coverage[idx50]),
                   xytext=(0.55, emp_coverage[idx50] - 0.08),
                   fontsize=9, color='#e74c3c')
    ax_b.scatter([NOMINAL_LEVELS[idx90]], [emp_coverage[idx90]], s=80,
                  color='#e74c3c', zorder=5)
    ax_b.annotate(f'90%: {emp_coverage[idx90]*100:.1f}%',
                   xy=(0.90, emp_coverage[idx90]),
                   xytext=(0.65, 0.97), fontsize=9, color='#e74c3c')
    ax_b.set_xlabel(r'Nominal credible-interval level $\ell$', fontsize=10)
    ax_b.set_ylabel('Empirical coverage', fontsize=10)
    ax_b.set_title('(b) Calibration diagnostic', fontsize=10)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.legend(fontsize=8, loc='upper left', frameon=False)
    ax_b.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return mad


def main():
    print("=" * 70)
    print("FULL PREDICTIVE DISTRIBUTION CALIBRATION (Section IX-G)")
    print("  10 seeds, 120 patients/seed, 60/20/40 split")
    print("  19 nominal credible-interval levels: 0.05, 0.10, ..., 0.95")
    print("=" * 70)

    pairs = run_coverage()
    emp = compute_coverage(pairs)

    print(f"\nTotal pre-flare day-patient pairs: n = {len(pairs):,}")
    print(f"\n  Nominal | Empirical | Deviation")
    print(f"  --------|-----------|----------")
    for lev, c in zip(NOMINAL_LEVELS, emp):
        print(f"   {lev:>5.2f}  |   {c:>5.3f}  |  {c - lev:+.3f}")

    os.makedirs('figures', exist_ok=True)
    out_path = os.path.join('figures', 'coverage.png')
    mad = render_figure(pairs, emp, out_path)
    print(f"\n  50% interval: {emp[9]*100:.1f}% empirical coverage")
    print(f"  90% interval: {emp[17]*100:.1f}% empirical coverage")
    print(f"  MAD from 45-degree line: {mad:.3f}")
    print(f"\nFigure 4 saved to {out_path}")


if __name__ == '__main__':
    main()
