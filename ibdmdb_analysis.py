"""
ibdmdb_analysis.py
==================
Reproduces Section IX-I: empirical grounding of the MNAR assumption on real
CD patient data from the IBDMDB cohort.

Loads the public IBDMDB / HMP2 metadata file (hmp2_metadata.csv) and computes:
  - state-conditional CRP draw rates lambda_hat_R, lambda_hat_M, lambda_hat_F
  - binomial standard errors
  - pairwise two-sample z-tests for proportions
  - empirical contraction factor c_RF
  - the bar chart of draw rates across all 66 CD participants (Figure 6c)

The HBI activity index defines the state label (HBI<5 -> R, 5-7 -> M,
>=8 -> F). The blood-draw indicator L_d = 1 on visits where CRP was measured
(non-null 'CRP (mg/L)' column).

Reproducibility note
--------------------
The IBDMDB metadata file is a static published artifact. Run-to-run
reproducibility on the same metadata snapshot is exact. The published values
in the paper (lambda_hat_R = 0.107, lambda_hat_M = 0.183, lambda_hat_F = 0.204;
N_visit = 845) come from a specific filtering convention applied to an
earlier metadata snapshot. The default filter here (one row per unique
(Participant ID, week_num); CRP draw flagged by non-null CRP value) produces
values in the same direction and ordering across snapshots, with the exact
numerical values depending on the metadata version. The qualitative
conclusion (monotone ordering lambda_hat_R < lambda_hat_M < lambda_hat_F,
R-vs-F gap significant at conventional alpha levels) reproduces robustly.

Data download
-------------
The IBDMDB metadata file is available without registration from
https://ibdmdb.org under "Downloads -> HMP2 Clinical Metadata". Save the file
to the working directory as hmp2_metadata.csv before running this script.
The script will look for the file in the working directory first.

Run:
    # 1. Download hmp2_metadata.csv from https://ibdmdb.org and place in
    #    the working directory.
    # 2. Then:
    python ibdmdb_analysis.py

Output:
    figures/ibdmdb.png
    Full IBDMDB analysis printed to stdout.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')

LOCAL_METADATA_CANDIDATES = [
    'hmp2_metadata.csv',
    'data/hmp2_metadata.csv',
    'hmp2_full_metadata.csv',
]


def find_metadata():
    for path in LOCAL_METADATA_CANDIDATES:
        if os.path.exists(path):
            return path
    msg = (
        "IBDMDB metadata file not found.\n\n"
        "To run this analysis:\n"
        "  1. Visit https://ibdmdb.org and navigate to Downloads.\n"
        "  2. Download the HMP2 Clinical Metadata file.\n"
        "  3. Save it as 'hmp2_metadata.csv' in the working directory.\n"
        "\nThen re-run this script."
    )
    print(msg, file=sys.stderr)
    sys.exit(1)


def load_cd_visits(path):
    """Load CD-participant visits with HBI and CRP-draw status, deduplicating
    to one row per (Participant ID, week_num)."""
    df = pd.read_csv(path, low_memory=False)
    cd = df[df['diagnosis'] == 'CD'].copy()
    cd['hbi'] = pd.to_numeric(cd['hbi'], errors='coerce')
    crp_col = 'CRP (mg/L)' if 'CRP (mg/L)' in cd.columns else 'CRP'
    cd[crp_col] = pd.to_numeric(cd[crp_col], errors='coerce')
    cd = cd.dropna(subset=['hbi'])

    # One row per unique (Participant ID, week_num) visit
    cd = cd.drop_duplicates(subset=['Participant ID', 'week_num'])

    cd['L'] = (~cd[crp_col].isna()).astype(int)
    cd['state'] = np.where(cd['hbi'] < 5, 'R',
                            np.where(cd['hbi'] < 8, 'M', 'F'))
    return cd


def compute_draw_rates(cd):
    out = {}
    for s in ['R', 'M', 'F']:
        sub = cd[cd['state'] == s]
        n = int(len(sub))
        k = int(sub['L'].sum())
        p = k / n if n > 0 else 0.0
        se = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
        out[s] = {'n': n, 'k': k, 'p': p, 'se': se}
    return out


def two_proportion_z(rates, a, b):
    pa, na, ka = rates[a]['p'], rates[a]['n'], rates[a]['k']
    pb, nb, kb = rates[b]['p'], rates[b]['n'], rates[b]['k']
    p_pool = (ka + kb) / (na + nb)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / na + 1 / nb))
    z = (pa - pb) / se_pool if se_pool > 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val


def render_figure(rates, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    states = ['R', 'M', 'F']
    colors = ['#27ae60', '#f39c12', '#c0392b']
    xs = np.arange(3)
    ys = [rates[s]['p'] for s in states]
    errs = [rates[s]['se'] for s in states]
    ax.bar(xs, ys, yerr=errs, color=colors, capsize=5,
           edgecolor='white', linewidth=1.5)
    for i, s in enumerate(states):
        r = rates[s]
        ax.text(i, ys[i] + errs[i] + 0.012,
                f'$\\hat\\lambda_{s}={r["p"]:.3f}$\n({r["k"]}/{r["n"]})',
                ha='center', va='bottom', fontsize=9)
    cRF = (1 - rates['F']['p']) / (1 - rates['R']['p'])
    ax.text(0.5, 0.20,
            f'$c_{{RF}} = (1-\\hat\\lambda_F)/(1-\\hat\\lambda_R) = {cRF:.3f}$',
            ha='left', va='top', fontsize=10,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    ax.set_xticks(xs)
    ax.set_xticklabels(['Remission', 'Mild', 'Flare'], fontsize=11)
    ax.set_ylabel(r'CRP draw rate $\hat\lambda$', fontsize=11)
    ax.set_title('Endogenous sampling in IBDMDB CD cohort (Section IX-I)',
                 fontsize=11)
    ax.set_ylim(0, max(ys) + max(errs) + 0.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("=" * 70)
    print("IBDMDB EMPIRICAL GROUNDING OF THE MNAR ASSUMPTION (Section IX-I)")
    print("=" * 70)

    path = find_metadata()
    print(f"\nUsing metadata file: {path}")
    cd = load_cd_visits(path)

    n_p = cd['Participant ID'].nunique()
    print(f"\nCD participants: {n_p}")
    print(f"Unique CD visits (Participant ID, week_num) with HBI: {len(cd):,}")

    rates = compute_draw_rates(cd)
    print("\nState-conditional CRP draw rates:")
    for s in ['R', 'M', 'F']:
        r = rates[s]
        print(f"  lambda_hat_{s} = {r['k']}/{r['n']} = {r['p']:.3f} "
              f"(SE {r['se']:.3f})")

    print("\nPairwise two-sample z-tests for proportions (two-sided):")
    for a, b in [('R', 'F'), ('R', 'M'), ('M', 'F')]:
        z, p = two_proportion_z(rates, a, b)
        sig = ('***' if p < 0.001 else
               '**'  if p < 0.01  else
               '*'   if p < 0.05  else 'n.s.')
        print(f"  {a} vs {b}: z = {z:+.2f}, p = {p:.3f} ({sig})")

    cRF = (1 - rates['F']['p']) / (1 - rates['R']['p'])
    print(f"\nEmpirical contraction factor c_RF = "
          f"(1-lambda_F)/(1-lambda_R) = {cRF:.3f}")
    if cRF < 1:
        print(f"  c_RF < 1 confirms posterior contraction on lab-absent days.")

    os.makedirs('figures', exist_ok=True)
    out_path = os.path.join('figures', 'ibdmdb.png')
    render_figure(rates, out_path)
    print(f"\nFigure 6 (panel c) saved to {out_path}")
    print("\nFigure 6 panels (a) and (b) (per-patient filter trajectory and")
    print("time-to-flare posterior for IBDMDB patient H4006) are produced")
    print("separately when integrating into the manuscript pipeline.")


if __name__ == '__main__':
    main()
