"""
gen_figures.py
==============
Convenience orchestrator that runs all analysis scripts and produces the
seven figures of the paper. Equivalent to running each script individually.

Run:
    python gen_figures.py

Outputs (in ./figures/):
    calibration.png    Figure 3 (Section IX-F)
    coverage.png       Figure 4 (Section IX-G)
    misspec.png        Figure 5 (Section IX-H)
    ibdmdb.png         Figure 6 (Section IX-I)

Figures 1 (HMM graphical model), 2 (single-patient interface), and 7
(parameter recovery) are produced by separate utility scripts and are
checked in as static assets in figures/static/.
"""
import subprocess
import sys
import os

SCRIPTS = [
    ('calibration_analysis.py', 'Figure 3: state-conditional calibration'),
    ('coverage_analysis.py',    'Figure 4: full predictive distribution'),
    ('misspec_analysis.py',     'Figure 5: emission misspecification'),
    ('ibdmdb_analysis.py',      'Figure 6: IBDMDB empirical grounding'),
]


def main():
    os.makedirs('figures', exist_ok=True)
    for script, descr in SCRIPTS:
        print(f"\n{'=' * 70}")
        print(f"Running {script}  --  {descr}")
        print('=' * 70)
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"\n  WARNING: {script} exited with code {result.returncode}")
    print("\nAll analyses complete. Figures in ./figures/")


if __name__ == '__main__':
    main()
