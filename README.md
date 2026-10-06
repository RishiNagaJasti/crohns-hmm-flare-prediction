# crohns-hmm-flare-prediction

A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study.

This repository contains the code, tests, and reproducibility materials accompanying the paper *"A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study" (Jasti, 2026).*

## Release identity

- **Version:** 2.3.3
- **Release tag:** `resubmission-v2.3.3` (a git tag; no branch shares this name)
- **Public archive DOI:** 10.5281/zenodo.21971684

## Release history

- **v1.0:** the original IEEE Access submission, which was rejected with an invitation to resubmit. Its code is preserved under the tag `v1.0-ieee-access-submission` (branch `main`).
- **v2.3.3 (current):** the resubmission release (supersedes v2.3.2, v2.3.1 and v2.3.0; the analysis code and numerical results are unchanged; the manuscript text, the manuscript-filling script, the release verifier, the Makefile (prerequisite checks, single-threaded manuscript filling and messages), the Dockerfile, the documentation and the release metadata differ). The manuscript source in this repository is the source of the submitted PDF, and `make reproduce` rebuilds and verifies it end to end.

## Reproduce the paper

Check out the release tag first (`git checkout resubmission-v2.3.3`). There are two ways to reproduce the results, and a quicker check of the archived outputs:

**Analysis only (no TeX or IEEE template needed):**

    make results

This runs the unit tests and the full simulation, estimation, and evaluation, writes every results table and generated figure to `final_outputs/`, and then regenerates the reported tables from that run's archived patient- and landmark-level outputs by rerunning the aggregation code, and checks that they agree with the written tables. It needs only Python 3.13 and make. The full simulation takes roughly 15 to 45 minutes on a laptop (two worker processes). The tabulated numbers are written to `final_outputs/tables/`; every other reported value is computed by `fill_manuscript.py` from the archived outputs and the generating constants in the pipeline source.

**Checking the archived outputs without refitting any model (about one minute):** download and unzip the outputs archive from Zenodo (DOI 10.5281/zenodo.21971684), then run

    PYTHONPATH=. python3 verify_release.py --outputs /path/to/final_outputs --numbers-only --release-mode submission

which checks the archived configuration, prediction archives, and model objects and regenerates the performance, paired-difference, calibration, and non-current-flare tables by rerunning the aggregation code on the archived patient- and landmark-level files.

**Full release build (analysis, manuscript PDF, and release verification):**

    make reproduce IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip

This additionally needs TeX Live (including the IEEEtran class; Debian/Ubuntu package `texlive-publishers`) with `latexmk`, `pdftotext` (poppler), `unzip`, `shasum`, and the IEEE Access template (see below). `make reproduce` checks these prerequisites before the simulation starts and stops immediately if one is missing.

Individual stages are also available:

- `make test`: run the unit tests
- `make simulate`: run the simulation pipeline (writes `final_outputs/`)
- `make paper`: generate the manuscript TeX from `final_outputs/` and compile the PDF (needs the template)
- `make verify`: run the release verifier (after `make paper`; see below)
- `make quick`: a fast smoke test (unit tests plus a reduced two-seed simulation; not the manuscript result)

Numerical determinism requires single-threaded BLAS; the Makefile sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1 for the tests, the simulation, the manuscript filling, and verification.

`make verify` checks the configuration in Appendix B of the paper and the verification-script part of the acceptance tests in Appendix C (the others are unit tests run by `make test`), and additionally checks: day-zero mass, archived model objects, regeneration of the reported tables (by rerunning the aggregation code on the archived outputs) and of the manuscript TeX, the presence of the regenerated primary log and Brier scores in the PDF, the artifact inventory, the release tag, and the checksum manifest. It needs the outputs of `make simulate`, `make paper`, `make metadata`, and `make manifest`, and a clean worktree with HEAD at the release tag. `make results` runs only the numerical part of these checks.

### Requirements

- Python 3.13 (the reported results were produced with Python 3.13.5; package versions are pinned in `requirements.lock.txt`)
- For the PDF and full verification only: TeX Live (including the IEEEtran class) with `latexmk`, `pdftotext` (poppler), `unzip`, `shasum`, `git` (the verifier requires a clean checkout of the release tag), and the IEEE Access LaTeX template (see below; not redistributed here)

A `Dockerfile` provides the locked Python environment. Because the IEEE template is not redistributed, building the PDF inside the container requires mounting the template archive and setting `IEEE_ARCHIVE`.

### Getting the IEEE Access template

This repository does not redistribute IEEE's LaTeX template files. Before running `make paper`, download the template yourself:

1. Go to https://template-selector.ieee.org/
2. Search for "IEEE Access", pick "Research Article", download the LaTeX template ZIP. IEEE Access also links it directly from https://ieeeaccess.ieee.org/authors/preparing-your-article/; ACCESS_latex_template_20260513-1-1.zip is the version this release was built against.
3. Place the archive at `$HOME/Downloads/IEEE LaTeX Template.zip`, or override with:

       make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip

The build extracts the files it needs (`.cls`, `.sty`, template PNGs, and the fonts IEEE ships with the template) into `build/ieeeaccess/`, verifies the class, style, and PNG files against SHA-256 checksums in `vendor/ieeeaccess/UPSTREAM_SHA256.txt` (recorded from the IEEE Access LaTeX template dated 2026-05-13; a newer IEEE template may fail this check if IEEE has changed any of the checked files, see `vendor/ieeeaccess/UPSTREAM.md`), and compiles with the unmodified IEEE class. The template's font files are used only inside the uncommitted build directory and are never redistributed.

## Repository layout

    crohns-hmm-flare-prediction/
    ├── README.md                            # this file
    ├── LICENSE                              # MIT
    ├── CITATION.cff                         # machine-readable citation metadata
    ├── ARTIFACTS.csv                        # inventory of the principal artifacts produced or claimed by this repo
    ├── Makefile                             # orchestrates the full pipeline
    ├── Dockerfile                           # locked Python environment plus Debian TeX Live (IEEE template mounted at run time)
    ├── requirements.txt
    ├── requirements.lock.txt                # exact pinned versions
    ├── manuscript_template.tex              # LaTeX source with @@TOKEN@@ placeholders
    ├── Crohns_HMM_Time_to_Flare_Study.tex   # generated by make paper (not committed)
    ├── Crohns_HMM_Time_to_Flare_Study_v2.3.pdf   # compiled by make paper (not committed)
    ├── fig1_graphicalmodel.tex              # standalone graphical-model figure source
    ├── crohns_hmm_pipeline.py               # simulation and estimation pipeline
    ├── generate_figures_from_outputs.py     # figure generation from stored outputs
    ├── fill_manuscript.py                   # substitutes results into the template
    ├── finalize_outputs.py                  # post-processes simulation outputs
    ├── build_manifest.py                    # generates release manifest
    ├── build_release_metadata.py            # generates release metadata
    ├── verify_release.py                    # mode-aware release verifier
    ├── release_config.json                  # release version, DOI, and mode (single source of truth)
    ├── IMPLEMENTATION_AND_VERIFICATION_SUMMARY.md  # what changed for the resubmission and how it is verified
    ├── TEMPLATE_BUILD_NOTE.md               # IEEE Access template build rationale
    ├── .github/workflows/smoke.yml          # CI: unit tests + quick simulation + config validation
    ├── figures/                             # Figure 1 (committed); other figures are copied here by make paper
    ├── final_outputs/                       # written by make simulate (not committed; archived on Zenodo)
    ├── tests/                               # test suite
    └── vendor/ieeeaccess/                   # IEEE Access template metadata (checksums + documentation; no redistributed files)

## Honest scope statement

This is methods-and-simulation work. No clinical efficacy claim is made, and no real patient data is used. The framework produces a predictive distribution over time to flare; whether this output translates to improved patient outcomes is an empirical question that real-data prospective studies under IRB oversight would need to answer.

## Scientific scope

This remains a controlled simulation study. The state emissions are well separated, the principal HMM is correctly specified, the empirical-Bayes mixture conditions on final EM responsibilities, and the real-data IBDMDB illustration of the original submission has been removed rather than presented as clinical validation. The paper's claims are intentionally narrower than those in the rejected version.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use this code or framework in your work, please cite the accompanying paper:

    @misc{Jasti2026crohnshmm,
      author       = {Rishi Jasti},
      title        = {A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study},
      year         = {2026},
      howpublished = {Manuscript},
      note         = {Code: \url{https://github.com/RishiNagaJasti/crohns-hmm-flare-prediction}}
    }

If you use the archived code and reproducibility materials, please also cite:

    @software{jasti2026code,
      author       = {Rishi Jasti},
      title        = {A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: Code and Reproducibility Materials},
      version      = {2.3.3},
      year         = {2026},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.21971684}
    }

## Contact

Rishi Jasti — jasti27r@ncssm.edu

North Carolina School of Science and Mathematics, Durham, NC, USA
