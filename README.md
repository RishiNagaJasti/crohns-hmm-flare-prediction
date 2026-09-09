# crohns-hmm-flare-prediction

A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study.

This repository contains the code, tests, and reproducibility materials accompanying the paper *"A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study" (Jasti, 2026).*

## Release identity

- **Version:** 2.3.0
- **Release tag:** `time-to-flare-study-v2.3.0`
- **Public archive DOI:** [10.5281/zenodo.21971684](https://doi.org/10.5281/zenodo.21971684)
- **Release mode:** candidate (as of this working copy; final `submission` mode is set when the archive is published)

## Release history

- **v1.0 (rejected):** the initial IEEE Access submission that received major revision requests. Preserved for audit history under the tag `v1.0-ieee-access-submission`.
- **v2.1.0 (superseded):** intermediate development snapshot.
- **v2.2.0 (superseded):** first resubmission attempt with corrections; superseded by v2.3.0 to address remaining release-integrity gaps identified during external review.
- **v2.3.0 (current):** current release with corrected mode-aware release verifier, tokenized DOI, IEEE Access template migration, and complete artifact inventory.

## Reproduce the paper

The full pipeline (tests, simulation, manuscript compilation, release metadata, manifest, verification) runs from a single make target:

    make reproduce

Individual stages are also available:

- `make test` — run the test suite
- `make simulate` — run the simulation pipeline
- `make paper` — compile the manuscript PDF
- `make verify` — check that the release satisfies the acceptance-test assertions in the paper
- `make quick` — run a fast subset for smoke testing (unit tests + one-seed simulation)

For a step-by-step walkthrough with acceptance tests at each stage, environment locking, and troubleshooting, see [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md).

### Requirements

- Python 3.13.5 (exact version pinned in `requirements.lock.txt`)
- TeX Live 2026 or newer, including `latexmk` (for `make paper`)
- IEEE Access LaTeX template (see below; not redistributed here)
- See `REPRODUCIBILITY_GUIDE.md` for full environment setup, including required thread-count settings for numerical determinism.

A `Dockerfile` is provided as a self-contained alternative.

### Getting the IEEE Access template

This repository does not redistribute IEEE's LaTeX template files. Before running `make paper`, download the template yourself:

1. Go to https://template-selector.ieee.org/
2. Search for "IEEE Access", pick "Research Article", download the LaTeX template ZIP
3. Place the archive at `$HOME/Downloads/IEEE LaTeX Template.zip`, or override with:

       make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip

The build extracts only the files it needs (`.cls`, `.sty`, template PNGs), verifies each against SHA-256 checksums in `vendor/ieeeaccess/UPSTREAM_SHA256.txt`, and applies a font-fallback patch that substitutes system Times Roman for the proprietary Formata and GiovanniStd fonts. Proprietary font binaries are never extracted or redistributed. See `vendor/ieeeaccess/UPSTREAM.md` for the full rationale.

## Repository layout

    crohns-hmm-flare-prediction/
    ├── README.md                            # this file
    ├── LICENSE                              # MIT
    ├── CITATION.cff                         # machine-readable citation metadata
    ├── ARTIFACTS.csv                        # inventory of every artifact produced or claimed by this repo
    ├── Makefile                             # orchestrates the full pipeline
    ├── Dockerfile                           # self-contained reproducible environment
    ├── requirements.txt
    ├── requirements.lock.txt                # exact pinned versions
    ├── manuscript_template.tex              # LaTeX source with @@TOKEN@@ placeholders
    ├── Crohns_HMM_Time_to_Flare_Study.tex   # generated manuscript (do not edit)
    ├── Crohns_HMM_Time_to_Flare_Study.pdf   # compiled PDF
    ├── fig1_graphicalmodel.tex              # standalone graphical-model figure source
    ├── crohns_hmm_pipeline.py               # simulation and estimation pipeline
    ├── generate_figures_from_outputs.py     # figure generation from stored outputs
    ├── fill_manuscript.py                   # substitutes results into the template
    ├── finalize_outputs.py                  # post-processes simulation outputs
    ├── build_manifest.py                    # generates release manifest
    ├── build_release_metadata.py            # generates release metadata
    ├── verify_release.py                    # mode-aware release verifier
    ├── release_config.json                  # release version, DOI, and mode (single source of truth)
    ├── REPRODUCIBILITY_GUIDE.md             # authoritative reproduction guide
    ├── REVIEWER_RESPONSE_MATRIX.md          # reviewer-comment tracking
    ├── IMPLEMENTATION_AND_VERIFICATION_SUMMARY.md
    ├── TEMPLATE_BUILD_NOTE.md               # IEEE Access template build rationale
    ├── .github/workflows/smoke.yml          # CI: unit tests + quick simulation + config validation
    ├── figures/                             # generated figures
    ├── final_outputs/                       # simulation outputs used to fill the paper (archived on Zenodo)
    ├── tests/                               # test suite
    ├── patches/                             # patches applied to vendored dependencies at build time
    └── vendor/ieeeaccess/                   # IEEE Access template metadata (checksums + documentation; no redistributed files)

## Honest scope statement

This is methods-and-simulation work. No clinical efficacy claim is made, and no real patient data is used. The framework is presented as a candidate decision-support tool that produces a posterior distribution over time-to-flare; whether this output translates to improved patient outcomes is an empirical question that real-data prospective studies under IRB oversight would need to answer.

## Scientific scope

This remains a controlled simulation study. The state emissions are well separated, the principal HMM is correctly specified, the empirical-Bayes mixture conditions on final EM responsibilities, and the real-data IBDMDB illustration has been deferred rather than presented as clinical validation. The paper's claims are intentionally narrower than those in the rejected version.

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
      version      = {2.3.0},
      year         = {2026},
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.21971684}
    }

## Contact

Rishi Jasti — jasti27r@ncssm.edu

North Carolina School of Science and Mathematics, Durham, NC, USA
