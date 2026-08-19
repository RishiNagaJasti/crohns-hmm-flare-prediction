# crohns-hmm-flare-prediction

A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study.

This repository contains the code accompanying the paper *"A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study" (Jasti, 2026).*

## Reproduce the paper

The full pipeline (tests, simulation, manuscript compilation, release metadata, manifest, verification) runs from a single make target:

```bash
make reproduce
```

Individual stages are also available:

- `make test` — run the test suite
- `make simulate` — run the simulation pipeline
- `make paper` — compile the manuscript PDF
- `make verify` — check that the release satisfies the acceptance-test assertions in the paper

For a step-by-step walkthrough with acceptance tests at each stage, environment locking, and troubleshooting, see [`REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md).

### Requirements

- Python 3.13.5 (exact version pinned in `requirements.lock.txt`)
- TeX Live 2026 or newer, including `latexmk` (for `make paper`)
- See `REPRODUCIBILITY_GUIDE.md` Stage 2 for full environment setup, including required thread-count settings for numerical determinism.

A `Dockerfile` is provided as a self-contained alternative.

## Repository layout

```
crohns-hmm-flare-prediction/
├── README.md
├── LICENSE
├── Makefile                          # orchestrates the full pipeline
├── Dockerfile                        # self-contained reproducible environment
├── requirements.txt
├── requirements.lock.txt             # exact pinned versions
├── manuscript_template.tex           # LaTeX source with placeholders
├── Crohns_HMM_Time_to_Flare_Study.tex # generated manuscript (do not edit)
├── Crohns_HMM_Time_to_Flare_Study.pdf # compiled PDF
├── fig1_graphicalmodel.tex           # standalone graphical-model figure source
├── crohns_hmm_pipeline.py            # simulation and estimation pipeline
├── generate_figures_from_outputs.py  # figure generation from stored outputs
├── fill_manuscript.py                # substitutes results into the template
├── finalize_outputs.py               # post-processes simulation outputs
├── build_manifest.py                 # generates release manifest
├── build_release_metadata.py         # generates release metadata
├── verify_release.py                 # checks acceptance-test assertions
├── release_config.json               # release version and DOI
├── REPRODUCIBILITY_GUIDE.md          # authoritative reproduction guide
├── REVIEWER_RESPONSE_MATRIX.md       # reviewer-comment tracking
├── IMPLEMENTATION_AND_VERIFICATION_SUMMARY.md
├── TEMPLATE_BUILD_NOTE.md
├── figures/                          # generated figures
├── final_outputs/                    # simulation outputs used to fill the paper
├── tests/                            # test suite
├── patches/                          # patches applied to vendored dependencies
├── vendor/                           # vendored dependencies
└── build/                            # build artifacts (transient)
```

## Honest scope statement

This is methods-and-simulation work. No clinical efficacy claim is made, and no real patient data is used. The framework is presented as a candidate decision-support tool that produces a posterior distribution over time-to-flare; whether this output translates to improved patient outcomes is an empirical question that real-data prospective studies under IRB oversight would need to answer.

## License

MIT. See `LICENSE`.

## Citation

If you use this code or framework in your work, please cite the accompanying paper:

```
@misc{Jasti2026crohnshmm,
  author       = {Rishi Jasti},
  title        = {A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: A Simulation Study},
  year         = {2026},
  howpublished = {Manuscript},
  note         = {Code: \url{https://github.com/RishiNagaJasti/crohns-hmm-flare-prediction}}
}
```

If you use the archived code and reproducibility materials, please also cite:

```
@software{jasti2026code,
  author       = {Rishi Jasti},
  title        = {A Hidden Markov Framework for Time-to-Flare Prediction in Crohn's Disease Under Endogenous Laboratory Sampling: Code and Reproducibility Materials},
  version      = {2.2.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21971684}
}
```

## Contact

Rishi Jasti -- jasti27r@ncssm.edu

North Carolina School of Science and Mathematics, Durham, NC, USA
