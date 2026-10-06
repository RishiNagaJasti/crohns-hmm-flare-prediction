# Template build note

The IEEE template itself is not redistributed in this repository. `make ieeeaccess-assets` extracts `ieeeaccess.cls`, `spotcolor.sty`, the template PNGs, and the font files that IEEE ships with the template from an archive you download from IEEE (see `vendor/ieeeaccess/UPSTREAM.md`). It checks the class, style, and PNG files against `vendor/ieeeaccess/UPSTREAM_SHA256.txt` and writes them to `build/ieeeaccess/`, which is not committed. The class is used unmodified, so the PDF is typeset with IEEE's own fonts.

## Rebuilding

    rm -rf build/ieeeaccess
    make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip

`make clean-build` also deletes `final_outputs/` and the committed Figure 1 PDF, so do not run it before `make verify`. `make clean-paper` also deletes the committed Figure 1 PDF. `make paper` rebuilds it, but the rebuilt file is not byte-identical, so restore the committed copy with `git checkout -- figures/fig1_graphicalmodel.pdf` before `make verify`, which requires a clean worktree.

## Auditing

    unzip -p /path/to/IEEE_LaTeX_Template.zip '*/ieeeaccess.cls' > /tmp/ieeeaccess.cls
    diff -u /tmp/ieeeaccess.cls build/ieeeaccess/ieeeaccess.cls
    pdffonts Crohns_HMM_Time_to_Flare_Study_v2.3.pdf
