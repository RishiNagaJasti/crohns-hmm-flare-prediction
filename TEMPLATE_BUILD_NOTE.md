# Template build note

The IEEE template itself is not redistributed in this repository. `make ieeeaccess-assets` extracts `ieeeaccess.cls`, `spotcolor.sty`, the template PNGs, and the font files that IEEE ships with the template from an archive you download from IEEE (see `vendor/ieeeaccess/UPSTREAM.md`). It checks the class, style, and PNG files against `vendor/ieeeaccess/UPSTREAM_SHA256.txt` and writes them to `build/ieeeaccess/`, which is not committed. The class is used unmodified, so the PDF is typeset with IEEE's own fonts.

## Rebuilding

    make clean-build
    make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip

## Auditing

    unzip -p /path/to/IEEE_LaTeX_Template.zip '*/ieeeaccess.cls' > /tmp/ieeeaccess.cls
    diff -u /tmp/ieeeaccess.cls build/ieeeaccess/ieeeaccess.cls
    pdffonts Crohns_HMM_Time_to_Flare_Study_v2.3.pdf
