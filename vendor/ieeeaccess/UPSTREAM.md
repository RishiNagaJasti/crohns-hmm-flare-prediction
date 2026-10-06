# IEEE Access template

This repository does not redistribute IEEE's LaTeX template. To build the
paper, download the template yourself and place it at the path expected by
the Makefile.

## Getting the template

1. Go to https://template-selector.ieee.org/
2. Search for "IEEE Access"
3. Choose "Research Article"
4. Download the LaTeX template as a ZIP archive
5. Place the archive at:
       $HOME/Downloads/IEEE LaTeX Template.zip

   or specify a different location via the Makefile variable:
       make paper IEEE_ARCHIVE=/path/to/IEEE_LaTeX_Template.zip

## What the build does

`make ieeeaccess-assets` extracts a subset of files from the archive into
`build/ieeeaccess/`, verifies the class, style, and PNG files' SHA-256 against
`UPSTREAM_SHA256.txt`, and compiles with the unmodified IEEE class and the
fonts IEEE distributes with the template.

Only the following files are copied into the build directory:

- `ieeeaccess.cls`
- `spotcolor.sty`
- `author1.png`, `author2.png`, `author3.png`
- `bullet.png`, `equation3.png`, `fig1.png`
- `logo.png`, `notaglinelogo.png`
- the template's font files (`*.pfb`, `*.tfm`, `*.map`, `*.fd`)

The font files are used only inside `build/ieeeaccess/`, which is not
committed; they are never redistributed.

## Reproducing exactly

The SHA-256 values in `UPSTREAM_SHA256.txt` correspond to the IEEE Access
template archive that this release was built against. IEEE may update the
template over time; if your download has different SHA-256 values, the
build will fail with a clear message. In that case, either:

- Verify the change is legitimate (a real IEEE template revision) and
  update `UPSTREAM_SHA256.txt` with the new SHA-256 values (this edits a
  tracked file, so `make verify` will then fail its clean-worktree check; use
  it only for a local PDF build), or
- Download the template version this repository was built against, which
  IEEE Access links directly from
  https://ieeeaccess.ieee.org/authors/preparing-your-article/
  (ACCESS_latex_template_20260513-1-1.zip).

## Redistribution

IEEE's template license does not clearly grant public redistribution
rights, and template PNGs (including the IEEE logo) are trademarks with
their own restrictions. This repository therefore does not include any
IEEE template files. Each user obtains the template from IEEE directly.
