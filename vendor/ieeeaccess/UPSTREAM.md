# Upstream IEEE Access template

    Source:                        Official IEEE Access LaTeX template
    Download location:             <https://template-selector.ieee.org/secure/templateSelector/downloadTemplate?publicationTypeId=1&titleId=201&articleId=1&fileId=541>
    Download date:                 <FILL IN: 2026-08-14>
    Downloaded archive SHA-256:    <60c7efc9db8ac9e8bdb31c550ad4e03cb6f258a878ececc0bc690b6203e45a67>
    Upstream ieeeaccess.cls SHA-256: <67b73c4da05479d592a9449c38a81fcc7779d78753085bb44199e7f2746b757d>
    Purpose:                       Journal class and associated template assets
    Local modifications in this directory: None

The files in `upstream/` are a byte-for-byte copy of the official package used
for this submission. **Nothing in this directory is edited.**

The official class refers to the proprietary Formata and Giovanni font
resources, which IEEE does not distribute with the template and which are not
present in the locked TeX Live environment used here. A documented, font-only
compatibility substitution is therefore applied to a *generated* copy under
`build/ieeeaccess/` at build time. See `patches/ieeeaccess-font-fallback.patch`
and `TEMPLATE_BUILD_NOTE.md`.

## Redistribution

Check the redistribution terms shipped with the official package before
publishing these files. If public redistribution is not clearly permitted,
remove `upstream/` from the public repository and instead have the build accept
a user-supplied official archive whose SHA-256 is verified against
`UPSTREAM_SHA256.txt`.

Proprietary font binaries (`*.pfb`, matching `*.tfm`) must never be committed,
and must never be obtained from unofficial sources.

## Recording checksums

    sha256sum vendor/ieeeaccess/upstream/ieeeaccess.cls
    sha256sum vendor/ieeeaccess/upstream/* > vendor/ieeeaccess/UPSTREAM_SHA256.txt
