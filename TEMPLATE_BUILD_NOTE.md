# Template build note

The official IEEE Access class refers to font resources that are not available
in the locked TeX Live environment used by this repository. For reproducible
local compilation, the build applies a documented, font-only compatibility
substitution using standard TeX fonts. The upstream IEEE Access files are
retained unchanged under `vendor/ieeeaccess/upstream/`. The substitution does
not alter page geometry, required IEEE Access commands, equations, figures,
tables, or scientific content. IEEE may apply its production fonts during final
typesetting.

The class actually used for compilation is generated at build time into
`build/ieeeaccess/` and is a transparent derivative, not the official
unmodified IEEE class.

## Scope of the substitution

The patch touches font selection only:

| Site | Change |
|---|---|
| `\pdfmapfile` lines | the Formata and Giovanni map files are commented out |
| `\DeclareFontShape{T1}{formata}{...}` | remapped to `ptmr8t` / `ptmri8t` / `ptmb8t` / `ptmbi8t` |
| `\DeclareFontShape{T1}{giovannistd}{n}{it}` | remapped to `ptmri8t` |
| `\font\symbfont` | `t1-formata-regsymb` replaced with `ptmr8t` |

It does not touch margins, columns, page geometry, Access title/history/DOI/
correspondence/biography commands, colours beyond the font fallback, floats,
equations, bibliography formatting, section spacing, or any scientific content.

## Rebuilding

    make clean-build
    make paper

## Auditing

    sha256sum vendor/ieeeaccess/upstream/ieeeaccess.cls
    git diff --no-index vendor/ieeeaccess/upstream/ieeeaccess.cls build/ieeeaccess/ieeeaccess.cls || true
    pdffonts Crohns_HMM_Time_to_Flare_Study.pdf
