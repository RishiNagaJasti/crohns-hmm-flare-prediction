#!/usr/bin/env bash
# Regenerate the font-fallback patch from the untouched upstream class and the
# working patched class. Run this once, after downloading the official template.
#
#   1. put the official ieeeaccess.cls in vendor/ieeeaccess/upstream/
#   2. put the compiling font-substituted class in working_patched/
#   3. ./patches/make_patch.sh
#   4. inspect the patch by eye, then delete working_patched/
set -euo pipefail
UP=vendor/ieeeaccess/upstream/ieeeaccess.cls
WK=working_patched/ieeeaccess.cls
OUT=patches/ieeeaccess-font-fallback.patch

[[ -f "$UP" ]] || { echo "missing $UP - download the official template first"; exit 1; }
[[ -f "$WK" ]] || { echo "missing $WK"; exit 1; }

diff -u --label a/ieeeaccess.cls --label b/ieeeaccess.cls "$UP" "$WK" > "$OUT" || true

echo "wrote $OUT"
echo
echo "=== review: the patch must contain font changes ONLY ==="
cat "$OUT"
echo
echo "=== red flags - any hit below means STOP and investigate ==="
grep -nE '^[+-].*(textwidth|columnsep|topmargin|oddsidemargin|paperwidth|\\title|\\history|\\doi|\\address|\\corresp|\\markboth|biograph|\\section|\\bibliography)' "$OUT" \
  && echo "^^ NON-FONT CHANGE DETECTED" || echo "none - font-only, as intended"
