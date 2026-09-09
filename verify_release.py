#!/usr/bin/env python3
"""Independent release gate for the Crohn's HMM analysis package.

Mode-aware. Two modes:
  --release-mode candidate   Placeholder DOI expected; internal build.
  --release-mode submission  Real Zenodo DOI required; consistent across
                             release-facing files; optionally verified online.
Preserves all pre-existing scientific checks from the v2.2.0 verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import crohns_hmm_pipeline as crp

EXPECTED_MODELS = set(crp.MODEL_ORDER)
EXPECTED_CONFIG = {
    "n_seeds": 10, "n_total": 120, "n_train": 65, "n_val": 15,
    "n_days": 120, "future_days": 240, "pmf_horizon": 180,
    "hazard_horizon": 120, "n_param_draws": 16, "n_hazard_draws": 16,
    "n_starts": 3, "max_em_iter": 35, "performance_bootstrap": 2000,
    "calibration_bootstrap": 500, "base_seed": 1729,
}
DOI_RE = re.compile(r"^10\.5281/zenodo\.[0-9]+$")
PLACEHOLDER = "PENDING_AUTHOR_DEPOSIT"
DOI_FILES = (
    "release_config.json",
    "release_metadata.json",
    "README.md",
    "CITATION.cff",
    "manuscript_template.tex",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_landmarks(out: Path) -> pd.DataFrame:
    p = out / "landmark_predictions.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(out / "landmark_predictions.csv.gz")


def compare_csv(a: Path, b: Path, sort_cols: list[str]) -> None:
    x = pd.read_csv(a).sort_values(sort_cols).reset_index(drop=True)
    y = pd.read_csv(b).sort_values(sort_cols).reset_index(drop=True)
    assert_frame_equal(x, y, check_exact=False, rtol=1e-12, atol=1e-12)


def verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    assert manifest.exists()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = root / rel
        assert path.exists(), f"manifest path missing: {rel}"
        assert sha256(path) == digest, f"manifest digest mismatch: {rel}"


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def doi_resolves(doi: str, timeout: int = 20) -> None:
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(url, headers={"User-Agent": "crohns-hmm-release-verifier/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not (200 <= resp.status < 400):
                raise AssertionError(f"DOI {doi} returned HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        raise AssertionError(f"DOI {doi} returned HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise AssertionError(f"DOI {doi} could not be resolved: {e.reason}")


def verify_artifacts_csv(root: Path) -> None:
    """Cross-check ARTIFACTS.csv claims against actual state.

    - storage=git: file must exist in the repo tree
    - storage=archive_only: path (with seed_XX and brace-expansion resolved)
      must appear in MANIFEST.sha256
    - storage=generated: file must exist (checked after make paper has run)
    - storage=excluded: file must NOT exist in the tracked tree
    """
    import csv
    import fnmatch
    import re as _re

    artifacts_csv = root / "ARTIFACTS.csv"
    assert artifacts_csv.exists(), "ARTIFACTS.csv is missing"

    valid_storage = {"git", "generated", "archive_only", "excluded"}
    rows = list(csv.DictReader(artifacts_csv.open()))
    assert rows, "ARTIFACTS.csv has no data rows"

    # Load manifest for archive_only cross-check
    manifest_path = root / "MANIFEST.sha256"
    manifest_paths = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            _, rel = line.split("  ", 1)
            manifest_paths.add(rel.strip())

    problems = []

    for row in rows:
        aid = row["artifact_id"]
        rel_path = row["path"]
        storage = row["storage"]

        assert storage in valid_storage, (
            f"ARTIFACTS.csv row {aid!r}: invalid storage {storage!r}"
        )

        # Expand brace patterns like {a,b,c} into a list of concrete paths
        def expand_braces(pattern: str) -> list[str]:
            m = _re.search(r"\{([^{}]+)\}", pattern)
            if not m:
                return [pattern]
            options = m.group(1).split(",")
            prefix = pattern[:m.start()]
            suffix = pattern[m.end():]
            results = []
            for opt in options:
                results.extend(expand_braces(prefix + opt + suffix))
            return results

        # Resolve seed_XX placeholder into all 10 concrete seeds
        def expand_seeds(pattern: str) -> list[str]:
            if "seed_XX" not in pattern:
                return [pattern]
            return [pattern.replace("seed_XX", f"seed_{i:02d}") for i in range(10)]

        # Fully expand: braces first, then seeds, then apply glob if any
        concrete_patterns = []
        for after_braces in expand_braces(rel_path):
            concrete_patterns.extend(expand_seeds(after_braces))

        if storage == "git":
            # Check files exist. Glob patterns need at least one match.
            for pattern in concrete_patterns:
                if "*" in pattern:
                    matches = list(root.glob(pattern))
                    if not matches:
                        problems.append(f"{aid}: git storage but no files match {pattern!r}")
                else:
                    if not (root / pattern).exists():
                        problems.append(f"{aid}: git storage but file missing: {pattern}")
        elif storage == "generated":
            # Same as git after build has run
            for pattern in concrete_patterns:
                if "*" in pattern:
                    matches = list(root.glob(pattern))
                    if not matches:
                        problems.append(f"{aid}: generated storage but no files match {pattern!r}")
                else:
                    if not (root / pattern).exists():
                        problems.append(f"{aid}: generated storage but file missing after build: {pattern}")
        elif storage == "archive_only":
            # Every concrete pattern must appear in MANIFEST.sha256 (glob-match allowed)
            if not manifest_paths:
                problems.append(f"{aid}: archive_only but MANIFEST.sha256 is missing or empty")
                continue
            for pattern in concrete_patterns:
                if "*" in pattern:
                    matched = any(fnmatch.fnmatch(mp, pattern) for mp in manifest_paths)
                    if not matched:
                        problems.append(f"{aid}: archive_only pattern {pattern!r} matches nothing in MANIFEST.sha256")
                else:
                    if pattern not in manifest_paths:
                        problems.append(f"{aid}: archive_only path {pattern!r} not in MANIFEST.sha256")
        elif storage == "excluded":
            # File must NOT be tracked/present
            for pattern in concrete_patterns:
                if "*" in pattern:
                    matches = list(root.glob(pattern))
                    if matches:
                        problems.append(f"{aid}: excluded but files present: {[str(m) for m in matches[:3]]}")
                else:
                    if (root / pattern).exists():
                        problems.append(f"{aid}: excluded but file present: {pattern}")

    assert not problems, (
        "ARTIFACTS.csv inconsistencies:\n  " + "\n  ".join(problems)
    )


def verify_release_identity(root: Path, release_mode: str, check_online: bool) -> None:
    """Mode-aware release identity, provenance, and DOI consistency checks."""
    config_path = root / "release_config.json"
    meta_path = root / "release_metadata.json"
    config = json.loads(config_path.read_text())
    meta = json.loads(meta_path.read_text())

    # Config mode matches CLI mode (defensive).
    config_mode = config.get("release_mode")
    assert config_mode == release_mode, (
        f"release_mode mismatch: CLI said {release_mode!r}, "
        f"release_config.json says {config_mode!r}"
    )

    # Version and expected tag string.
    version = config["software_version"]
    expected_tag = f"time-to-flare-study-v{version}"

    # Git provenance: clean worktree, exact tag, tag resolves to HEAD.
    status = run_git(root, "status", "--porcelain", "--untracked-files=all")
    assert status == "", f"worktree is not clean; uncommitted changes:\n{status}"
    head_commit = run_git(root, "rev-parse", "HEAD")
    try:
        exact_tag = run_git(root, "describe", "--tags", "--exact-match")
    except subprocess.CalledProcessError:
        raise AssertionError(f"HEAD is not exactly at a tag; expected {expected_tag}")
    assert exact_tag == expected_tag, f"tag mismatch: expected {expected_tag}, found {exact_tag}"
    tag_commit = run_git(root, "rev-list", "-n", "1", expected_tag)
    assert tag_commit == head_commit, (
        f"tag {expected_tag} points at {tag_commit}, HEAD is {head_commit}"
    )

    # Metadata agreement with git and config.
    assert re.fullmatch(r"[0-9a-f]{40}", meta["source_commit"]), "source_commit not a full SHA"
    assert meta["source_commit"] == head_commit, (
        f"metadata source_commit {meta['source_commit']} != HEAD {head_commit}"
    )
    assert meta.get("release_tag") == expected_tag, (
        f"metadata release_tag {meta.get('release_tag')!r} != {expected_tag!r}"
    )
    assert meta.get("software_version") == version, "metadata software_version disagrees with config"

    # DOI presence and mode-specific validation.
    config_doi = config["public_archive_doi"]
    meta_doi = meta["public_archive_doi"]
    assert config_doi == meta_doi, (
        f"DOI mismatch between release_config.json ({config_doi!r}) "
        f"and release_metadata.json ({meta_doi!r})"
    )

    if release_mode == "candidate":
        # Accept either the placeholder (no DOI reserved yet) or a valid
        # Zenodo DOI (reserved but archive not yet published). Do not check
        # online resolution here; a reserved-but-unpublished DOI returns 404
        # until the deposit is published, and that is expected in candidate mode.
        assert config_doi == PLACEHOLDER or DOI_RE.fullmatch(config_doi), (
            f"candidate mode requires DOI == {PLACEHOLDER!r} or a valid Zenodo DOI, "
            f"found {config_doi!r}"
        )
        # The manuscript TEMPLATE (not the generated .tex) must never hardcode
        # a DOI; it must use the @@PUBLIC_ARCHIVE_DOI@@ token so submission-mode
        # rebuilds propagate the DOI cleanly.
        template_text = (root / "manuscript_template.tex").read_text()
        assert not re.search(r"10\.5281/zenodo\.[0-9]+", template_text), (
            "candidate mode: manuscript_template.tex still contains a hardcoded "
            "Zenodo DOI; use the token @@PUBLIC_ARCHIVE_DOI@@ instead"
        )
        return

    # submission mode from here on.
    assert config_doi != PLACEHOLDER, "submission mode: DOI is still the placeholder"
    assert DOI_RE.fullmatch(config_doi), f"submission mode: DOI {config_doi!r} does not match Zenodo pattern"

    # DOI byte-identical across all release-facing files.
    seen = {}
    for rel in DOI_FILES:
        p = root / rel
        assert p.exists(), f"submission mode: missing {rel}"
        text = p.read_text()
        assert PLACEHOLDER not in text, f"submission mode: placeholder remains in {rel}"
        assert config_doi in text, f"submission mode: DOI {config_doi!r} not found in {rel}"
        # Also confirm no OTHER Zenodo DOI appears.
        others = {m for m in DOI_RE.findall(text) if m != config_doi}
        # DOI_RE.findall requires full-string match; use finditer with a non-anchored pattern:
        others = set(re.findall(r"10\.5281/zenodo\.[0-9]+", text)) - {config_doi}
        assert not others, f"submission mode: {rel} contains stray DOI(s): {sorted(others)}"
        seen[rel] = text.count(config_doi)
    # Optional online resolution.
    if check_online:
        doi_resolves(config_doi)


def verify_no_stale_version_strings(root: Path, current_version: str) -> None:
    """Fail if any active release file mentions an earlier version tag."""
    # Only scan files we actually manage; skip vendor/, build/, .git/, etc.
    candidates = [
        "README.md", "CITATION.cff",
        "release_config.json", "release_metadata.json",
        "manuscript_template.tex", "Crohns_HMM_Time_to_Flare_Study.tex",
        "IMPLEMENTATION_AND_VERIFICATION_SUMMARY.md",
        "REPRODUCIBILITY_GUIDE.md", "TEMPLATE_BUILD_NOTE.md",
    ]
    # Prior versions to reject. Extend if more releases accumulate.
    prior_tags = ["time-to-flare-study-v2.1.0", "time-to-flare-study-v2.2.0"]
    prior_tags = [t for t in prior_tags if not t.endswith(f"v{current_version}")]
    problems = []
    for rel in candidates:
        p = root / rel
        if not p.exists():
            continue
        text = p.read_text()
        for tag in prior_tags:
            if tag in text:
                problems.append(f"{rel}: contains stale tag {tag!r}")
    assert not problems, "stale version references:\n  " + "\n  ".join(problems)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--outputs", type=Path, default=Path("final_outputs"))
    ap.add_argument("--release-mode", choices=("candidate", "submission"), required=True,
                    help="candidate: placeholder DOI expected; submission: real DOI required")
    ap.add_argument("--check-doi-online", action="store_true",
                    help="in submission mode, also verify DOI resolves at doi.org")
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.outputs if args.outputs.is_absolute() else root / args.outputs

    config = json.loads((out / "config.json").read_text())
    for key, value in EXPECTED_CONFIG.items():
        assert config[key] == value, f"{key}: expected {value}, found {config[key]}"
    n_test = config["n_total"] - config["n_train"] - config["n_val"]

    landmarks = load_landmarks(out)
    patients = pd.read_csv(out / "patient_metrics.csv")
    wide = pd.read_csv(out / "tables" / "main_results_wide.csv")
    paired = pd.read_csv(out / "tables" / "paired_differences.csv")
    assert set(landmarks.model.unique()) == EXPECTED_MODELS
    assert set(patients.model.unique()) == EXPECTED_MODELS
    assert set(wide.model.unique()) == EXPECTED_MODELS
    assert set(paired.comparator.unique()) == EXPECTED_MODELS - {crp.MODEL_DRAW}
    assert len(landmarks) == config["n_seeds"] * n_test * config["n_days"] * len(EXPECTED_MODELS)
    assert len(patients) == config["n_seeds"] * n_test * len(EXPECTED_MODELS)
    assert landmarks.event_full.notna().all()
    assert landmarks.cover90.isin([0.0, 1.0]).all()
    assert np.isfinite(patients[["nll", "ibs4", "entropy", "cover90"]].to_numpy()).all()

    expected_shape = (len(crp.MODEL_ORDER), n_test, config["n_days"], config["pmf_horizon"] + 2)
    for seed_index in range(config["n_seeds"]):
        path = out / "prediction_distributions" / f"seed_{seed_index:02d}.npz"
        assert path.exists()
        with np.load(path, allow_pickle=False) as z:
            assert tuple(z["pmf"].shape) == expected_shape
            assert tuple(z["state_posterior"].shape) == expected_shape[:-1] + (crp.K,)
            assert tuple(z["conditional_mean"].shape) == expected_shape[:-1]
            assert tuple(z["unconditional_mean"].shape) == expected_shape[:-1]
            assert tuple(z["model_names"].tolist()) == crp.MODEL_ORDER
            np.testing.assert_allclose(z["pmf"].sum(axis=-1), 1.0, atol=3e-6)
            np.testing.assert_allclose(z["pmf"][..., 0], z["state_posterior"][..., 2], atol=3e-6)
            assert np.all(z["pmf"] >= -1e-7)
            assert np.issubdtype(z["event_full"].dtype, np.integer)

        art = out / "model_artifacts" / f"seed_{seed_index:02d}"
        hmm_specs = [
            ("hmm_draw", False), ("hmm_no_draw", False),
            ("draw_stratified_hmm", True),
        ]
        for slug, is_stratified in hmm_specs:
            base = json.loads((art / f"{slug}_base.json").read_text())
            ens = json.loads((art / f"{slug}_ensemble.json").read_text())
            assert ens["n_members"] == config["n_param_draws"] + 1
            assert len(ens["members"]) == ens["n_members"]
            assert len(base["start_diagnostics"]) == config["n_starts"]
            assert sum(bool(x["selected"]) for x in base["start_diagnostics"]) == 1
            assert len(base["em_loglik_trace"]) >= 1
            assert all(len(x["loglik_trace"]) >= 1 for x in base["start_diagnostics"])
            if is_stratified:
                assert base["pm_wear_mu"] is not None and base["pm_wear_sd"] is not None
                assert all(x["pm_wear_mu"] is not None for x in ens["members"])

        hz_base = json.loads((art / "event_history_base.json").read_text())
        hz_ens = json.loads((art / "event_history_ensemble.json").read_text())
        assert hz_ens["n_members"] == config["n_hazard_draws"] + 1
        assert len(hz_ens["members"]) == hz_ens["n_members"]
        assert "prior_panel_observed" in hz_base["feature_names"]
        assert hz_base["scaler"]["n_features_in"] == len(hz_base["feature_names"])
        assert hz_ens["members"][0]["bootstrap_patient_ids"] is None
        for member in hz_ens["members"][1:]:
            assert len(member["bootstrap_patient_ids"]) == config["n_train"]
        with (art / "event_history_base.pkl").open("rb") as fh:
            obj = pickle.load(fh)
            assert isinstance(obj, dict)
            assert obj == hz_base
        with (art / "event_history_ensemble.pkl").open("rb") as fh:
            obj = pickle.load(fh)
            assert isinstance(obj, dict)
            assert obj["n_members"] == config["n_hazard_draws"] + 1
            assert obj == hz_ens

        sim_seed = config["base_seed"] + 1009 * seed_index
        cohort = crp.simulate_patients(config["n_total"], config["n_days"], config["future_days"], sim_seed)
        expected_ref = crp.training_lab_reference(cohort[:config["n_train"]])
        np.testing.assert_allclose(np.asarray(hz_base["initial_loglab"]), expected_ref, atol=1e-12)
        assert not np.allclose(expected_ref, crp.TRUE_L_LOGMU[0])

    source = (root / "crohns_hmm_pipeline.py").read_text()
    assert "TRUE_L_MED[0]" not in source and "TRUE_L_LOGMU[0]" not in source

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        crp.aggregate_results(landmarks, patients, tmp, tuple(config["horizons"]),
                              config["performance_bootstrap"], config["calibration_bootstrap"])
        crp.aggregate_noncurrent_flare_sensitivity(
            landmarks, tmp, tuple(config["horizons"]),
            config["performance_bootstrap"], config["calibration_bootstrap"])
        compare_csv(out / "tables" / "performance_summary.csv", tmp / "tables" / "performance_summary.csv", ["model", "metric"])
        compare_csv(out / "tables" / "paired_differences.csv", tmp / "tables" / "paired_differences.csv", ["comparator", "metric"])
        compare_csv(out / "tables" / "main_results_wide.csv", tmp / "tables" / "main_results_wide.csv", ["model"])
        compare_csv(out / "tables" / "noncurrent_flare_performance.csv", tmp / "tables" / "noncurrent_flare_performance.csv", ["model", "metric"])
        compare_csv(out / "tables" / "noncurrent_flare_paired_differences.csv", tmp / "tables" / "noncurrent_flare_paired_differences.csv", ["comparator", "metric"])
        assert json.loads((out / "tables" / "calibration_30d.json").read_text()) == json.loads((tmp / "tables" / "calibration_30d.json").read_text())
        assert json.loads((out / "tables" / "noncurrent_flare_sensitivity.json").read_text()) == json.loads((tmp / "tables" / "noncurrent_flare_sensitivity.json").read_text())

        generated = tmp / "generated.tex"
        subprocess.run([str(Path(crp.sys.executable)), str(root / "fill_manuscript.py"),
                        "--root", str(root), "--outputs", str(out), "--output", str(generated)], check=True)
        assert generated.read_bytes() == (root / "Crohns_HMM_Time_to_Flare_Study.tex").read_bytes()

    tex = (root / "Crohns_HMM_Time_to_Flare_Study.tex").read_text()
    assert "@@" not in tex
    for phrase in ["day-zero mass", "Draw-stratified-emission HMM",
                   "training-cohort median", "Non-current-flare sensitivity",
                   "complete PMF arrays"]:
        assert phrase in tex, f"expected phrase missing from generated TeX: {phrase!r}"
    assert re.search(r"HMM \+ draw model\s*&\s*[0-9]", tex)

    required_figures = {"time_to_flare_interface.pdf", "calibration_30d.pdf", "score_nll.pdf",
                        "score_ibs4.pdf", "coverage90.pdf", "lambda_recovery.pdf"}
    assert required_figures.issubset({p.name for p in (root / "figures").glob("*.pdf")})
    pdf = root / "Crohns_HMM_Time_to_Flare_Study.pdf"
    assert pdf.exists() and pdf.stat().st_size > 100_000

    # Artifact inventory cross-check.
    verify_artifacts_csv(root)

    # Release identity, provenance, and DOI mode.
    verify_release_identity(root, args.release_mode, args.check_doi_online)

    # No stale version strings in release-facing files.
    verify_no_stale_version_strings(root, config["software_version"])

    # File-level integrity for anything in MANIFEST.sha256.
    verify_manifest(root)

    print(f"Release verification passed ({args.release_mode} mode).")


if __name__ == "__main__":
    main()
