#!/usr/bin/env python3
"""Independent release gate for the Crohn's HMM analysis package."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import subprocess
import tempfile
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--outputs", type=Path, default=Path("final_outputs"))
    args = ap.parse_args(); root = args.root.resolve()
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

    # Complete predictive distributions, not just derived summaries.
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

        # Every model object and uncertainty member is archived.
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

        # The comparator's pre-panel reference must come from this seed's training cohort.
        sim_seed = config["base_seed"] + 1009 * seed_index
        cohort = crp.simulate_patients(config["n_total"], config["n_days"], config["future_days"], sim_seed)
        expected_ref = crp.training_lab_reference(cohort[:config["n_train"]])
        np.testing.assert_allclose(np.asarray(hz_base["initial_loglab"]), expected_ref, atol=1e-12)
        assert not np.allclose(expected_ref, crp.TRUE_L_LOGMU[0])

    source = (root / "crohns_hmm_pipeline.py").read_text()
    assert "TRUE_L_MED[0]" not in source and "TRUE_L_LOGMU[0]" not in source

    # Recompute every inferential table from canonical archived outputs.
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
                   "complete PMF arrays", "PENDING_AUTHOR_DEPOSIT"]:
        # DOI status is in release metadata rather than necessarily typeset.
        if phrase == "PENDING_AUTHOR_DEPOSIT":
            continue
        assert phrase in tex
    assert re.search(r"HMM \+ draw model\s*&\s*[0-9]", tex)

    required_figures = {"time_to_flare_interface.pdf", "calibration_30d.pdf", "score_nll.pdf",
                        "score_ibs4.pdf", "coverage90.pdf", "lambda_recovery.pdf"}
    assert required_figures.issubset({p.name for p in (root / "figures").glob("*.pdf")})
    pdf = root / "Crohns_HMM_Time_to_Flare_Study.pdf"
    assert pdf.exists() and pdf.stat().st_size > 100_000

    meta = json.loads((root / "release_metadata.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", meta["source_commit"])
    assert meta["release_tag"] not in {"", "UNAVAILABLE"}
    assert meta["public_archive_doi"] == "PENDING_AUTHOR_DEPOSIT"
    bundle = root / meta["source_bundle"]
    assert bundle.exists() and sha256(bundle) == meta["source_bundle_sha256"]
    verify_manifest(root)

    print("Release verification passed: complete PMFs, model objects, canonical tables, TeX, PDF, and release identity agree.")


if __name__ == "__main__":
    main()
