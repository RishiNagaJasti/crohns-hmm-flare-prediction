from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import crohns_hmm_pipeline as crp


def true_params(model_type: str = "endogenous") -> crp.HMMParams:
    pm_mu = None
    pm_sd = None
    if model_type == "pattern_mixture":
        pm_mu = np.repeat(crp.TRUE_W_MU[:, None, :], 2, axis=1)
        pm_sd = np.repeat(crp.TRUE_W_SD[:, None, :], 2, axis=1)
    return crp.HMMParams(
        pi=crp.TRUE_PI.copy(),
        P=crp.TRUE_P.copy(),
        wear_mu=crp.TRUE_W_MU.copy(),
        wear_sd=crp.TRUE_W_SD.copy(),
        lab_logmu=crp.TRUE_L_LOGMU.copy(),
        lab_logsd=crp.TRUE_L_LOGSD.copy(),
        lam=crp.TRUE_LAMBDA.copy() if model_type == "endogenous" else None,
        model_type=model_type,
        pm_wear_mu=pm_mu,
        pm_wear_sd=pm_sd,
    )


def test_absent_laboratory_value_never_enters_likelihood() -> None:
    patient = crp.simulate_patients(1, 30, 20, seed=11)[0]
    absent = np.flatnonzero(~patient.draw)
    assert len(absent) > 0
    day = int(absent[0])
    lab_a = patient.lab.copy(); lab_b = patient.lab.copy()
    lab_a[day] = np.array([1e-12, 1e12, 3.14159])
    lab_b[day] = np.array([9e8, 7e-8, 2718.0])
    pa = crp.Patient(patient.patient_id, patient.states_full.copy(), patient.wear.copy(), lab_a, patient.draw.copy())
    pb = crp.Patient(patient.patient_id, patient.states_full.copy(), patient.wear.copy(), lab_b, patient.draw.copy())
    np.testing.assert_allclose(crp.emission_loglik(pa, true_params()),
                               crp.emission_loglik(pb, true_params()), rtol=0.0, atol=0.0)


def test_observed_lognormal_value_changes_likelihood() -> None:
    patient = crp.simulate_patients(1, 80, 20, seed=22)[0]
    drawn = np.flatnonzero(patient.draw)
    assert len(drawn) > 0
    day = int(drawn[0])
    altered = crp.Patient(patient.patient_id, patient.states_full.copy(), patient.wear.copy(),
                          patient.lab.copy(), patient.draw.copy())
    altered.lab[day] *= 2.0
    assert not np.allclose(crp.emission_loglik(patient, true_params())[day],
                           crp.emission_loglik(altered, true_params())[day])


def test_hitting_time_pmf_normalizes_with_day_zero_and_tail() -> None:
    patient = crp.simulate_patients(1, 40, 40, seed=33)[0]
    pred = crp.predict_hmm(patient, [true_params()], H=25)
    np.testing.assert_allclose(pred.pmf.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(pred.pmf >= 0.0)
    np.testing.assert_allclose(pred.pmf[:, 0], pred.state_post[:, 2], atol=1e-12)
    assert np.all(pred.pmf[:, -1] >= 0.0)


def test_transient_conditional_mean_obeys_convex_bound() -> None:
    patient = crp.simulate_patients(1, 50, 50, seed=44)[0]
    params = true_params()
    pred = crp.predict_hmm(patient, [params], H=40)
    _, _, N = crp.hitting_components(params.P)
    means = N @ np.ones(2)
    finite = np.isfinite(pred.cond_mean)
    assert finite.any()
    assert np.all(pred.cond_mean[finite] >= means.min() - 1e-10)
    assert np.all(pred.cond_mean[finite] <= means.max() + 1e-10)


def test_parameter_mixture_conditions_after_marginalization() -> None:
    patient = crp.simulate_patients(1, 20, 20, seed=55)[0]
    p1 = true_params(); p2 = true_params()
    p2.pi = np.array([0.05, 0.05, 0.90])
    p2.P = np.array([[0.80, 0.10, 0.10], [0.05, 0.75, 0.20], [0.01, 0.09, 0.90]])
    pred = crp.predict_hmm(patient, [p1, p2], H=30)
    transient_mass = pred.state_post[:, :2].sum(axis=1)
    expected = np.divide(pred.uncond_mean, transient_mass,
                         out=np.full_like(pred.uncond_mean, np.nan), where=transient_mass > 1e-8)
    np.testing.assert_allclose(pred.cond_mean, expected, equal_nan=True, atol=1e-12)


def test_tail_category_is_observed_not_missing() -> None:
    H = 10
    patient = crp.Patient(0, np.zeros(30, dtype=np.int64),
                          np.tile(crp.TRUE_W_MU[0], (20, 1)),
                          np.full((20, crp.J), np.nan), np.zeros(20, dtype=bool))
    assert crp.true_next_flare_full(patient, d=3, max_h=H) == H + 1


def test_simulator_uses_fresh_day_level_wearable_noise() -> None:
    patient = crp.simulate_patients(1, 100, 20, seed=66)[0]
    repeated = False
    for state in range(crp.K):
        rows = patient.wear[patient.states_obs == state]
        if len(rows) >= 3:
            repeated = True
            assert np.any(np.std(rows, axis=0) > 0.0)
    assert repeated


def test_fit_hmm_does_not_read_oracle_states() -> None:
    class ObservationOnly:
        def __init__(self, wear: np.ndarray, lab: np.ndarray, draw: np.ndarray):
            self.wear = wear; self.lab = lab; self.draw = draw
        @property
        def n_obs(self) -> int: return int(self.wear.shape[0])
        @property
        def states_full(self): raise AssertionError("oracle states were accessed")
        @property
        def states_obs(self): raise AssertionError("oracle states were accessed")

    rng = np.random.default_rng(77); patients = []
    for pid in range(12):
        state = pid % 3
        wear = rng.normal(crp.TRUE_W_MU[state], crp.TRUE_W_SD[state], size=(12, crp.W))
        draw = rng.random(12) < crp.TRUE_LAMBDA[state]
        lab = np.full((12, crp.J), np.nan)
        if draw.any():
            lab[draw] = np.exp(rng.normal(crp.TRUE_L_LOGMU[state], crp.TRUE_L_LOGSD[state],
                                          size=(draw.sum(), crp.J)))
        patients.append(ObservationOnly(wear, lab, draw))
    fit = crp.fit_hmm(patients, "endogenous", n_starts=1, max_iter=3, seed=8)
    np.testing.assert_allclose(fit.P.sum(axis=1), 1.0, atol=1e-10)


def test_fitting_source_contains_no_state_array_reference() -> None:
    source = inspect.getsource(crp.fit_hmm)
    assert ".states_full" not in source and ".states_obs" not in source


def test_split_identifiers_are_disjoint() -> None:
    patients = crp.simulate_patients(20, 10, 10, seed=88)
    train, val, test = patients[:10], patients[10:14], patients[14:]
    ids = [set(p.patient_id for p in group) for group in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])


def test_event_history_reference_is_training_derived_not_generator() -> None:
    states = np.zeros(8, dtype=np.int64)
    draw = np.array([False, True, False, False, True, False, False, False])
    lab = np.full((8, crp.J), np.nan)
    lab[draw] = np.array([[100.0, 200.0, 20.0], [400.0, 800.0, 40.0]])
    p = crp.Patient(1, states, np.tile(crp.TRUE_W_MU[0], (8, 1)), lab, draw)
    ref = crp.training_lab_reference([p])
    expected = np.median(np.log(lab[draw]), axis=0)
    np.testing.assert_allclose(ref, expected)
    assert not np.allclose(ref, crp.TRUE_L_LOGMU[0])


def test_event_history_has_explicit_panel_history_indicator() -> None:
    states = np.zeros(5, dtype=np.int64)
    draw = np.array([False, False, True, False, False])
    lab = np.full((5, crp.J), np.nan); lab[2] = np.array([5.0, 10.0, 7.0])
    p = crp.Patient(1, states, np.tile(crp.TRUE_W_MU[0], (5, 1)), lab, draw)
    F = crp.landmark_features(p, np.log(np.array([2.0, 8.0, 6.0])))
    idx = crp.event_history_feature_names().index("prior_panel_observed")
    np.testing.assert_array_equal(F[:, idx], np.array([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_hmm_multistart_diagnostics_are_archivable() -> None:
    patients = crp.simulate_patients(9, 18, 20, seed=99)
    fit = crp.fit_hmm(patients, "endogenous", n_starts=2, max_iter=3, seed=5)
    assert fit.start_diagnostics is not None and len(fit.start_diagnostics) == 2
    assert sum(bool(x["selected"]) for x in fit.start_diagnostics) == 1
    json.dumps(crp.hmm_params_to_dict(fit))


def test_draw_stratified_serialization_preserves_stratum_parameters() -> None:
    p = true_params("pattern_mixture")
    obj = crp.hmm_params_to_dict(p)
    assert obj["pm_wear_mu"] is not None and obj["pm_wear_sd"] is not None
    assert np.asarray(obj["pm_wear_mu"]).shape == (crp.K, 2, crp.W)


def test_complete_prediction_archive_schema_round_trip(tmp_path: Path) -> None:
    p = crp.simulate_patients(1, 12, 20, seed=101)[0]
    pred = crp.predict_hmm(p, [true_params()], H=10)
    path = tmp_path / "dist.npz"
    np.savez_compressed(path, pmf=pred.pmf.astype(np.float32),
                        state_posterior=pred.state_post.astype(np.float32))
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_allclose(z["pmf"].sum(axis=-1), 1.0, atol=2e-6)
        np.testing.assert_allclose(z["pmf"][:, 0], z["state_posterior"][:, 2], atol=2e-6)


def test_hazard_pickle_payload_is_module_independent(tmp_path: Path) -> None:
    patients = crp.simulate_patients(18, 30, 20, seed=202)
    model = crp.fit_hazard_model(patients, max_horizon=15, seed=7)
    target = tmp_path / "hazard"
    crp.save_hazard_model(model, target)
    import pickle
    with target.with_suffix(".pkl").open("rb") as fh:
        payload = pickle.load(fh)
    assert isinstance(payload, dict)
    assert payload == json.loads(target.with_suffix(".json").read_text())
    assert payload["feature_names"] == crp.event_history_feature_names()
