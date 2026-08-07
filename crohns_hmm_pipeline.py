#!/usr/bin/env python3
"""Reproducible analysis pipeline for the Crohn's time-to-flare study.

This script is intentionally self-contained. It implements:
  * a coherent simulator with fresh day-level Gaussian wearable draws;
  * a jointly observed laboratory panel with log-normal emissions;
  * latent-state HMM estimation by Baum-Welch/EM (no oracle labels);
  * an endogenous-sampling likelihood with the laboratory-draw indicator;
  * a no-lambda HMM and a draw-stratified-emission HMM comparator;
  * a supervised discrete-time event-history comparator;
  * unconditional and transient-conditional time-to-flare laws;
  * right-censoring-aware log scores, fixed-horizon Brier scores,
    calibration, sharpness, and long-horizon interval coverage;
  * empirical-Bayes HMM mixtures and patient-bootstrap hazard ensembles;
  * patient/seed-clustered uncertainty for reported performance.

The file generates manuscript tables, figures, and machine-readable outputs.
It is provided as a transparent implementation of the stated analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import subprocess
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, logsumexp
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
K = 3
STATE_NAMES = ("Remission", "Mild", "Flare")
W_NAMES = ("Temperature", "Resting heart rate", "Stool frequency")
L_NAMES = ("CRP", "ESR", "WBC")
W = len(W_NAMES)
J = len(L_NAMES)

# Population data-generating parameters. The flare state is not absorbing in
# the physiological process; it is made absorbing only for first-passage
# calculations at each landmark.
TRUE_PI = np.array([0.55, 0.35, 0.10], dtype=float)
TRUE_P = np.array([
    [0.93, 0.06, 0.01],
    [0.05, 0.87, 0.08],
    [0.01, 0.09, 0.90],
], dtype=float)
TRUE_W_MU = np.array([
    [36.80, 70.0, 2.0],
    [37.05, 76.0, 4.0],
    [37.35, 84.0, 7.0],
], dtype=float)
TRUE_W_SD = np.array([
    [0.18, 4.5, 0.70],
    [0.22, 5.0, 0.85],
    [0.28, 5.5, 1.00],
], dtype=float)
# Log-normal laboratory medians and log-scale standard deviations.
TRUE_L_MED = np.array([
    [2.0, 10.0, 6.5],
    [10.0, 25.0, 9.0],
    [40.0, 50.0, 13.0],
], dtype=float)
TRUE_L_LOGMU = np.log(TRUE_L_MED)
TRUE_L_LOGSD = np.array([
    [0.45, 0.35, 0.16],
    [0.42, 0.32, 0.15],
    [0.38, 0.30, 0.14],
], dtype=float)
TRUE_LAMBDA = np.array([0.03, 0.15, 0.35], dtype=float)

DEFAULT_HORIZONS = (7, 14, 30, 60)
MODEL_DRAW = "HMM + draw model"
MODEL_NO_DRAW = "HMM without draw model"
MODEL_DRAW_STRATIFIED = "Draw-stratified-emission HMM"
MODEL_EVENT_HISTORY = "Discrete-time event-history"
MODEL_ORDER = (MODEL_DRAW, MODEL_NO_DRAW, MODEL_DRAW_STRATIFIED, MODEL_EVENT_HISTORY)


@dataclass
class Patient:
    patient_id: int
    states_full: NDArray[np.int64]
    wear: NDArray[np.float64]
    lab: NDArray[np.float64]
    draw: NDArray[np.bool_]

    @property
    def n_obs(self) -> int:
        return int(self.wear.shape[0])

    @property
    def states_obs(self) -> NDArray[np.int64]:
        return self.states_full[: self.n_obs]


@dataclass
class HMMParams:
    pi: NDArray[np.float64]
    P: NDArray[np.float64]
    wear_mu: NDArray[np.float64]
    wear_sd: NDArray[np.float64]
    lab_logmu: NDArray[np.float64]
    lab_logsd: NDArray[np.float64]
    lam: Optional[NDArray[np.float64]]
    model_type: str = "endogenous"
    # Draw-stratified wearable parameters indexed [state, stratum, feature].
    pm_wear_mu: Optional[NDArray[np.float64]] = None
    pm_wear_sd: Optional[NDArray[np.float64]] = None
    loglik: float = float("nan")
    n_iter: int = 0
    em_loglik_trace: Optional[List[float]] = None
    start_diagnostics: Optional[List[Dict[str, object]]] = None

    def copy(self) -> "HMMParams":
        return HMMParams(
            pi=self.pi.copy(), P=self.P.copy(),
            wear_mu=self.wear_mu.copy(), wear_sd=self.wear_sd.copy(),
            lab_logmu=self.lab_logmu.copy(), lab_logsd=self.lab_logsd.copy(),
            lam=None if self.lam is None else self.lam.copy(),
            model_type=self.model_type,
            pm_wear_mu=None if self.pm_wear_mu is None else self.pm_wear_mu.copy(),
            pm_wear_sd=None if self.pm_wear_sd is None else self.pm_wear_sd.copy(),
            loglik=self.loglik, n_iter=self.n_iter,
            em_loglik_trace=None if self.em_loglik_trace is None else list(self.em_loglik_trace),
            start_diagnostics=(None if self.start_diagnostics is None else
                               json.loads(json.dumps(self.start_diagnostics))),
        )


@dataclass
class HazardModel:
    scaler: StandardScaler
    current_model: LogisticRegression
    future_model: LogisticRegression
    max_train_horizon: int
    initial_loglab: NDArray[np.float64]
    fit_seed: int
    training_patient_ids: List[int]
    bootstrap_patient_ids: Optional[List[int]] = None


@dataclass
class Prediction:
    pmf: NDArray[np.float64]       # shape [T, H+2], last column is >H tail
    state_post: NDArray[np.float64]  # shape [T, K]
    cond_mean: NDArray[np.float64]
    uncond_mean: NDArray[np.float64]


def normalize_rows(a: NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.clip(a, EPS, None)
    return a / a.sum(axis=1, keepdims=True)


def simulate_patients(
    n_patients: int,
    n_obs: int,
    future_days: int,
    seed: int,
    scenario: str = "well_specified",
) -> List[Patient]:
    """Simulate complete latent paths and the first n_obs days of observations.

    Scenarios:
      well_specified: common P and model-matched emissions.
      heavy_tail: Student-t wearable and log-lab errors at test time.
      heterogeneity: patient random intercepts and perturbed transitions.
      semi_markov: duration-dependent persistence in latent states.
    """
    rng = np.random.default_rng(seed)
    total = n_obs + future_days
    patients: List[Patient] = []

    for pid in range(n_patients):
        if scenario == "heterogeneity":
            # Moderate, mean-zero patient heterogeneity not represented in the
            # pooled fitted model.
            w_off = rng.normal(0.0, np.array([0.10, 2.0, 0.25]), size=W)
            l_off = rng.normal(0.0, np.array([0.18, 0.12, 0.08]), size=J)
            rows = []
            for k in range(K):
                concentration = 80.0 * TRUE_P[k] + 0.5
                rows.append(rng.dirichlet(concentration))
            P_patient = np.vstack(rows)
        else:
            w_off = np.zeros(W)
            l_off = np.zeros(J)
            P_patient = TRUE_P

        states = np.empty(total, dtype=np.int64)
        states[0] = rng.choice(K, p=TRUE_PI)
        duration = 1
        for t in range(1, total):
            prev = states[t - 1]
            if scenario == "semi_markov":
                # Persistence increases modestly with duration, violating the
                # memoryless transition assumption while preserving ordering.
                base = TRUE_P[prev].copy()
                extra = min(0.06, 0.004 * max(duration - 1, 0))
                move_mass = 1.0 - base[prev]
                if move_mass > EPS:
                    base[np.arange(K) != prev] *= max(move_mass - extra, 0.002) / move_mass
                    base[prev] = 1.0 - base[np.arange(K) != prev].sum()
                probs = base
            else:
                probs = P_patient[prev]
            states[t] = rng.choice(K, p=probs)
            duration = duration + 1 if states[t] == prev else 1

        wear = np.empty((n_obs, W), dtype=float)
        lab = np.full((n_obs, J), np.nan, dtype=float)
        draw = np.empty(n_obs, dtype=bool)

        for t in range(n_obs):
            k = states[t]
            if scenario == "heavy_tail":
                # Scale t_4 noise to unit variance: sqrt((nu-2)/nu).
                z = rng.standard_t(df=4, size=W) * math.sqrt(0.5)
                wear[t] = TRUE_W_MU[k] + w_off + TRUE_W_SD[k] * z
            else:
                wear[t] = rng.normal(TRUE_W_MU[k] + w_off, TRUE_W_SD[k])

            draw[t] = rng.random() < TRUE_LAMBDA[k]
            if draw[t]:
                if scenario == "heavy_tail":
                    z = rng.standard_t(df=4, size=J) * math.sqrt(0.5)
                    log_lab = TRUE_L_LOGMU[k] + l_off + TRUE_L_LOGSD[k] * z
                else:
                    log_lab = rng.normal(TRUE_L_LOGMU[k] + l_off, TRUE_L_LOGSD[k])
                lab[t] = np.exp(log_lab)

        patients.append(Patient(pid, states, wear, lab, draw))
    return patients


def _hard_initial_labels(patients: Sequence[Patient], random_state: int) -> List[NDArray[np.int64]]:
    X = np.vstack([p.wear for p in patients])
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    raw = km.fit_predict(Z)
    centers = km.cluster_centers_
    severity = centers.mean(axis=1)
    order = np.argsort(severity)
    remap = np.empty(K, dtype=int)
    remap[order] = np.arange(K)
    labels_all = remap[raw]
    labels: List[NDArray[np.int64]] = []
    idx = 0
    for p in patients:
        labels.append(labels_all[idx: idx + p.n_obs])
        idx += p.n_obs
    return labels


def _params_from_labels(
    patients: Sequence[Patient],
    labels: Sequence[NDArray[np.int64]],
    model_type: str,
) -> HMMParams:
    pi_counts = np.ones(K)
    trans = np.ones((K, K)) * 0.5
    for lab in labels:
        pi_counts[lab[0]] += 1
        for t in range(len(lab) - 1):
            trans[lab[t], lab[t + 1]] += 1
    pi = pi_counts / pi_counts.sum()
    P = normalize_rows(trans)

    wear_mu = np.zeros((K, W))
    wear_sd = np.ones((K, W))
    lab_mu = np.zeros((K, J))
    lab_sd = np.ones((K, J)) * 0.5
    lam = np.full(K, 0.15) if model_type == "endogenous" else None

    global_w = np.vstack([p.wear for p in patients])
    global_l = np.log(np.vstack([p.lab[p.draw] for p in patients if p.draw.any()]))
    global_w_mu, global_w_sd = global_w.mean(0), np.maximum(global_w.std(0), 0.1)
    global_l_mu, global_l_sd = global_l.mean(0), np.maximum(global_l.std(0), 0.1)

    pm_mu = np.zeros((K, 2, W)) if model_type == "pattern_mixture" else None
    pm_sd = np.ones((K, 2, W)) if model_type == "pattern_mixture" else None

    for k in range(K):
        wrows = []
        lrows = []
        n_k = 0
        n_l = 0
        for p, z in zip(patients, labels):
            mask = z == k
            n_k += int(mask.sum())
            n_l += int(p.draw[mask].sum())
            if mask.any():
                wrows.append(p.wear[mask])
            mask_l = mask & p.draw
            if mask_l.any():
                lrows.append(np.log(p.lab[mask_l]))
        if wrows:
            wdat = np.vstack(wrows)
            wear_mu[k] = wdat.mean(0)
            wear_sd[k] = np.maximum(wdat.std(0), np.array([0.08, 1.5, 0.25]))
        else:
            wear_mu[k], wear_sd[k] = global_w_mu, global_w_sd
        if lrows:
            ldat = np.vstack(lrows)
            lab_mu[k] = ldat.mean(0)
            lab_sd[k] = np.maximum(ldat.std(0), 0.08)
        else:
            lab_mu[k], lab_sd[k] = global_l_mu, global_l_sd
        if lam is not None:
            lam[k] = (n_l + 1.0) / (n_k + 2.0)

        if model_type == "pattern_mixture":
            assert pm_mu is not None and pm_sd is not None
            for ell in (0, 1):
                rows = []
                for p, z in zip(patients, labels):
                    m = (z == k) & (p.draw.astype(int) == ell)
                    if m.any():
                        rows.append(p.wear[m])
                if rows:
                    dat = np.vstack(rows)
                    pm_mu[k, ell] = dat.mean(0)
                    pm_sd[k, ell] = np.maximum(dat.std(0), np.array([0.08, 1.5, 0.25]))
                else:
                    pm_mu[k, ell] = wear_mu[k]
                    pm_sd[k, ell] = wear_sd[k]

    out = HMMParams(pi, P, wear_mu, wear_sd, lab_mu, lab_sd, lam,
                    model_type=model_type, pm_wear_mu=pm_mu, pm_wear_sd=pm_sd)
    return _sort_states(out)


def _sort_states(params: HMMParams) -> HMMParams:
    # Severity score based on standardized state means. This fixes label
    # switching without using true latent states.
    if params.model_type == "pattern_mixture" and params.pm_wear_mu is not None:
        means = params.pm_wear_mu.mean(axis=1)
    else:
        means = params.wear_mu
    scale = np.maximum(means.std(axis=0), 1e-6)
    score = ((means - means.mean(axis=0)) / scale).mean(axis=1)
    order = np.argsort(score)
    p = params.copy()
    p.pi = p.pi[order]
    p.P = p.P[np.ix_(order, order)]
    p.wear_mu = p.wear_mu[order]
    p.wear_sd = p.wear_sd[order]
    p.lab_logmu = p.lab_logmu[order]
    p.lab_logsd = p.lab_logsd[order]
    if p.lam is not None:
        p.lam = p.lam[order]
    if p.pm_wear_mu is not None:
        p.pm_wear_mu = p.pm_wear_mu[order]
        p.pm_wear_sd = p.pm_wear_sd[order]
    return p


def emission_loglik(patient: Patient, params: HMMParams) -> NDArray[np.float64]:
    T = patient.n_obs
    ll = np.zeros((T, K), dtype=float)
    const_w = math.log(2 * math.pi)
    const_l = math.log(2 * math.pi)

    for k in range(K):
        if params.model_type == "pattern_mixture":
            assert params.pm_wear_mu is not None and params.pm_wear_sd is not None
            for ell in (0, 1):
                mask = patient.draw == bool(ell)
                if not mask.any():
                    continue
                mu = params.pm_wear_mu[k, ell]
                sd = np.maximum(params.pm_wear_sd[k, ell], 1e-5)
                z = (patient.wear[mask] - mu) / sd
                ll[mask, k] += -0.5 * np.sum(const_w + 2 * np.log(sd) + z * z, axis=1)
        else:
            sd = np.maximum(params.wear_sd[k], 1e-5)
            z = (patient.wear - params.wear_mu[k]) / sd
            ll[:, k] += -0.5 * np.sum(const_w + 2 * np.log(sd) + z * z, axis=1)

        if patient.draw.any():
            loglab = np.log(np.clip(patient.lab[patient.draw], EPS, None))
            sd_l = np.maximum(params.lab_logsd[k], 1e-5)
            z_l = (loglab - params.lab_logmu[k]) / sd_l
            # Full log-normal density.  The Jacobian term -log(y) is common
            # across states on a given day and therefore cancels from the
            # posterior ratios, but retaining it keeps the stated likelihood
            # and the implemented probability density exactly aligned.
            ll[patient.draw, k] += (
                -0.5 * np.sum(const_l + 2 * np.log(sd_l) + z_l * z_l, axis=1)
                - np.sum(loglab, axis=1)
            )

        if params.model_type == "endogenous":
            assert params.lam is not None
            lam = float(np.clip(params.lam[k], 1e-6, 1 - 1e-6))
            ll[:, k] += np.where(patient.draw, math.log(lam), math.log1p(-lam))
        # no-lambda and pattern-mixture condition on the observed pattern and
        # therefore do not multiply a Bernoulli draw probability.
    return ll


def forward_backward(patient: Patient, params: HMMParams) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    logB = emission_loglik(patient, params)
    logP = np.log(np.clip(params.P, EPS, None))
    logpi = np.log(np.clip(params.pi, EPS, None))
    T = patient.n_obs

    la = np.empty((T, K), dtype=float)
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + logsumexp(la[t - 1][:, None] + logP, axis=0)
    ll = float(logsumexp(la[-1]))

    lb = np.zeros((T, K), dtype=float)
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(logP + logB[t + 1][None, :] + lb[t + 1][None, :], axis=1)

    gamma = np.exp(la + lb - ll)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.empty((max(T - 1, 0), K, K), dtype=float)
    for t in range(T - 1):
        x = la[t][:, None] + logP + logB[t + 1][None, :] + lb[t + 1][None, :] - ll
        xi[t] = np.exp(x)
        xi[t] /= xi[t].sum()
    return ll, gamma, xi


def filter_only(patient: Patient, params: HMMParams) -> Tuple[float, NDArray[np.float64]]:
    logB = emission_loglik(patient, params)
    logP = np.log(np.clip(params.P, EPS, None))
    logpi = np.log(np.clip(params.pi, EPS, None))
    T = patient.n_obs
    la = np.empty((T, K), dtype=float)
    raw0 = logpi + logB[0]
    c0 = logsumexp(raw0)
    la[0] = raw0 - c0
    ll = float(c0)
    for t in range(1, T):
        raw = logB[t] + logsumexp(la[t - 1][:, None] + logP, axis=0)
        c = logsumexp(raw)
        la[t] = raw - c
        ll += float(c)
    return ll, np.exp(la)


def fit_hmm(
    patients: Sequence[Patient],
    model_type: str,
    n_starts: int = 2,
    max_iter: int = 80,
    tol: float = 1e-4,
    init_params: Optional[HMMParams] = None,
    seed: int = 0,
) -> HMMParams:
    if model_type not in {"endogenous", "no_lambda", "pattern_mixture"}:
        raise ValueError(model_type)
    best: Optional[HMMParams] = None

    starts: List[HMMParams] = []
    if init_params is not None:
        q = init_params.copy()
        q.model_type = model_type
        if model_type == "no_lambda":
            q.lam = None
        starts.append(q)
    for s in range(max(n_starts - len(starts), 0)):
        labels = _hard_initial_labels(patients, random_state=seed + 37 * s)
        starts.append(_params_from_labels(patients, labels, model_type))

    diagnostics: List[Dict[str, object]] = []
    best_start = -1
    for start_index, initial in enumerate(starts):
        params = initial.copy()
        prev = -np.inf
        trace: List[float] = []
        for iteration in range(1, max_iter + 1):
            total_ll = 0.0
            pi_sum = np.ones(K) * 0.5
            trans_sum = np.ones((K, K)) * 0.5
            w_num = np.zeros((K, W))
            w_sq = np.zeros((K, W))
            w_den = np.zeros(K)
            l_num = np.zeros((K, J))
            l_sq = np.zeros((K, J))
            l_den = np.zeros(K)
            draw_num = np.ones(K)
            draw_den = np.ones(K) * 2.0
            if model_type == "pattern_mixture":
                pm_num = np.zeros((K, 2, W))
                pm_sq = np.zeros((K, 2, W))
                pm_den = np.zeros((K, 2))
            else:
                pm_num = pm_sq = pm_den = None

            for p in patients:
                ll, gamma, xi = forward_backward(p, params)
                total_ll += ll
                pi_sum += gamma[0]
                if len(xi):
                    trans_sum += xi.sum(axis=0)
                for k in range(K):
                    g = gamma[:, k]
                    w_den[k] += g.sum()
                    w_num[k] += (g[:, None] * p.wear).sum(axis=0)
                    w_sq[k] += (g[:, None] * p.wear * p.wear).sum(axis=0)
                    if p.draw.any():
                        gd = g[p.draw]
                        llab = np.log(np.clip(p.lab[p.draw], EPS, None))
                        l_den[k] += gd.sum()
                        l_num[k] += (gd[:, None] * llab).sum(axis=0)
                        l_sq[k] += (gd[:, None] * llab * llab).sum(axis=0)
                    if model_type == "endogenous":
                        draw_num[k] += g[p.draw].sum()
                        draw_den[k] += g.sum()
                    if model_type == "pattern_mixture":
                        assert pm_num is not None and pm_sq is not None and pm_den is not None
                        for ell in (0, 1):
                            mask = p.draw == bool(ell)
                            ge = g[mask]
                            if ge.size:
                                pm_den[k, ell] += ge.sum()
                                pm_num[k, ell] += (ge[:, None] * p.wear[mask]).sum(axis=0)
                                pm_sq[k, ell] += (ge[:, None] * p.wear[mask] ** 2).sum(axis=0)

            new = params.copy()
            new.pi = pi_sum / pi_sum.sum()
            new.P = normalize_rows(trans_sum)
            for k in range(K):
                den = max(w_den[k], EPS)
                new.wear_mu[k] = w_num[k] / den
                var = w_sq[k] / den - new.wear_mu[k] ** 2
                new.wear_sd[k] = np.sqrt(np.maximum(var, np.array([0.05**2, 1.0**2, 0.15**2])))
                den_l = max(l_den[k], EPS)
                if l_den[k] > 0.2:
                    new.lab_logmu[k] = l_num[k] / den_l
                    var_l = l_sq[k] / den_l - new.lab_logmu[k] ** 2
                    new.lab_logsd[k] = np.sqrt(np.maximum(var_l, 0.05**2))
                if model_type == "endogenous":
                    assert new.lam is not None
                    new.lam[k] = np.clip(draw_num[k] / draw_den[k], 1e-4, 1 - 1e-4)
                if model_type == "pattern_mixture":
                    assert new.pm_wear_mu is not None and new.pm_wear_sd is not None
                    assert pm_num is not None and pm_sq is not None and pm_den is not None
                    for ell in (0, 1):
                        if pm_den[k, ell] > 0.2:
                            m = pm_num[k, ell] / pm_den[k, ell]
                            v = pm_sq[k, ell] / pm_den[k, ell] - m**2
                            new.pm_wear_mu[k, ell] = m
                            new.pm_wear_sd[k, ell] = np.sqrt(np.maximum(v, np.array([0.05**2, 1.0**2, 0.15**2])))
            new = _sort_states(new)
            new.loglik = total_ll
            new.n_iter = iteration
            trace.append(float(total_ll))
            params = new
            if iteration > 2 and abs(total_ll - prev) <= tol * (1.0 + abs(prev)):
                break
            prev = total_ll

        # Re-evaluate after final relabeling.
        final_ll = sum(forward_backward(p, params)[0] for p in patients)
        params.loglik = float(final_ll)
        params.em_loglik_trace = list(trace)
        diagnostics.append({
            "start_index": int(start_index),
            "final_loglik": float(final_ll),
            "n_iter": int(params.n_iter),
            "loglik_trace": [float(x) for x in trace],
            "selected": False,
        })
        if best is None or params.loglik > best.loglik:
            best = params
            best_start = start_index
    assert best is not None
    diagnostics[best_start]["selected"] = True
    best.start_diagnostics = diagnostics
    best.em_loglik_trace = list(diagnostics[best_start]["loglik_trace"])
    return best


def bootstrap_hmm_ensemble(
    patients: Sequence[Patient],
    model_type: str,
    base: HMMParams,
    n_boot: int,
    seed: int,
    max_iter: int = 45,
) -> List[HMMParams]:
    rng = np.random.default_rng(seed)
    out = [base]
    n = len(patients)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [patients[int(i)] for i in idx]
        fitted = fit_hmm(sample, model_type, n_starts=1, max_iter=max_iter,
                         tol=2e-4, init_params=base, seed=seed + b + 1)
        out.append(fitted)
    return out


def expected_hmm_sufficient_statistics(
    patients: Sequence[Patient], params: HMMParams
) -> Dict[str, NDArray[np.float64]]:
    """Responsibility-weighted sufficient statistics at the EM solution.

    These statistics support a fast empirical-Bayes parameter mixture.  The
    mixture is not a full MCMC posterior: it conditions on the final EM
    responsibilities, but it does propagate transition, emission, initial-
    state, and draw-rate uncertainty through the filtering and hitting-time
    calculations.
    """
    stats: Dict[str, NDArray[np.float64]] = {
        "pi": np.full(K, 0.5),
        "trans": np.full((K, K), 0.5),
        "wear_n": np.zeros(K),
        "wear_sum": np.zeros((K, W)),
        "wear_sumsq": np.zeros((K, W)),
        "lab_n": np.zeros(K),
        "lab_sum": np.zeros((K, J)),
        "lab_sumsq": np.zeros((K, J)),
        "draw_yes": np.ones(K),
        "draw_no": np.ones(K),
    }
    if params.model_type == "pattern_mixture":
        stats["pm_n"] = np.zeros((K, 2))
        stats["pm_sum"] = np.zeros((K, 2, W))
        stats["pm_sumsq"] = np.zeros((K, 2, W))

    for patient in patients:
        _, gamma, xi = forward_backward(patient, params)
        stats["pi"] += gamma[0]
        if len(xi):
            stats["trans"] += xi.sum(axis=0)
        for k in range(K):
            g = gamma[:, k]
            stats["wear_n"][k] += g.sum()
            stats["wear_sum"][k] += (g[:, None] * patient.wear).sum(axis=0)
            stats["wear_sumsq"][k] += (g[:, None] * patient.wear**2).sum(axis=0)
            if patient.draw.any():
                gd = g[patient.draw]
                z = np.log(np.clip(patient.lab[patient.draw], EPS, None))
                stats["lab_n"][k] += gd.sum()
                stats["lab_sum"][k] += (gd[:, None] * z).sum(axis=0)
                stats["lab_sumsq"][k] += (gd[:, None] * z**2).sum(axis=0)
            if params.model_type == "endogenous":
                stats["draw_yes"][k] += g[patient.draw].sum()
                stats["draw_no"][k] += g[~patient.draw].sum()
            if params.model_type == "pattern_mixture":
                for ell in (0, 1):
                    mask = patient.draw == bool(ell)
                    ge = g[mask]
                    if ge.size:
                        stats["pm_n"][k, ell] += ge.sum()
                        stats["pm_sum"][k, ell] += (ge[:, None] * patient.wear[mask]).sum(axis=0)
                        stats["pm_sumsq"][k, ell] += (ge[:, None] * patient.wear[mask]**2).sum(axis=0)
    return stats


def _draw_normal_parameters(
    rng: np.random.Generator,
    n_eff: float,
    sum_x: NDArray[np.float64],
    sumsq_x: NDArray[np.float64],
    base_mu: NDArray[np.float64],
    base_sd: NDArray[np.float64],
    min_sd: NDArray[np.float64] | float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Approximate conjugate draw for independent Gaussian channels."""
    n = max(float(n_eff), 2.5)
    empirical_mu = sum_x / max(float(n_eff), EPS) if n_eff > EPS else base_mu
    empirical_var = (sumsq_x / max(float(n_eff), EPS) - empirical_mu**2) if n_eff > EPS else base_sd**2
    empirical_var = np.maximum(empirical_var, np.asarray(min_sd)**2)
    # Weak prior centered at the EM estimate, equivalent to two observations.
    kappa0 = 2.0
    mu0 = base_mu
    pooled_mu = (max(float(n_eff), 0.0) * empirical_mu + kappa0 * mu0) / (max(float(n_eff), 0.0) + kappa0)
    pooled_var = (max(float(n_eff), 0.0) * empirical_var + kappa0 * base_sd**2) / (max(float(n_eff), 0.0) + kappa0)
    df = max(int(round(n + kappa0 - 1.0)), 3)
    var_draw = pooled_var * df / rng.chisquare(df, size=pooled_var.shape)
    var_draw = np.maximum(var_draw, np.asarray(min_sd)**2)
    mu_draw = rng.normal(pooled_mu, np.sqrt(var_draw / (n + kappa0)))
    return mu_draw, np.sqrt(var_draw)


def approximate_parameter_ensemble(
    patients: Sequence[Patient],
    base: HMMParams,
    n_draws: int,
    seed: int,
) -> List[HMMParams]:
    """Fast responsibility-weighted parameter-uncertainty mixture.

    Draws use Dirichlet posteriors for initial/transition probabilities, a
    beta posterior for endogenous draw rates, and weakly regularized
    normal/scale draws for emission parameters.  The base EM estimate is
    included as draw zero.
    """
    if n_draws <= 0:
        return [base]
    stats = expected_hmm_sufficient_statistics(patients, base)
    rng = np.random.default_rng(seed)
    out = [base]
    for _ in range(n_draws):
        p = base.copy()
        p.pi = rng.dirichlet(np.clip(stats["pi"], 1e-3, None))
        p.P = np.vstack([rng.dirichlet(np.clip(stats["trans"][k], 1e-3, None)) for k in range(K)])
        for k in range(K):
            p.wear_mu[k], p.wear_sd[k] = _draw_normal_parameters(
                rng, stats["wear_n"][k], stats["wear_sum"][k], stats["wear_sumsq"][k],
                base.wear_mu[k], base.wear_sd[k], np.array([0.05, 1.0, 0.15]))
            p.lab_logmu[k], p.lab_logsd[k] = _draw_normal_parameters(
                rng, stats["lab_n"][k], stats["lab_sum"][k], stats["lab_sumsq"][k],
                base.lab_logmu[k], base.lab_logsd[k], 0.05)
            if p.model_type == "endogenous":
                assert p.lam is not None
                p.lam[k] = rng.beta(max(stats["draw_yes"][k], 1e-3),
                                    max(stats["draw_no"][k], 1e-3))
            if p.model_type == "pattern_mixture":
                assert p.pm_wear_mu is not None and p.pm_wear_sd is not None
                for ell in (0, 1):
                    p.pm_wear_mu[k, ell], p.pm_wear_sd[k, ell] = _draw_normal_parameters(
                        rng, stats["pm_n"][k, ell], stats["pm_sum"][k, ell],
                        stats["pm_sumsq"][k, ell], base.pm_wear_mu[k, ell],
                        base.pm_wear_sd[k, ell], np.array([0.05, 1.0, 0.15]))
        p = _sort_states(p)
        out.append(p)
    return out

def hitting_components(P: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    Q = P[:2, :2].copy()
    r = P[:2, 2].copy()
    # P rows sum to one, so r is the transient-to-flare probability. The flare
    # row is irrelevant because F is made absorbing for this calculation.
    N = np.linalg.inv(np.eye(2) - Q)
    return Q, r, N


def predict_hmm(patient: Patient, ensemble: Sequence[HMMParams], H: int) -> Prediction:
    T = patient.n_obs
    pmf_acc = np.zeros((T, H + 2), dtype=float)
    state_acc = np.zeros((T, K), dtype=float)
    uncond_acc = np.zeros(T, dtype=float)

    for params in ensemble:
        _, alpha = filter_only(patient, params)
        Q, r, N = hitting_components(params.P)
        q = alpha[:, :2].copy()
        mass_t = q.sum(axis=1)
        state_acc += alpha
        pmf = np.zeros((T, H + 2), dtype=float)
        pmf[:, 0] = alpha[:, 2]
        v = q.copy()
        for n in range(1, H + 1):
            pmf[:, n] = v @ r
            v = v @ Q
        pmf[:, H + 1] = v.sum(axis=1)
        pmf /= pmf.sum(axis=1, keepdims=True)
        pmf_acc += pmf
        tvec = N @ np.ones(2)
        uncond = q @ tvec
        uncond_acc += uncond

    m = float(len(ensemble))
    pmf_acc /= m
    state_acc /= m
    uncond_acc /= m
    # For the parameter-mixture predictive law, conditioning must be applied
    # after marginalizing the joint state/parameter distribution.  Averaging
    # draw-specific conditional means would generally use the wrong weights.
    transient_mass = state_acc[:, :2].sum(axis=1)
    cond_mean = np.divide(
        uncond_acc, transient_mass,
        out=np.full_like(uncond_acc, np.nan),
        where=transient_mass > 1e-8,
    )
    pmf_acc /= pmf_acc.sum(axis=1, keepdims=True)
    return Prediction(pmf_acc, state_acc, cond_mean, uncond_acc)


def training_lab_reference(patients: Sequence[Patient]) -> NDArray[np.float64]:
    """Training-derived causal reference for pre-panel log-laboratory features."""
    rows = [np.log(np.clip(p.lab[p.draw], EPS, None))
            for p in patients if p.draw.any()]
    if not rows:
        raise ValueError("training cohort contains no observed laboratory panels")
    return np.median(np.vstack(rows), axis=0)


def event_history_feature_names() -> List[str]:
    names: List[str] = []
    names.extend([f"wear_current_{x}" for x in W_NAMES])
    names.extend([f"wear_mean7_{x}" for x in W_NAMES])
    names.extend([f"wear_slope7_{x}" for x in W_NAMES])
    names.extend(["panel_drawn_today", "prior_panel_observed"])
    names.extend([f"last_log_{x}" for x in L_NAMES])
    names.append("time_since_panel_scaled")
    return names


def landmark_features(patient: Patient, initial_loglab: NDArray[np.float64]) -> NDArray[np.float64]:
    """Causal features available through each event-history landmark."""
    T = patient.n_obs
    feats: List[NDArray[np.float64]] = []
    last_loglab = np.asarray(initial_loglab, dtype=float).copy()
    if last_loglab.shape != (J,):
        raise ValueError(f"initial_loglab must have shape {(J,)}, got {last_loglab.shape}")
    time_since = 60.0
    prior_panel_observed = 0.0
    for t in range(T):
        if patient.draw[t]:
            last_loglab = np.log(np.clip(patient.lab[t], EPS, None))
            time_since = 0.0
            prior_panel_observed = 1.0
        else:
            time_since += 1.0
        lo = max(0, t - 6)
        hist = patient.wear[lo:t + 1]
        mean7 = hist.mean(axis=0)
        if len(hist) >= 2:
            x = np.arange(len(hist), dtype=float)
            xc = x - x.mean()
            slope = (xc[:, None] * (hist - hist.mean(axis=0))).sum(axis=0) / max((xc**2).sum(), EPS)
        else:
            slope = np.zeros(W)
        feats.append(np.concatenate([
            patient.wear[t], mean7, slope,
            np.array([float(patient.draw[t]), prior_panel_observed]),
            last_loglab,
            np.array([min(time_since, 60.0) / 60.0]),
        ]))
    result = np.vstack(feats)
    if result.shape[1] != len(event_history_feature_names()):
        raise AssertionError("event-history feature count mismatch")
    return result


def _next_flare_in_observed(states: NDArray[np.int64], d: int) -> Optional[int]:
    hits = np.flatnonzero(states[d:] == 2)
    return int(hits[0]) if len(hits) else None


def fit_hazard_model(
    patients: Sequence[Patient],
    max_horizon: int = 120,
    seed: int = 0,
    bootstrap_patient_ids: Optional[List[int]] = None,
) -> HazardModel:
    Xcur: List[NDArray[np.float64]] = []
    ycur: List[int] = []
    Xhaz: List[NDArray[np.float64]] = []
    yhaz: List[int] = []
    initial_loglab = training_lab_reference(patients)
    for p in patients:
        F = landmark_features(p, initial_loglab)
        st = p.states_obs
        for d in range(0, p.n_obs, 3):
            Xcur.append(F[d]); ycur.append(int(st[d] == 2))
            if st[d] == 2:
                continue
            hit_rel = _next_flare_in_observed(st, d + 1) if d + 1 < len(st) else None
            hit = None if hit_rel is None else hit_rel + 1
            c = min(max_horizon, len(st) - 1 - d)
            for n in range(1, c + 1):
                time_basis = np.array([math.log1p(n), math.sqrt(n) / 10.0, n / max_horizon])
                Xhaz.append(np.concatenate([F[d], time_basis]))
                event = hit is not None and hit == n
                yhaz.append(int(event))
                if event:
                    break
    Xcur_arr = np.vstack(Xcur)
    Xhaz_arr = np.vstack(Xhaz)
    if len(np.unique(ycur)) < 2 or len(np.unique(yhaz)) < 2:
        raise ValueError("event-history training sample contains only one outcome class")
    scaler = StandardScaler().fit(np.vstack([Xcur_arr, Xhaz_arr[:, :Xcur_arr.shape[1]]]))
    Xcur_s = scaler.transform(Xcur_arr)
    Xhaz_base = scaler.transform(Xhaz_arr[:, :Xcur_arr.shape[1]])
    Xhaz_s = np.hstack([Xhaz_base, Xhaz_arr[:, Xcur_arr.shape[1]:]])
    current = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    current.fit(Xcur_s, np.asarray(ycur))
    future = LogisticRegression(max_iter=1000, C=0.5, random_state=seed + 1)
    future.fit(Xhaz_s, np.asarray(yhaz))
    return HazardModel(
        scaler=scaler, current_model=current, future_model=future,
        max_train_horizon=max_horizon, initial_loglab=initial_loglab,
        fit_seed=int(seed), training_patient_ids=[int(p.patient_id) for p in patients],
        bootstrap_patient_ids=(None if bootstrap_patient_ids is None else
                               [int(x) for x in bootstrap_patient_ids]),
    )


def bootstrap_hazard_ensemble(
    patients: Sequence[Patient], base: HazardModel, n_boot: int, seed: int
) -> List[HazardModel]:
    rng = np.random.default_rng(seed)
    out = [base]
    n = len(patients)
    for b in range(n_boot):
        fitted: Optional[HazardModel] = None
        for attempt in range(20):
            idx = rng.integers(0, n, size=n)
            sample = [patients[int(i)] for i in idx]
            sampled_ids = [int(patients[int(i)].patient_id) for i in idx]
            try:
                fitted = fit_hazard_model(
                    sample, max_horizon=base.max_train_horizon,
                    seed=seed + 100 * b + attempt + 1,
                    bootstrap_patient_ids=sampled_ids,
                )
                break
            except ValueError:
                continue
        if fitted is None:
            raise RuntimeError(f"failed to fit hazard bootstrap member {b}")
        out.append(fitted)
    return out


def predict_hazard(patient: Patient, ensemble: Sequence[HazardModel], H: int) -> Prediction:
    """Vectorized discrete-time event-history prediction."""
    T = patient.n_obs
    F_by_model = [landmark_features(patient, model.initial_loglab) for model in ensemble]
    pmf_acc = np.zeros((T, H + 2))
    state_acc = np.zeros((T, K))
    n_grid = np.arange(1, H + 1, dtype=float)
    tb = np.column_stack([np.log1p(n_grid), np.sqrt(n_grid) / 10.0,
                          n_grid / max(1.0, float(ensemble[0].max_train_horizon))])
    for model, F in zip(ensemble, F_by_model):
        Xs = model.scaler.transform(F)
        p0 = model.current_model.predict_proba(Xs)[:, 1]
        Xbase = np.repeat(Xs, H, axis=0)
        Tb = np.tile(tb, (T, 1))
        h = model.future_model.predict_proba(np.hstack([Xbase, Tb]))[:, 1].reshape(T, H)
        h = np.clip(h, 1e-5, 1 - 1e-5)
        one_minus = 1.0 - h
        prev_surv = np.concatenate([np.ones((T, 1)), np.cumprod(one_minus[:, :-1], axis=1)], axis=1)
        pmf = np.zeros((T, H + 2))
        pmf[:, 0] = p0
        pmf[:, 1:H + 1] = (1.0 - p0)[:, None] * prev_surv * h
        pmf[:, H + 1] = (1.0 - p0) * np.prod(one_minus, axis=1)
        pmf /= pmf.sum(axis=1, keepdims=True)
        pmf_acc += pmf
        state_acc[:, 2] += p0
        state_acc[:, 0] += 1 - p0
    pmf_acc /= len(ensemble)
    state_acc /= len(ensemble)
    grid = np.arange(H + 1, dtype=float)
    uncond = (pmf_acc[:, :H + 1] * grid[None, :]).sum(axis=1) + pmf_acc[:, H + 1] * (H + 1)
    mass = 1 - pmf_acc[:, 0]
    cond = np.divide(uncond, mass, out=np.full_like(uncond, np.nan), where=mass > 1e-8)
    return Prediction(pmf_acc, state_acc, cond, uncond)


def true_next_flare_full(patient: Patient, d: int, max_h: int) -> int:
    """Return the coarsened complete-data outcome on {0,...,H,>H}.

    The simulator supplies at least ``max_h`` latent future days beyond every
    evaluated landmark.  Therefore, absence of a flare in the finite window is
    an observed tail-category outcome, represented by ``max_h + 1``; it is not
    missing data and must not be dropped from interval-coverage calculations.
    """
    window = patient.states_full[d: d + max_h + 1]
    if len(window) < max_h + 1:
        raise ValueError("latent continuation is too short for coverage evaluation")
    hits = np.flatnonzero(window == 2)
    return int(hits[0]) if len(hits) else max_h + 1


def quantile_from_pmf(row: NDArray[np.float64], q: float, H: int) -> int:
    c = np.cumsum(row)
    idx = int(np.searchsorted(c, q, side="left"))
    return min(idx, H + 1)


def evaluate_prediction(patient: Patient, pred: Prediction, H: int,
                        horizons: Sequence[int]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    records: List[Dict[str, float]] = []
    st_obs = patient.states_obs
    for d in range(patient.n_obs):
        row = pred.pmf[d]
        hit_obs = _next_flare_in_observed(st_obs, d)
        c = patient.n_obs - 1 - d
        if hit_obs is not None:
            nll = -math.log(max(row[hit_obs], EPS))
            censored = 0
            event_obs = hit_obs
        else:
            surv = max(1.0 - row[:c + 1].sum(), EPS)
            nll = -math.log(surv)
            censored = 1
            event_obs = np.nan

        entropy = -float(np.sum(row * np.log(np.clip(row, EPS, None))))
        lo = quantile_from_pmf(row, 0.05, H)
        hi = quantile_from_pmf(row, 0.95, H)
        width90 = float(hi - lo)
        full_hit = true_next_flare_full(patient, d, H)
        rec: Dict[str, float] = {
            "day": d,
            "state": int(st_obs[d]),
            "p_flare_now": float(pred.state_post[d, 2]),
            "nll": nll,
            "censored": censored,
            "event_obs": event_obs,
            "event_full": float(full_hit),
            "entropy": entropy,
            "width90": width90,
            "q05": lo,
            "q95": hi,
            "cover90": float(lo <= full_hit <= hi),
            "cond_mean": float(pred.cond_mean[d]),
            "uncond_mean": float(pred.uncond_mean[d]),
        }
        for h in horizons:
            rec[f"risk_{h}"] = float(row[:h + 1].sum())
            if d + h < patient.n_obs:
                y = float((st_obs[d:d + h + 1] == 2).any())
                rec[f"y_{h}"] = y
                rec[f"brier_{h}"] = (rec[f"risk_{h}"] - y) ** 2
            else:
                rec[f"y_{h}"] = np.nan
                rec[f"brier_{h}"] = np.nan
        records.append(rec)
    df = pd.DataFrame(records)
    summary = {
        "nll": float(df["nll"].mean()),
        "entropy": float(df["entropy"].mean()),
        "width90": float(df["width90"].mean()),
        "cover90": float(df["cover90"].mean()),
        "auroc": float(roc_auc_score((df["state"] == 2).astype(int), df["p_flare_now"]))
                 if (df["state"] == 2).nunique() > 1 else float("nan"),
    }
    for h in horizons:
        summary[f"brier_{h}"] = float(df[f"brier_{h}"].mean())
    summary["ibs4"] = float(np.mean([summary[f"brier_{h}"] for h in horizons]))
    return df, summary


def hierarchical_ci(values: pd.DataFrame, value_col: str, seed_col: str = "seed",
                    patient_col: str = "patient", B: int = 2000,
                    rng_seed: int = 2026) -> Tuple[float, float, float]:
    """Hierarchical seed-then-patient bootstrap for a patient-level metric."""
    del patient_col  # each input row is already one patient-level estimate
    rng = np.random.default_rng(rng_seed)
    seeds = np.array(sorted(values[seed_col].unique()))
    grouped = []
    for s in seeds:
        arr = values.loc[values[seed_col] == s, value_col].to_numpy(float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            grouped.append(arr)
    if not grouped:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(np.concatenate(grouped)))
    if B <= 0:
        return point, float("nan"), float("nan")

    # selected[b, j] identifies the source seed occupying bootstrap position j.
    selected = rng.integers(0, len(seeds), size=(B, len(seeds)))
    seed_position_means = np.empty_like(selected, dtype=float)
    # Vectorize patient resampling within all replicates that select the same
    # source seed in the same bootstrap position.
    for j in range(len(seeds)):
        for s_idx, arr in enumerate(grouped):
            rows = np.flatnonzero(selected[:, j] == s_idx)
            if not len(rows):
                continue
            idx = rng.integers(0, len(arr), size=(len(rows), len(arr)))
            seed_position_means[rows, j] = np.nanmean(arr[idx], axis=1)
    boots = np.nanmean(seed_position_means, axis=1)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def cluster_bootstrap_calibration(df: pd.DataFrame, risk_col: str, y_col: str,
                                  B: int = 300, seed: int = 123) -> Dict[str, float]:
    """Calibration intercept/slope with hierarchical seed-patient bootstrap.

    The point estimate is the pooled logistic calibration model
    ``logit P(Y=1) = intercept + slope*logit(risk)``. Confidence intervals
    resample simulation seeds first and then patients within every selected
    seed, retaining all repeated landmark rows for each selected patient.
    """
    use = df[["seed", "patient", risk_col, y_col]].copy()
    use = use[np.isfinite(use[y_col]) & np.isfinite(use[risk_col])]
    use[risk_col] = np.clip(use[risk_col].astype(float), 1e-5, 1 - 1e-5)
    use[y_col] = use[y_col].astype(float)
    if use.empty or use[y_col].nunique() < 2:
        return {k: float("nan") for k in [
            "intercept", "intercept_lo", "intercept_hi",
            "slope", "slope_lo", "slope_hi", "cil", "cil_lo", "cil_hi",
        ]}

    def fit_arrays(x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[float, float, float]:
        """Fast ridge-stabilized two-parameter logistic calibration fit.

        A direct Newton solve avoids the very slow convergence that generic
        GLM software can exhibit under near-separation in bootstrap samples.
        The ridge is numerically negligible for identified samples but makes
        every resample finite and deterministic.
        """
        z = np.log(x / (1.0 - x))
        X = np.column_stack([np.ones_like(z), z])
        beta = np.array([0.0, 1.0], dtype=float)
        ridge = np.diag([1e-8, 1e-6])
        for _ in range(40):
            eta = np.clip(X @ beta, -35.0, 35.0)
            prob = expit(eta)
            w = np.maximum(prob * (1.0 - prob), 1e-8)
            grad = X.T @ (y - prob) - ridge @ beta
            info = X.T @ (w[:, None] * X) + ridge
            try:
                step = np.linalg.solve(info, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(info) @ grad
            beta_new = beta + step
            if np.max(np.abs(step)) < 1e-8:
                beta = beta_new
                break
            beta = beta_new
        return float(beta[0]), float(beta[1]), float(np.mean(y - x))

    point_x = use[risk_col].to_numpy(float)
    point_y = use[y_col].to_numpy(float)
    intercept, slope, cil = fit_arrays(point_x, point_y)
    if B <= 0:
        return {
            "intercept": intercept, "intercept_lo": float("nan"), "intercept_hi": float("nan"),
            "slope": slope, "slope_lo": float("nan"), "slope_hi": float("nan"),
            "cil": cil, "cil_lo": float("nan"), "cil_hi": float("nan"),
        }

    # Store compact numpy groups to avoid repeatedly concatenating DataFrames.
    by_seed: Dict[int, List[Tuple[NDArray[np.float64], NDArray[np.float64]]]] = {}
    for s0, sg in use.groupby("seed", sort=False):
        groups: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for _, pg in sg.groupby("patient", sort=False):
            groups.append((pg[risk_col].to_numpy(float), pg[y_col].to_numpy(float)))
        by_seed[int(s0)] = groups

    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(by_seed), dtype=int)
    draws = np.empty((B, 3), dtype=float)
    successful = 0
    for _ in range(B):
        xs: List[NDArray[np.float64]] = []
        ys: List[NDArray[np.float64]] = []
        for s0 in rng.choice(seeds, size=len(seeds), replace=True):
            groups = by_seed[int(s0)]
            idx = rng.integers(0, len(groups), size=len(groups))
            for j in idx:
                xg, yg = groups[int(j)]
                xs.append(xg); ys.append(yg)
        try:
            draws[successful] = fit_arrays(np.concatenate(xs), np.concatenate(ys))
            successful += 1
        except Exception:
            continue
    required = min(B, max(5, B // 2))
    if successful < required:
        # Small smoke tests or nearly separated risks can make some bootstrap
        # samples unidentifiable.  Preserve the pooled point estimate and mark
        # the interval unavailable rather than hanging or fabricating precision.
        return {
            "intercept": intercept, "intercept_lo": float("nan"), "intercept_hi": float("nan"),
            "slope": slope, "slope_lo": float("nan"), "slope_hi": float("nan"),
            "cil": cil, "cil_lo": float("nan"), "cil_hi": float("nan"),
        }
    arr = draws[:successful]
    ci = np.quantile(arr, [0.025, 0.975], axis=0)
    return {
        "intercept": intercept, "intercept_lo": float(ci[0, 0]), "intercept_hi": float(ci[1, 0]),
        "slope": slope, "slope_lo": float(ci[0, 1]), "slope_hi": float(ci[1, 1]),
        "cil": cil, "cil_lo": float(ci[0, 2]), "cil_hi": float(ci[1, 2]),
    }


def hmm_params_to_dict(params: HMMParams) -> Dict[str, object]:
    return {
        "pi": params.pi.tolist(), "P": params.P.tolist(),
        "wear_mu": params.wear_mu.tolist(), "wear_sd": params.wear_sd.tolist(),
        "lab_logmu": params.lab_logmu.tolist(), "lab_logsd": params.lab_logsd.tolist(),
        "lam": None if params.lam is None else params.lam.tolist(),
        "model_type": params.model_type, "loglik": params.loglik,
        "n_iter": params.n_iter,
        "pm_wear_mu": None if params.pm_wear_mu is None else params.pm_wear_mu.tolist(),
        "pm_wear_sd": None if params.pm_wear_sd is None else params.pm_wear_sd.tolist(),
        "em_loglik_trace": params.em_loglik_trace,
        "start_diagnostics": params.start_diagnostics,
    }


def hmm_params_from_dict(obj: Dict[str, object]) -> HMMParams:
    return HMMParams(
        pi=np.asarray(obj["pi"], dtype=float),
        P=np.asarray(obj["P"], dtype=float),
        wear_mu=np.asarray(obj["wear_mu"], dtype=float),
        wear_sd=np.asarray(obj["wear_sd"], dtype=float),
        lab_logmu=np.asarray(obj["lab_logmu"], dtype=float),
        lab_logsd=np.asarray(obj["lab_logsd"], dtype=float),
        lam=None if obj.get("lam") is None else np.asarray(obj["lam"], dtype=float),
        model_type=str(obj["model_type"]),
        pm_wear_mu=None if obj.get("pm_wear_mu") is None else np.asarray(obj["pm_wear_mu"], dtype=float),
        pm_wear_sd=None if obj.get("pm_wear_sd") is None else np.asarray(obj["pm_wear_sd"], dtype=float),
        loglik=float(obj.get("loglik", float("nan"))),
        n_iter=int(obj.get("n_iter", 0)),
        em_loglik_trace=obj.get("em_loglik_trace"),
        start_diagnostics=obj.get("start_diagnostics"),
    )


def load_params(path: Path) -> HMMParams:
    return hmm_params_from_dict(json.loads(path.read_text()))


def save_params(params: HMMParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hmm_params_to_dict(params), indent=2))


def save_hmm_ensemble(ensemble: Sequence[HMMParams], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_members": len(ensemble),
        "members": [hmm_params_to_dict(x) for x in ensemble],
    }, indent=2))


def hazard_model_to_dict(model: HazardModel) -> Dict[str, object]:
    return {
        "max_train_horizon": int(model.max_train_horizon),
        "initial_loglab": model.initial_loglab.tolist(),
        "fit_seed": int(model.fit_seed),
        "training_patient_ids": [int(x) for x in model.training_patient_ids],
        "bootstrap_patient_ids": None if model.bootstrap_patient_ids is None else [int(x) for x in model.bootstrap_patient_ids],
        "feature_names": event_history_feature_names(),
        "future_time_basis_names": ["log1p_horizon", "sqrt_horizon_div_10", "horizon_scaled"],
        "scaler": {
            "mean": model.scaler.mean_.tolist(), "scale": model.scaler.scale_.tolist(),
            "var": model.scaler.var_.tolist(), "n_features_in": int(model.scaler.n_features_in_),
            "n_samples_seen": np.asarray(model.scaler.n_samples_seen_).tolist(),
        },
        "current_model": {
            "coef": model.current_model.coef_.tolist(), "intercept": model.current_model.intercept_.tolist(),
            "classes": model.current_model.classes_.tolist(), "n_features_in": int(model.current_model.n_features_in_),
            "C": float(model.current_model.C), "n_iter": model.current_model.n_iter_.tolist(),
        },
        "future_model": {
            "coef": model.future_model.coef_.tolist(), "intercept": model.future_model.intercept_.tolist(),
            "classes": model.future_model.classes_.tolist(), "n_features_in": int(model.future_model.n_features_in_),
            "C": float(model.future_model.C), "n_iter": model.future_model.n_iter_.tolist(),
        },
    }


def save_hazard_model(model: HazardModel, path: Path) -> None:
    """Serialize a hazard model without binding the archive to ``__main__``.

    Worker seeds execute this file as a script, so pickling the dataclass object
    directly would record its class as ``__main__.HazardModel`` and make the
    archive impossible to load from an independent verifier.  The portable
    pickle therefore stores the same complete plain-data payload as the JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = hazard_model_to_dict(model)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    with path.with_suffix(".pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def save_hazard_ensemble(ensemble: Sequence[HazardModel], path: Path) -> None:
    """Serialize all hazard members as a module-independent data payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_members": len(ensemble),
        "members": [hazard_model_to_dict(x) for x in ensemble],
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    with path.with_suffix(".pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_environment(path: Path) -> None:
    import contextlib, io, scipy, sklearn
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.__config__.show()
    path.write_text(json.dumps({
        "python_version": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "implementation": platform.python_implementation(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "thread_environment": {k: os.environ.get(k) for k in
            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "numpy_configuration": buf.getvalue(),
    }, indent=2))


def write_archive_indices(out_dir: Path) -> None:
    dist_rows: List[Dict[str, object]] = []
    for path in sorted((out_dir / "prediction_distributions").glob("seed_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            max_err = float(np.max(np.abs(z["pmf"].sum(axis=-1) - 1.0)))
            pmf_shape = list(z["pmf"].shape)
            state_shape = list(z["state_posterior"].shape)
        dist_rows.append({"file": str(path.relative_to(out_dir)), "sha256": sha256_file(path),
                          "size_bytes": path.stat().st_size, "pmf_shape": json.dumps(pmf_shape),
                          "state_posterior_shape": json.dumps(state_shape), "max_pmf_sum_error": max_err})
    pd.DataFrame(dist_rows).to_csv(out_dir / "prediction_distribution_index.csv", index=False)
    art_rows: List[Dict[str, object]] = []
    for path in sorted((out_dir / "model_artifacts").rglob("*")):
        if path.is_file():
            art_rows.append({"file": str(path.relative_to(out_dir)), "sha256": sha256_file(path),
                             "size_bytes": path.stat().st_size})
    pd.DataFrame(art_rows).to_csv(out_dir / "model_artifact_index.csv", index=False)


def run_seed(seed_index: int, cfg: argparse.Namespace, out_dir: Path,
             scenario: str = "well_specified") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, HMMParams]]:
    seed = cfg.base_seed + 1009 * seed_index + (0 if scenario == "well_specified" else 500_000)
    pats = simulate_patients(cfg.n_total, cfg.n_days, cfg.future_days, seed, scenario=scenario)
    train = pats[:cfg.n_train]
    val = pats[cfg.n_train:cfg.n_train + cfg.n_val]
    test = pats[cfg.n_train + cfg.n_val:]
    del val

    models: Dict[str, object] = {}
    fitted: Dict[str, HMMParams] = {}
    for name, mtype in [(MODEL_DRAW, "endogenous"), (MODEL_NO_DRAW, "no_lambda"),
                        (MODEL_DRAW_STRATIFIED, "pattern_mixture")]:
        base = fit_hmm(train, mtype, n_starts=cfg.n_starts,
                       max_iter=cfg.max_em_iter, seed=seed + len(fitted) * 101)
        ens = approximate_parameter_ensemble(train, base, cfg.n_param_draws,
                                             seed=seed + 7000 + len(fitted) * 151)
        models[name] = ens
        fitted[name] = base

    hazard_base = fit_hazard_model(train, max_horizon=cfg.hazard_horizon, seed=seed + 91)
    hazard_ensemble = bootstrap_hazard_ensemble(train, hazard_base, cfg.n_hazard_draws, seed + 9091)
    models[MODEL_EVENT_HISTORY] = hazard_ensemble

    artifact_dir = out_dir / "model_artifacts" / f"seed_{seed_index:02d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for model_name, slug in {MODEL_DRAW: "hmm_draw", MODEL_NO_DRAW: "hmm_no_draw",
                             MODEL_DRAW_STRATIFIED: "draw_stratified_hmm"}.items():
        ensemble = models[model_name]
        assert isinstance(ensemble, list)
        save_params(fitted[model_name], artifact_dir / f"{slug}_base.json")
        save_hmm_ensemble(ensemble, artifact_dir / f"{slug}_ensemble.json")
    save_hazard_model(hazard_base, artifact_dir / "event_history_base")
    save_hazard_ensemble(hazard_ensemble, artifact_dir / "event_history_ensemble")

    all_landmarks: List[pd.DataFrame] = []
    patient_summaries: List[Dict[str, float]] = []
    n_test = len(test); n_models = len(MODEL_ORDER); T = cfg.n_days
    pmf_archive = np.empty((n_models, n_test, T, cfg.pmf_horizon + 2), dtype=np.float32)
    state_archive = np.empty((n_models, n_test, T, K), dtype=np.float32)
    cond_archive = np.empty((n_models, n_test, T), dtype=np.float32)
    uncond_archive = np.empty((n_models, n_test, T), dtype=np.float32)
    true_state_archive = np.empty((n_test, T), dtype=np.int8)
    full_event_archive = np.empty((n_test, T), dtype=np.int16)
    for pidx, p in enumerate(test):
        true_state_archive[pidx] = p.states_obs.astype(np.int8)
        full_event_archive[pidx] = np.array(
            [true_next_flare_full(p, d, cfg.pmf_horizon) for d in range(T)], dtype=np.int16)
        for midx, model_name in enumerate(MODEL_ORDER):
            ensemble = models[model_name]
            pred = (predict_hazard(p, ensemble, cfg.pmf_horizon)
                    if model_name == MODEL_EVENT_HISTORY
                    else predict_hmm(p, ensemble, cfg.pmf_horizon))
            pmf_archive[midx, pidx] = pred.pmf.astype(np.float32)
            state_archive[midx, pidx] = pred.state_post.astype(np.float32)
            cond_archive[midx, pidx] = pred.cond_mean.astype(np.float32)
            uncond_archive[midx, pidx] = pred.uncond_mean.astype(np.float32)
            df, sm = evaluate_prediction(p, pred, cfg.pmf_horizon, cfg.horizons)
            df["seed"] = seed_index; df["patient"] = pidx
            df["cluster"] = f"{seed_index}:{pidx}"; df["model"] = model_name
            all_landmarks.append(df)
            sm.update({"seed": seed_index, "patient": pidx, "model": model_name})
            patient_summaries.append(sm)

    dist_dir = out_dir / "prediction_distributions"; dist_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dist_dir / f"seed_{seed_index:02d}.npz", pmf=pmf_archive,
        state_posterior=state_archive, conditional_mean=cond_archive,
        unconditional_mean=uncond_archive, true_state=true_state_archive,
        event_full=full_event_archive, model_names=np.asarray(MODEL_ORDER),
        patient_local_index=np.arange(n_test, dtype=np.int16),
        patient_simulator_id=np.asarray([p.patient_id for p in test], dtype=np.int16),
        outcome_categories=np.asarray([str(i) for i in range(cfg.pmf_horizon + 1)] + [f">{cfg.pmf_horizon}"]),
    )
    return pd.concat(all_landmarks, ignore_index=True), pd.DataFrame(patient_summaries), fitted


def make_figures(landmarks: pd.DataFrame, patient_metrics: pd.DataFrame,
                 exemplar: Tuple[Patient, Dict[str, Prediction]],
                 fitted_params: List[HMMParams], out_dir: Path,
                 horizons: Sequence[int]) -> None:
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Figure 2: updated interface, distinguishing conditional and unconditional means.
    p, preds = exemplar
    pred = preds[MODEL_DRAW]
    t = np.arange(p.n_obs)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax1 = fig.add_axes([0.10, 0.68, 0.84, 0.25])
    ax1.plot(t, pred.state_post[:, 0], label="Pr(Remission)")
    ax1.plot(t, pred.state_post[:, 1], label="Pr(Mild)")
    ax1.plot(t, pred.state_post[:, 2], label="Pr(Flare)")
    flare_days = np.flatnonzero((p.states_obs == 2) & np.r_[True, p.states_obs[:-1] != 2])
    for d in flare_days:
        ax1.axvline(d, linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Posterior probability")
    ax1.set_xlim(0, p.n_obs - 1); ax1.set_ylim(0, 1.02)
    ax1.legend(ncol=3, fontsize=7, loc="upper center")
    ax1.set_title("Latent-state filter posterior")

    ax2 = fig.add_axes([0.10, 0.34, 0.84, 0.25])
    ax2.plot(t, pred.cond_mean, label="Conditional on not currently in flare")
    ax2.plot(t, pred.uncond_mean, label="Unconditional (includes mass at day 0)")
    for d in flare_days:
        ax2.axvline(d, linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Expected days to flare")
    ax2.set_xlim(0, p.n_obs - 1)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("Two mathematically distinct time-to-flare means")

    ax3 = fig.add_axes([0.10, 0.12, 0.84, 0.10])
    draw_days = np.flatnonzero(p.draw)
    ax3.vlines(draw_days, 0, 1, linewidth=0.8)
    ax3.set_ylim(0, 1); ax3.set_yticks([]); ax3.set_xlim(0, p.n_obs - 1)
    ax3.set_xlabel("Study day"); ax3.set_title("Laboratory panel drawn")
    fig.savefig(figdir / "time_to_flare_interface.pdf", bbox_inches="tight")
    fig.savefig(figdir / "time_to_flare_interface.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: 30-day calibration by deciles.
    h = 30 if 30 in horizons else horizons[0]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for model, g in landmarks.dropna(subset=[f"y_{h}"]).groupby("model"):
        gg = g.copy()
        # qcut may drop duplicate bins for extreme predictions.
        gg["bin"] = pd.qcut(gg[f"risk_{h}"], q=10, duplicates="drop")
        cal = gg.groupby("bin", observed=True).agg(pred=(f"risk_{h}", "mean"),
                                                    obs=(f"y_{h}", "mean"),
                                                    n=(f"y_{h}", "size"))
        ax.plot(cal["pred"], cal["obs"], marker="o", linewidth=1.2, label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Ideal")
    ax.set_xlabel(f"Mean predicted Pr(flare within {h} days)")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title(f"Fixed-horizon calibration with complete {h}-day follow-up")
    fig.tight_layout()
    fig.savefig(figdir / "calibration_30d.pdf")
    fig.savefig(figdir / "calibration_30d.png", dpi=220)
    plt.close(fig)

    # Figure 4: proper scores and sharpness (patient-level means with seed SD).
    metric_labels = [("nll", "Censored negative log score"),
                     ("ibs4", "Mean Brier score"),
                     ("entropy", "Predictive entropy")]
    for metric, label in metric_labels:
        s = patient_metrics.groupby(["seed", "model"])[metric].mean().reset_index()
        agg = s.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(label + " (lower is better)")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(figdir / f"score_{metric}.pdf")
        fig.savefig(figdir / f"score_{metric}.png", dpi=220)
        plt.close(fig)

    # Figure 5: long-horizon equal-tailed interval coverage curve.
    levels = np.arange(0.1, 1.0, 0.1)
    # Reconstruct coverage from stored full event times and model-specific PMFs
    # is expensive after aggregation; manuscript focuses on 90% coverage and
    # 30-day calibration. This plot therefore shows cluster-bootstrap 90%
    # coverage by model as a compact reliability display.
    cov = patient_metrics.groupby(["seed", "model"])["cover90"].mean().reset_index()
    agg = cov.groupby("model")["cover90"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(agg))
    ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4)
    ax.axhline(0.90, linestyle="--", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Empirical coverage of nominal 90% interval")
    ax.set_ylim(0.65, 1.0)
    ax.set_title("Nominal 90% interval coverage on latent continuation")
    fig.tight_layout()
    fig.savefig(figdir / "coverage90.pdf")
    fig.savefig(figdir / "coverage90.png", dpi=220)
    plt.close(fig)

    # Figure 6: parameter recovery for lambda and hitting-time means.
    lam_est = np.vstack([p.lam for p in fitted_params if p.lam is not None])
    means = []
    for pfit in fitted_params:
        Q, r, N = hitting_components(pfit.P)
        means.append(N @ np.ones(2))
    means = np.vstack(means)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(K)
    ax.errorbar(x - 0.05, lam_est.mean(0), yerr=lam_est.std(0), fmt="o", capsize=4,
                label="Estimated draw rate")
    ax.scatter(x + 0.05, TRUE_LAMBDA, marker="x", s=55, label="Generating draw rate")
    ax.set_xticks(x); ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("Laboratory-draw probability")
    ax.set_title("Latent-state EM recovery of endogenous sampling rates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "lambda_recovery.pdf")
    fig.savefig(figdir / "lambda_recovery.png", dpi=220)
    plt.close(fig)

    true_N = np.linalg.inv(np.eye(2) - TRUE_P[:2, :2])
    true_means = true_N @ np.ones(2)
    pd.DataFrame({
        "state": ["Remission", "Mild"],
        "true_mean": true_means,
        "estimated_mean": means.mean(0),
        "estimated_sd": means.std(0),
    }).to_csv(out_dir / "tables" / "hitting_time_recovery.csv", index=False)


def aggregate_results(landmarks: pd.DataFrame, patient_metrics: pd.DataFrame,
                      out_dir: Path, horizons: Sequence[int],
                      performance_bootstrap: int = 2000,
                      calibration_bootstrap: int = 500) -> Dict[str, object]:
    tables = out_dir / "tables"; tables.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    metrics = ["nll", "ibs4", "entropy", "width90", "cover90", "auroc"]
    for model in patient_metrics["model"].unique():
        g = patient_metrics[patient_metrics["model"] == model]
        for metric in metrics:
            point, lo, hi = hierarchical_ci(g, metric, B=performance_bootstrap, rng_seed=100 + metrics.index(metric))
            summary_rows.append({"model": model, "metric": metric,
                                 "estimate": point, "ci_low": lo, "ci_high": hi})
        for h in horizons:
            metric = f"brier_{h}"
            point, lo, hi = hierarchical_ci(g, metric, B=performance_bootstrap, rng_seed=200 + h)
            summary_rows.append({"model": model, "metric": metric,
                                 "estimate": point, "ci_low": lo, "ci_high": hi})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "performance_summary.csv", index=False)

    # Calibration summaries at 30 days.
    calibration: Dict[str, Dict[str, float]] = {}
    h = 30 if 30 in horizons else horizons[0]
    for model, g in landmarks.groupby("model"):
        calibration[model] = cluster_bootstrap_calibration(
            g, f"risk_{h}", f"y_{h}", B=calibration_bootstrap,
            seed=301 + sum(ord(ch) for ch in model),
        )
    (tables / "calibration_30d.json").write_text(json.dumps(calibration, indent=2))

    # Paired differences relative to proposed model.
    base_name = MODEL_DRAW
    paired_rows = []
    base = patient_metrics[patient_metrics.model == base_name]
    for model in patient_metrics.model.unique():
        if model == base_name:
            continue
        other = patient_metrics[patient_metrics.model == model]
        merged = base.merge(other, on=["seed", "patient"], suffixes=("_base", "_other"))
        for metric in ["nll", "ibs4", "entropy", "width90"]:
            diff_col = f"diff_{metric}"
            # Positive means comparator is worse because comparator - proposed.
            merged[diff_col] = merged[f"{metric}_other"] - merged[f"{metric}_base"]
            p, lo, hi = hierarchical_ci(merged.rename(columns={diff_col: "value"}), "value",
                                        B=performance_bootstrap, rng_seed=444 + len(paired_rows))
            paired_rows.append({"comparator": model, "metric": metric,
                                "comparator_minus_proposed": p,
                                "ci_low": lo, "ci_high": hi})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(tables / "paired_differences.csv", index=False)

    # Compact wide table for LaTeX.
    wide_rows = []
    for model in patient_metrics.model.unique():
        row: Dict[str, object] = {"model": model}
        for metric in ["nll", "ibs4", "cover90", "entropy", "auroc"]:
            r = summary[(summary.model == model) & (summary.metric == metric)].iloc[0]
            row[metric] = r.estimate
            row[metric + "_lo"] = r.ci_low
            row[metric + "_hi"] = r.ci_high
        cal = calibration[model]
        row["cal_slope"] = cal["slope"]
        row["cal_slope_lo"] = cal["slope_lo"]
        row["cal_slope_hi"] = cal["slope_hi"]
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(tables / "main_results_wide.csv", index=False)

    result_obj = {
        "performance": summary.to_dict(orient="records"),
        "calibration_30d": calibration,
        "paired_differences": paired.to_dict(orient="records"),
    }
    (out_dir / "results.json").write_text(json.dumps(result_obj, indent=2))
    return result_obj



def aggregate_noncurrent_flare_sensitivity(
    landmarks: pd.DataFrame, out_dir: Path, horizons: Sequence[int],
    performance_bootstrap: int = 2000, calibration_bootstrap: int = 500,
) -> Dict[str, object]:
    tables = out_dir / "tables"; tables.mkdir(parents=True, exist_ok=True)
    base_rows = landmarks[landmarks["model"] == MODEL_DRAW]
    current_fraction = float((base_rows["state"] == 2).mean())
    use = landmarks[landmarks["state"] != 2].copy()
    agg_spec: Dict[str, str] = {"nll": "mean", "entropy": "mean", "width90": "mean", "cover90": "mean"}
    for h in horizons: agg_spec[f"brier_{h}"] = "mean"
    patient = use.groupby(["seed", "patient", "model"], as_index=False).agg(agg_spec)
    patient["ibs4"] = patient[[f"brier_{h}" for h in horizons]].mean(axis=1)
    patient.to_csv(tables / "noncurrent_flare_patient_metrics.csv", index=False)
    rows: List[Dict[str, object]] = []
    for model in MODEL_ORDER:
        g = patient[patient.model == model]
        for metric, rs in [("nll", 811), ("ibs4", 812)]:
            point, lo, hi = hierarchical_ci(g, metric, B=performance_bootstrap, rng_seed=rs)
            rows.append({"model": model, "metric": metric, "estimate": point, "ci_low": lo, "ci_high": hi})
    perf = pd.DataFrame(rows); perf.to_csv(tables / "noncurrent_flare_performance.csv", index=False)
    proposed = patient[patient.model == MODEL_DRAW]
    paired_rows: List[Dict[str, object]] = []
    for model in MODEL_ORDER:
        if model == MODEL_DRAW: continue
        other = patient[patient.model == model]
        merged = proposed.merge(other, on=["seed", "patient"], suffixes=("_base", "_other"))
        for metric, rs in [("nll", 821), ("ibs4", 822)]:
            merged["value"] = merged[f"{metric}_other"] - merged[f"{metric}_base"]
            point, lo, hi = hierarchical_ci(merged, "value", B=performance_bootstrap, rng_seed=rs)
            paired_rows.append({"comparator": model, "metric": metric,
                                "comparator_minus_proposed": point, "ci_low": lo, "ci_high": hi})
    paired = pd.DataFrame(paired_rows); paired.to_csv(tables / "noncurrent_flare_paired_differences.csv", index=False)
    h = 30 if 30 in horizons else horizons[0]
    calibration = {model: cluster_bootstrap_calibration(g, f"risk_{h}", f"y_{h}",
                    B=calibration_bootstrap, seed=901 + sum(ord(ch) for ch in model))
                   for model, g in use.groupby("model")}
    (tables / "noncurrent_flare_calibration_30d.json").write_text(json.dumps(calibration, indent=2))
    obj = {"current_flare_landmark_fraction": current_fraction,
           "n_noncurrent_landmark_rows": int(len(use)),
           "performance": perf.to_dict(orient="records"),
           "paired_differences": paired.to_dict(orient="records"),
           "calibration_30d": calibration}
    (tables / "noncurrent_flare_sensitivity.json").write_text(json.dumps(obj, indent=2))
    return obj

def select_exemplar(cfg: argparse.Namespace, fitted: Dict[str, HMMParams]) -> Tuple[Patient, Dict[str, Prediction]]:
    pats = simulate_patients(30, cfg.n_days, cfg.future_days,
                             cfg.base_seed + 987654, scenario="well_specified")
    chosen = pats[0]
    for p in pats:
        onset = np.flatnonzero((p.states_obs == 2) & np.r_[True, p.states_obs[:-1] != 2])
        if len(onset) and 35 <= onset[0] <= 90:
            chosen = p
            break
    preds = {
        name: predict_hmm(chosen, [param], cfg.pmf_horizon)
        for name, param in fitted.items()
    }
    return chosen, preds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference_outputs")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-total", type=int, default=120)
    ap.add_argument("--n-train", type=int, default=65)
    ap.add_argument("--n-val", type=int, default=15)
    ap.add_argument("--n-days", type=int, default=120)
    ap.add_argument("--future-days", type=int, default=240)
    ap.add_argument("--pmf-horizon", type=int, default=180)
    ap.add_argument("--hazard-horizon", type=int, default=120)
    ap.add_argument("--n-param-draws", type=int, default=16)
    ap.add_argument("--n-hazard-draws", type=int, default=16)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--max-em-iter", type=int, default=35)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--performance-bootstrap", type=int, default=2000)
    ap.add_argument("--calibration-bootstrap", type=int, default=500)
    ap.add_argument("--base-seed", type=int, default=1729)
    ap.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(); args.horizons = DEFAULT_HORIZONS
    if args.quick:
        args.n_seeds = 2; args.n_total = 60; args.n_train = 35; args.n_val = 5
        args.n_param_draws = 3; args.n_hazard_draws = 3; args.n_starts = 1
        args.n_jobs = 2; args.max_em_iter = 20
        args.performance_bootstrap = 40; args.calibration_bootstrap = 25
    if args.n_jobs < 1: ap.error("n_jobs must be at least 1")
    if args.n_train + args.n_val >= args.n_total: ap.error("n_train + n_val must be smaller than n_total")
    if args.future_days < args.pmf_horizon: ap.error("future_days must be at least pmf_horizon")
    return args


def main() -> None:
    cfg = parse_args(); out_dir = Path(cfg.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True); (out_dir / "figures").mkdir(exist_ok=True)
    # A worker subprocess handles exactly one simulation seed and writes its
    # compact return objects to disk.  Isolating seeds in fresh interpreters
    # avoids cross-seed numerical-library state and process-pool shutdown
    # deadlocks while preserving deterministic parallelism.
    if cfg.worker_seed is not None:
        seed_index = int(cfg.worker_seed)
        lm, pm, _ = run_seed(seed_index, cfg, out_dir)
        seed_dir = out_dir / "seed_results"; seed_dir.mkdir(parents=True, exist_ok=True)
        lm.to_csv(seed_dir / f"seed_{seed_index:02d}_landmarks.csv.gz", index=False, compression="gzip")
        pm.to_csv(seed_dir / f"seed_{seed_index:02d}_patient_metrics.csv", index=False)
        print(f"Worker completed seed {seed_index + 1}/{cfg.n_seeds}", flush=True)
        os._exit(0)  # all artifacts are closed; bypass library shutdown deadlocks

    # Only the parent writes release-wide metadata; workers share the output
    # directory but never race on these files.
    config_obj = vars(cfg).copy(); config_obj.pop("worker_seed", None)
    (out_dir / "config.json").write_text(json.dumps(config_obj, indent=2, default=list))
    capture_environment(out_dir / "environment_manifest.json")
    (out_dir / "source_provenance.json").write_text(json.dumps({
        "pipeline_file": Path(__file__).name, "pipeline_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap": {"performance_replicates": cfg.performance_bootstrap,
                      "calibration_replicates": cfg.calibration_bootstrap,
                      "hierarchy": "simulation seed, then patient within selected seed"},
        "model_uncertainty": {"hmm_parameter_draws_excluding_base": cfg.n_param_draws,
                              "hazard_patient_bootstrap_refits_excluding_base": cfg.n_hazard_draws,
                              "em_initializations_per_hmm": cfg.n_starts},
        "seed_execution": "one fresh Python subprocess per simulation seed",
    }, indent=2))

    seed_dir = out_dir / "seed_results"; seed_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    def worker_command(seed_index: int) -> List[str]:
        return [
            sys.executable, str(script), "--out", str(out_dir),
            "--worker-seed", str(seed_index), "--n-seeds", str(cfg.n_seeds),
            "--n-total", str(cfg.n_total), "--n-train", str(cfg.n_train),
            "--n-val", str(cfg.n_val), "--n-days", str(cfg.n_days),
            "--future-days", str(cfg.future_days), "--pmf-horizon", str(cfg.pmf_horizon),
            "--hazard-horizon", str(cfg.hazard_horizon),
            "--n-param-draws", str(cfg.n_param_draws),
            "--n-hazard-draws", str(cfg.n_hazard_draws),
            "--n-starts", str(cfg.n_starts), "--max-em-iter", str(cfg.max_em_iter),
            "--n-jobs", "1", "--performance-bootstrap", str(cfg.performance_bootstrap),
            "--calibration-bootstrap", str(cfg.calibration_bootstrap),
            "--base-seed", str(cfg.base_seed),
        ]

    active: Dict[int, subprocess.Popen[bytes]] = {}
    next_seed = 0
    while next_seed < cfg.n_seeds or active:
        while next_seed < cfg.n_seeds and len(active) < min(cfg.n_jobs, cfg.n_seeds):
            active[next_seed] = subprocess.Popen(worker_command(next_seed))
            next_seed += 1
        completed: List[int] = []
        for i, proc in active.items():
            rc = proc.poll()
            if rc is None:
                continue
            if rc != 0:
                raise subprocess.CalledProcessError(rc, worker_command(i))
            print(f"Collected seed {i + 1}/{cfg.n_seeds}", flush=True)
            completed.append(i)
        for i in completed:
            del active[i]
        if active:
            time.sleep(0.1)

    subprocess.run([sys.executable, str(Path(__file__).with_name("finalize_outputs.py")),
                    "--outputs", str(out_dir)], check=True)
    print(f"Complete. Outputs written to {out_dir}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
