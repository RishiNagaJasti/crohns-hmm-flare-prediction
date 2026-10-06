#!/usr/bin/env python3
"""Fill the IEEE manuscript template from generated machine-readable outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import crohns_hmm_pipeline as crp

MODEL_DRAW = "HMM + draw model"
MODEL_NO_DRAW = "HMM without draw model"
MODEL_STRAT = "Draw-stratified-emission HMM"
MODEL_HAZ = "Discrete-time event-history"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--outputs", type=Path, default=Path("reference_outputs"))
    ap.add_argument("--template", type=Path, default=Path("manuscript_template.tex"))
    ap.add_argument("--output", type=Path, default=Path("Crohns_HMM_Time_to_Flare_Study.tex"))
    return ap.parse_args()


def main() -> None:
    args = parse_args(); root = args.root.resolve()
    outputs = args.outputs if args.outputs.is_absolute() else root / args.outputs
    template = args.template if args.template.is_absolute() else root / args.template
    output = args.output if args.output.is_absolute() else root / args.output

    tpl = template.read_text()
    config = json.loads((outputs / "config.json").read_text())
    release_config = json.loads((root / "release_config.json").read_text())
    wide = pd.read_csv(outputs / "tables" / "main_results_wide.csv").set_index("model")
    paired = pd.read_csv(outputs / "tables" / "paired_differences.csv")
    params = pd.read_csv(outputs / "tables" / "parameter_recovery.csv")
    hit = pd.read_csv(outputs / "tables" / "hitting_time_recovery.csv").set_index("state")
    noncur_perf = pd.read_csv(outputs / "tables" / "noncurrent_flare_performance.csv")
    noncur_pair = pd.read_csv(outputs / "tables" / "noncurrent_flare_paired_differences.csv")
    noncur_obj = json.loads((outputs / "tables" / "noncurrent_flare_sensitivity.json").read_text())

    def num(x: float, d: int) -> str:
        """Fixed-precision number with a typographic minus sign."""
        text = f"{x:.{d}f}"
        return "$-$" + text[1:] if text.startswith("-") else text

    def rng(lo: float, hi: float, d: int = 3) -> str:
        # A negative lower limit is written with "to" so that the minus sign is not
        # read as part of a dash.
        sep = " to " if lo < 0 else "--"
        return f"{num(lo, d)}{sep}{num(hi, d)}"

    def rng_to(lo: float, hi: float, d: int) -> str:
        return f"{num(lo, d)} to {num(hi, d)}"

    noncur_cal = json.loads((outputs / "tables" / "noncurrent_flare_calibration_30d.json").read_text())

    def cal_slope_text(model: str) -> str:
        c = noncur_cal[model]
        return f"{c['slope']:.3f} ({rng(c['slope_lo'], c['slope_hi'], 3)})"

    def row(name: str) -> str:
        r = wide.loc[name]
        return (
            f"{r.nll:.3f} ({rng(r.nll_lo, r.nll_hi, 3)}) & "
            f"{r.ibs4:.4f} ({rng(r.ibs4_lo, r.ibs4_hi, 4)}) & "
            f"{r.cover90:.3f} ({rng(r.cover90_lo, r.cover90_hi, 3)}) & "
            f"{r.entropy:.3f} ({rng(r.entropy_lo, r.entropy_hi, 3)}) & "
            f"{r.cal_slope:.3f} ({rng(r.cal_slope_lo, r.cal_slope_hi, 3)})"
        )

    def pair(name: str, metric: str) -> pd.Series:
        return paired[(paired.comparator == name) & (paired.metric == metric)].iloc[0]

    def signed(x: float, d: int) -> str:
        text = f"{x:+.{d}f}"
        return "$-$" + text[1:] if text.startswith("-") else text

    def paircell(name: str) -> str:
        a = pair(name, "nll"); b = pair(name, "ibs4")
        return (
            f"{signed(a.comparator_minus_proposed, 4)} ({rng_to(a.ci_low, a.ci_high, 4)}) & "
            f"{signed(b.comparator_minus_proposed, 5)} ({rng_to(b.ci_low, b.ci_high, 5)})"
        )

    def noncur(model: str, metric: str) -> pd.Series:
        return noncur_perf[(noncur_perf.model == model) & (noncur_perf.metric == metric)].iloc[0]

    def noncur_diff(model: str, metric: str) -> pd.Series:
        return noncur_pair[(noncur_pair.comparator == model) & (noncur_pair.metric == metric)].iloc[0]

    def noncur_row(model: str) -> str:
        a = noncur(model, "nll"); b = noncur(model, "ibs4")
        return f"{a.estimate:.3f} ({rng(a.ci_low, a.ci_high, 3)}) & {b.estimate:.4f} ({rng(b.ci_low, b.ci_high, 4)})"

    prop = wide.loc[MODEL_DRAW]
    nol = pair(MODEL_NO_DRAW, "nll"); nolb = pair(MODEL_NO_DRAW, "ibs4")
    ncd = noncur_diff(MODEL_NO_DRAW, "nll"); ncdb = noncur_diff(MODEL_NO_DRAW, "ibs4")
    lam_parts = [f"${params[col].mean():.3f}\\pm{params[col].std(ddof=0):.3f}$"
                 for col in ["lambda_Remission", "lambda_Mild", "lambda_Flare"]]
    hit_parts = [f"${hit.loc[state, 'estimated_mean']:.2f}\\pm{hit.loc[state, 'estimated_sd']:.2f}$"
                 for state in ["Remission", "Mild"]]

    # ---- quantities computed from the archived landmark, patient and model files ----
    lm_path = outputs / "landmark_predictions.parquet"
    landmarks = (pd.read_parquet(lm_path) if lm_path.exists()
                 else pd.read_csv(outputs / "landmark_predictions.csv.gz"))
    patients = pd.read_csv(outputs / "patient_metrics.csv")
    prop_lm = landmarks[landmarks.model == MODEL_DRAW]
    n_test_landmarks = len(prop_lm)
    n_current = int((prop_lm.state == 2).sum())
    noncur_lm = prop_lm[prop_lm.state != 2]
    per_patient_noncur = noncur_lm.groupby(["seed", "patient"]).size()
    pooled_auroc = roc_auc_score((prop_lm.state == 2).astype(int), prop_lm.p_flare_now)
    n_auroc_patients = int(patients[patients.model == MODEL_DRAW].auroc.notna().sum())
    hmm_models = [MODEL_DRAW, MODEL_NO_DRAW, MODEL_STRAT]
    risk30 = landmarks[landmarks.model.isin(hmm_models) & landmarks.y_30.notna()].risk_30
    event30 = prop_lm[prop_lm.y_30.notna()].y_30.mean()

    base = patients[patients.model == MODEL_DRAW].set_index(["seed", "patient"])
    other = patients[patients.model == MODEL_NO_DRAW].set_index(["seed", "patient"])
    seed_diff = (other[["nll", "ibs4"]] - base[["nll", "ibs4"]]).groupby(level="seed").mean()

    def seed_t_interval(values: pd.Series) -> tuple:
        n = len(values)
        return stats.t.interval(0.95, n - 1, loc=values.mean(), scale=values.std(ddof=1) / np.sqrt(n))

    t_nll = seed_t_interval(seed_diff["nll"])
    t_ibs = seed_t_interval(seed_diff["ibs4"])
    n_seeds_favor = int((seed_diff["nll"] > 0).sum())

    coincident = 0; n_fits = 0; max_spread = 0.0; em_iters = []
    em_min_incr = np.inf; em_ratio_max = 0.0; em_last_gain_max = 0.0; em_remain_max = 0.0; em_abs_ll = []
    for i in range(config["n_seeds"]):
        for slug in ("hmm_draw", "hmm_no_draw", "draw_stratified_hmm"):
            obj = json.loads((outputs / "model_artifacts" / f"seed_{i:02d}" / f"{slug}_base.json").read_text())
            finals = [x["final_loglik"] for x in obj["start_diagnostics"]]
            em_iters.append(int(obj["n_iter"]))
            # Convergence evidence for the reported (selected) start: successive
            # log-likelihood increments, including the gain of the final M-step.
            chosen = [x for x in obj["start_diagnostics"] if x["selected"]][0]
            incr = np.diff(chosen["loglik_trace"] + [chosen["final_loglik"]])
            ratios = incr[1:] / incr[:-1]
            em_min_incr = min(em_min_incr, float(incr.min()))
            em_ratio_max = max(em_ratio_max, float(ratios.max()))
            em_last_gain_max = max(em_last_gain_max, float(incr[-1]))
            em_remain_max = max(em_remain_max, float(incr[-1] * ratios.max() / (1 - ratios.max())))
            em_abs_ll.append(abs(float(chosen["final_loglik"])))
            n_fits += 1
            coincident += int(max(finals) - min(finals) <= 0.01)
            max_spread = max(max_spread, max(finals) - min(finals))

    # Coverage of the nominal 90% interval by current simulator state (draw model).
    cov_by_state = prop_lm.groupby("state").cover90.mean()

    # Per-day Kullback-Leibler divergence between the generating emission laws of
    # every ordered pair of states: wearable channels versus the draw indicator.
    def kl_wearable(i: int, j: int) -> float:
        mu, sd = crp.TRUE_W_MU, crp.TRUE_W_SD
        return float(np.sum(np.log(sd[j] / sd[i]) + (sd[i] ** 2 + (mu[i] - mu[j]) ** 2) / (2 * sd[j] ** 2) - 0.5))

    def kl_draw(i: int, j: int) -> float:
        p, q = crp.TRUE_LAMBDA[i], crp.TRUE_LAMBDA[j]
        return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

    kl_ratios = [kl_wearable(i, j) / kl_draw(i, j) for i in range(crp.K) for j in range(crp.K) if i != j]

    evals, evecs = np.linalg.eig(crp.TRUE_P.T)
    stationary = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    stationary = stationary / stationary.sum()

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
             14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
             19: "nineteen", 20: "twenty"}
    ordinals = {1: "", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    tol_exp = int(round(np.log10(config["em_tol"])))
    assert np.isclose(config["em_tol"], 10.0 ** tol_exp), "EM tolerance must be a power of ten"

    def fmt_row(state: str, cells: list) -> str:
        return state + " & " + " & ".join(cells) + "\\\\"

    table1_wear = "\n".join(fmt_row(crp.STATE_NAMES[k], [
        f"${crp.TRUE_W_MU[k, 0]:.2f}\\pm{crp.TRUE_W_SD[k, 0]:.2f}$",
        f"${crp.TRUE_W_MU[k, 1]:.1f}\\pm{crp.TRUE_W_SD[k, 1]:.1f}$",
        f"${crp.TRUE_W_MU[k, 2]:.1f}\\pm{crp.TRUE_W_SD[k, 2]:.2f}$"]) for k in range(crp.K))
    table1_lab = "\n".join(fmt_row(crp.STATE_NAMES[k], [
        f"{crp.TRUE_L_MED[k, j]:.1f} ({crp.TRUE_L_LOGSD[k, j]:.2f})" for j in range(crp.J)])
        for k in range(crp.K))
    table1_eta = "\n".join(f"{crp.STATE_NAMES[k]} & \\multicolumn{{3}}{{c}}{{{crp.TRUE_PI[k]:.2f}}}\\\\"
                            for k in range(crp.K))
    true_p_rows = "\\\\\n".join("&".join(f"{x:.2f}" for x in row) for row in crp.TRUE_P)

    # The manuscript states that the log likelihood rose at every EM iteration.
    if not em_min_incr > 0:
        raise SystemExit("an archived EM trace is not strictly increasing")
    if not em_ratio_max < 1:
        raise SystemExit("an archived EM trace does not contract geometrically")

    replacements = {
        "@@PROP_NLL@@": f"{prop.nll:.3f}",
        "@@PROP_NLL_CI@@": rng(prop.nll_lo, prop.nll_hi, 3),
        "@@PROP_IBS@@": f"{prop.ibs4:.4f}",
        "@@PROP_IBS_CI@@": rng(prop.ibs4_lo, prop.ibs4_hi, 4),
        "@@PROP_COV@@": f"{100 * prop.cover90:.1f}\\%",
        "@@DIFF_NOLAM_NLL@@": signed(nol.comparator_minus_proposed, 4),
        "@@DIFF_NOLAM_NLL_CI@@": rng(nol.ci_low, nol.ci_high, 4),
        "@@DIFF_NOLAM_IBS@@": signed(nolb.comparator_minus_proposed, 5),
        "@@DIFF_NOLAM_IBS_CI@@": rng(nolb.ci_low, nolb.ci_high, 5),
        "@@DIFF_NOLAM_NLL_CI_ABS@@": rng_to(nol.ci_low, nol.ci_high, 4),
        "@@DIFF_NOLAM_IBS_CI_ABS@@": rng_to(nolb.ci_low, nolb.ci_high, 5),
        "@@N_SEEDS_FAVOR_NLL@@": str(n_seeds_favor),
        "@@SEED_T_CI_NLL@@": rng_to(t_nll[0], t_nll[1], 4),
        "@@SEED_T_CI_IBS@@": rng_to(t_ibs[0], t_ibs[1], 5),
        "@@NLL_DIFF_PCT@@": f"{100 * nol.comparator_minus_proposed / prop.nll:.2f}",
        "@@N_SEEDS_WORD@@": words[config["n_seeds"]],
        "@@N_SEEDS_DF_WORD@@": words[config["n_seeds"] - 1],
        "@@EM_TOL@@": f"$10^{{{tol_exp}}}$",
        "@@MAX_EM_ITER@@": str(config["max_em_iter"]),
        "@@COINCIDENT_PHRASE@@": (f"In all {n_fits} fits" if coincident == n_fits
                                  else f"In {coincident} of the {n_fits} fits"),
        "@@N_STARTS_WORD@@": words[config["n_starts"]],
        "@@N_PARAM_DRAWS_WORD_CAP@@": words[config["n_param_draws"]].capitalize(),
        "@@N_ALL_TEST_PATIENTS@@": str(config["n_seeds"] * (config["n_total"] - config["n_train"] - config["n_val"])),
        # Smallest power of ten that bounds the largest log-likelihood spread across EM starts.
        "@@START_SPREAD_BOUND@@": (lambda b: f"{b:.{max(0, -int(np.floor(np.log10(b))))}f}")(
            10.0 ** np.ceil(np.log10(max(max_spread, 1e-12)))),
        "@@LAMBDA_RECOVERY_AND@@": ", ".join(lam_parts[:-1]) + ", and " + lam_parts[-1],
        "@@HAZARD_STRIDE_ORD@@": ordinals[crp.HAZARD_LANDMARK_STRIDE] or "",
        "@@HAZARD_HORIZON@@": str(config["hazard_horizon"]),
        "@@TRUE_P_ROWS@@": true_p_rows,
        "@@TRUE_LAMBDA@@": ",".join(f"{x:.2f}" for x in crp.TRUE_LAMBDA),
        "@@TRUE_MEAN_R@@": f"{hit.loc['Remission', 'true_mean']:.3f}",
        "@@TRUE_MEAN_M@@": f"{hit.loc['Mild', 'true_mean']:.3f}",
        "@@STAT_FLARE@@": f"{stationary[2]:.2f}",
        "@@EVENT30_RATE@@": f"{100 * event30:.1f}\\%",
        "@@HMM_RISK30_MIN@@": f"{np.floor(100 * risk30.min()) / 100:.2f}",
        "@@HAZ_AUROC@@": f"{wide.loc[MODEL_HAZ].auroc:.4f}",
        "@@NCCAL_PROP@@": cal_slope_text(MODEL_DRAW),
        "@@NCCAL_HAZ@@": cal_slope_text(MODEL_HAZ),
        "@@TABLE1_WEAR_ROWS@@": table1_wear,
        "@@TABLE1_LAB_ROWS@@": table1_lab,
        "@@TABLE1_ETA_ROWS@@": table1_eta,
        "@@N_TEST_LANDMARKS@@": f"{n_test_landmarks:,}",
        "@@N_CURRENT_FLARE@@": f"{n_current:,}",
        "@@N_NONCURRENT_DISTINCT@@": f"{len(noncur_lm):,}",
        "@@NONCUR_MIN_LM@@": str(int(per_patient_noncur.min())),
        "@@NONCUR_MAX_LM@@": str(int(per_patient_noncur.max())),
        "@@NONCUR_LM_NLL_PROP@@": f"{noncur_lm.nll.mean():.3f}",
        "@@N_AUROC_PATIENTS@@": str(n_auroc_patients),
        "@@POOLED_AUROC_PROP@@": f"{pooled_auroc:.4f}",
        "@@ROW_PROP@@": row(MODEL_DRAW),
        "@@ROW_NOLAM@@": row(MODEL_NO_DRAW),
        "@@ROW_STRAT@@": row(MODEL_STRAT),
        "@@ROW_HAZ@@": row(MODEL_HAZ),
        "@@PAIR_NOLAM@@": paircell(MODEL_NO_DRAW),
        "@@PAIR_STRAT@@": paircell(MODEL_STRAT),
        "@@PAIR_HAZ@@": paircell(MODEL_HAZ),
        "@@LAMBDA_RECOVERY@@": ", ".join(lam_parts),
        "@@HIT_RECOVERY@@": " and ".join(hit_parts),
        "@@PROP_AUROC@@": f"{prop.auroc:.4f}",
        "@@PROP_AUROC_CI@@": rng(prop.auroc_lo, prop.auroc_hi, 4),
        "@@N_SEEDS@@": str(config["n_seeds"]),
        "@@N_TOTAL@@": str(config["n_total"]),
        "@@N_TRAIN@@": str(config["n_train"]),
        "@@N_VAL@@": str(config["n_val"]),
        "@@N_TEST@@": str(config["n_total"] - config["n_train"] - config["n_val"]),
        "@@PUBLIC_ARCHIVE_DOI@@": release_config["public_archive_doi"],
        "@@RELEASE_TAG@@": release_config["release_tag"],
        "@@SOFTWARE_VERSION@@": release_config["software_version"],
        "@@N_DAYS@@": str(config["n_days"]),
        "@@FUTURE_DAYS@@": str(config["future_days"]),
        "@@PMF_HORIZON@@": str(config["pmf_horizon"]),
        "@@N_PARAM_DRAWS@@": str(config["n_param_draws"]),
        "@@N_PARAM_MEMBERS@@": str(config["n_param_draws"] + 1),
        "@@N_HAZARD_DRAWS@@": str(config["n_hazard_draws"]),
        "@@N_HAZARD_MEMBERS@@": str(config["n_hazard_draws"] + 1),
        "@@N_STARTS@@": str(config["n_starts"]),
        "@@MAX_EM_ITER@@": str(config["max_em_iter"]),
        "@@PERFORMANCE_BOOTSTRAP@@": str(config["performance_bootstrap"]),
        "@@CALIBRATION_BOOTSTRAP@@": str(config["calibration_bootstrap"]),
        "@@CURRENT_FLARE_FRACTION@@": f"{100 * noncur_obj['current_flare_landmark_fraction']:.1f}\\%",
        "@@NONCURRENT_ROW_PROP@@": noncur_row(MODEL_DRAW),
        "@@NONCURRENT_ROW_NOLAM@@": noncur_row(MODEL_NO_DRAW),
        "@@NONCURRENT_DIFF_NLL@@": signed(ncd.comparator_minus_proposed, 4),
        "@@NONCURRENT_DIFF_NLL_CI@@": rng_to(ncd.ci_low, ncd.ci_high, 4),
        "@@NONCURRENT_DIFF_IBS@@": signed(ncdb.comparator_minus_proposed, 5),
        "@@NONCURRENT_DIFF_IBS_CI@@": rng(ncdb.ci_low, ncdb.ci_high, 5),
        "@@N_NONCURRENT_ROWS@@": f"{int(noncur_obj['n_noncurrent_landmark_rows']):,}",
        "@@EM_ITER_MIN@@": str(min(em_iters)),
        "@@EM_ITER_MAX@@": str(max(em_iters)),
        "@@COV_R@@": f"{cov_by_state[0]:.3f}",
        "@@COV_M@@": f"{cov_by_state[1]:.3f}",
        "@@COV_F@@": f"{cov_by_state[2]:.3f}",
        "@@KL_RATIO_MIN@@": f"{min(kl_ratios):.0f}",
        "@@KL_RATIO_MAX@@": f"{max(kl_ratios):.0f}",
        "@@EM_RATIO_MAX@@": f"{np.ceil(100 * em_ratio_max) / 100:.2f}",
        "@@EM_LAST_GAIN_MAX@@": f"{np.ceil(100 * em_last_gain_max) / 100:.2f}",
        "@@EM_REMAIN_MAX@@": f"{np.ceil(100 * em_remain_max) / 100:.2f}",
        "@@EM_ABS_LL_MIN@@": f"{min(em_abs_ll) / 1e4:.1f}",
        "@@EM_ABS_LL_MAX@@": f"{max(em_abs_ll) / 1e4:.1f}",
        "@@NULL_NLL_UB@@": f"{nol.ci_high:.4f}",
        "@@NULL_GM_PCT@@": f"{100 * np.expm1(nol.ci_high):.1f}\\%",
    }
    for key, value in replacements.items():
        tpl = tpl.replace(key, value)
    remaining = sorted(set(re.findall(r"@@[^@]+@@", tpl)))
    if remaining:
        raise SystemExit(f"Unreplaced placeholders: {remaining!r}")
    output.write_text(tpl)
    print(output)


if __name__ == "__main__":
    main()
