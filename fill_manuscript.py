#!/usr/bin/env python3
"""Fill the IEEE manuscript template from generated machine-readable outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

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
    wide = pd.read_csv(outputs / "tables" / "main_results_wide.csv").set_index("model")
    paired = pd.read_csv(outputs / "tables" / "paired_differences.csv")
    params = pd.read_csv(outputs / "tables" / "parameter_recovery.csv")
    hit = pd.read_csv(outputs / "tables" / "hitting_time_recovery.csv").set_index("state")
    noncur_perf = pd.read_csv(outputs / "tables" / "noncurrent_flare_performance.csv")
    noncur_pair = pd.read_csv(outputs / "tables" / "noncurrent_flare_paired_differences.csv")
    noncur_obj = json.loads((outputs / "tables" / "noncurrent_flare_sensitivity.json").read_text())

    def rng(lo: float, hi: float, d: int = 3) -> str:
        return f"{lo:.{d}f}--{hi:.{d}f}"

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
        return f"{x:+.{d}f}"

    def paircell(name: str) -> str:
        a = pair(name, "nll"); b = pair(name, "ibs4")
        return (
            f"{signed(a.comparator_minus_proposed, 4)} ({rng(a.ci_low, a.ci_high, 4)}) & "
            f"{signed(b.comparator_minus_proposed, 5)} ({rng(b.ci_low, b.ci_high, 5)})"
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
    lam_parts = [f"${params[col].mean():.3f}\\pm{params[col].std(ddof=1):.3f}$"
                 for col in ["lambda_Remission", "lambda_Mild", "lambda_Flare"]]
    hit_parts = [f"${hit.loc[state, 'estimated_mean']:.2f}\\pm{hit.loc[state, 'estimated_sd']:.2f}$"
                 for state in ["Remission", "Mild"]]

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
        "@@NONCURRENT_DIFF_NLL_CI@@": rng(ncd.ci_low, ncd.ci_high, 4),
        "@@NONCURRENT_DIFF_IBS@@": signed(ncdb.comparator_minus_proposed, 5),
        "@@NONCURRENT_DIFF_IBS_CI@@": rng(ncdb.ci_low, ncdb.ci_high, 5),
        "@@N_NONCURRENT_ROWS@@": f"{int(noncur_obj['n_noncurrent_landmark_rows']):,}",
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
