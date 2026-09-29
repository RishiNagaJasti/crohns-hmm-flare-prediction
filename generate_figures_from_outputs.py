#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import crohns_hmm_pipeline as crp


def load_landmarks(out: Path) -> pd.DataFrame:
    p=out/'landmark_predictions.parquet'
    return pd.read_parquet(p) if p.exists() else pd.read_csv(out/'landmark_predictions.csv.gz')


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--outputs',type=Path,required=True)
    a=ap.parse_args(); out=a.outputs.resolve(); cfgd=json.loads((out/'config.json').read_text())
    cfg=argparse.Namespace(**cfgd)
    landmarks=load_landmarks(out); patient_metrics=pd.read_csv(out/'patient_metrics.csv')
    slug={crp.MODEL_DRAW:'hmm_draw',crp.MODEL_NO_DRAW:'hmm_no_draw',crp.MODEL_DRAW_STRATIFIED:'draw_stratified_hmm'}
    fitted_proposed=[]; first={}
    for i in range(cfg.n_seeds):
        art=out/'model_artifacts'/f'seed_{i:02d}'
        current={name:crp.load_params(art/f'{sl}_base.json') for name,sl in slug.items()}
        fitted_proposed.append(current[crp.MODEL_DRAW])
        if i==0: first=current
    crp.make_figures(landmarks,patient_metrics,crp.select_exemplar(cfg,first),fitted_proposed,out,tuple(cfg.horizons))
    rows=[]
    for i,params in enumerate(fitted_proposed):
        _,_,N=crp.hitting_components(params.P)
        row={'seed':i}
        for k,name in enumerate(crp.STATE_NAMES): row[f'lambda_{name}']=params.lam[k] if params.lam is not None else float('nan')
        row['mean_R'],row['mean_M']=(N @ np.ones(2)).tolist(); rows.append(row)
    pd.DataFrame(rows).to_csv(out/'tables'/'parameter_recovery.csv',index=False)
    print('Figure and recovery worker complete',flush=True)
    os._exit(0)
if __name__=='__main__': main()
