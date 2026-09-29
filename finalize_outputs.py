#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import pandas as pd
import crohns_hmm_pipeline as crp


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--outputs',type=Path,required=True)
    a=ap.parse_args(); out=a.outputs.resolve(); cfgd=json.loads((out/'config.json').read_text()); cfg=argparse.Namespace(**cfgd)
    seed_dir=out/'seed_results'; all_lm=[]; all_pm=[]
    for i in range(cfg.n_seeds):
        all_lm.append(pd.read_csv(seed_dir/f'seed_{i:02d}_landmarks.csv.gz'))
        all_pm.append(pd.read_csv(seed_dir/f'seed_{i:02d}_patient_metrics.csv'))
        src=out/'model_artifacts'/f'seed_{i:02d}'/'hmm_draw_base.json'
        (out/'tables').mkdir(exist_ok=True)
        (out/'tables'/f'params_seed_{i:02d}.json').write_bytes(src.read_bytes())
    landmarks=pd.concat(all_lm,ignore_index=True); patients=pd.concat(all_pm,ignore_index=True)
    try:
        landmarks.to_parquet(out/'landmark_predictions.parquet',index=False)
    except ImportError:
        landmarks.to_csv(out/'landmark_predictions.csv.gz',index=False,compression='gzip')
    patients.to_csv(out/'patient_metrics.csv',index=False)
    p=out/'landmark_predictions.parquet'
    landmarks=pd.read_parquet(p) if p.exists() else pd.read_csv(out/'landmark_predictions.csv.gz')
    patients=pd.read_csv(out/'patient_metrics.csv')
    print('Aggregating primary results',flush=True)
    result=crp.aggregate_results(landmarks,patients,out,tuple(cfg.horizons),cfg.performance_bootstrap,cfg.calibration_bootstrap)
    print('Aggregating non-current-flare sensitivity',flush=True)
    sensitivity=crp.aggregate_noncurrent_flare_sensitivity(landmarks,out,tuple(cfg.horizons),cfg.performance_bootstrap,cfg.calibration_bootstrap)
    result['noncurrent_flare_sensitivity']=sensitivity
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print('Writing archive indices',flush=True); crp.write_archive_indices(out)
    print('Generating figures and recovery tables',flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('generate_figures_from_outputs.py')),'--outputs',str(out)],check=True)
    print(f'Finalization complete: {out}',flush=True)
    os._exit(0)
if __name__=='__main__': main()
