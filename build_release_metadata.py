#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNAVAILABLE"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()



def _release_config(root):
    """Read the tracked release configuration so a clean rebuild cannot
    overwrite a real archival DOI with the placeholder."""
    p = root / 'release_config.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--outputs',type=Path,default=Path('final_outputs'))
    a=ap.parse_args(); root=a.root.resolve(); out=a.outputs if a.outputs.is_absolute() else root/a.outputs
    commit=run(['git','rev-parse','HEAD'],root); tag=run(['git','describe','--tags','--exact-match'],root)
    status=run(['git','status','--porcelain','--untracked-files=no'],root)
    bundle=root/'release_source.bundle'
    if commit!='UNAVAILABLE':
        subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=root,check=True)
    meta={
      'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'source_commit':commit,
      'release_tag':tag,
      'tracked_worktree_clean': status == '',
      'source_bundle': bundle.name if bundle.exists() else None,
      'source_bundle_sha256': sha256(bundle) if bundle.exists() else None,
      'pipeline_sha256':sha256(root/'crohns_hmm_pipeline.py'),
      'manuscript_template_sha256':sha256(root/'manuscript_template.tex'),
      'generated_tex_sha256':sha256(root/'Crohns_HMM_Time_to_Flare_Study.tex'),
      'generated_pdf_sha256':sha256(root/'Crohns_HMM_Time_to_Flare_Study.pdf'),
      'output_config_sha256':sha256(out/'config.json'),
      'public_archive_doi':_release_config(root).get('public_archive_doi','PENDING_AUTHOR_DEPOSIT'),
      'software_version':_release_config(root).get('software_version'),
      'doi_note':'Read from release_config.json. A DOI cannot be invented; deposit the immutable release and set it there before submission.'
    }
    (root/'release_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(root/'release_metadata.json')
if __name__=='__main__': main()
