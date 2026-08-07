#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib
from pathlib import Path

EXCLUDE_PARTS={'.git','.venv','.pytest_cache','__pycache__','quick_outputs','smoke_outputs','profile_seed','profile_seed1','rendered','_renders'}
EXCLUDE_SUFFIXES={'.aux','.fdb_latexmk','.fls','.log','.out','.pyc','.pid'}

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); ap.add_argument('--output',type=Path,default=Path('MANIFEST.sha256'))
    a=ap.parse_args(); root=a.root.resolve(); out=a.output if a.output.is_absolute() else root/a.output
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p==out: continue
        rel=p.relative_to(root)
        if any(part in EXCLUDE_PARTS for part in rel.parts): continue
        if p.suffix in EXCLUDE_SUFFIXES: continue
        if p.name.endswith('.zip'): continue
        rows.append(f"{digest(p)}  {rel.as_posix()}")
    out.write_text('\n'.join(rows)+'\n')
    print(f'{len(rows)} files -> {out}')
if __name__=='__main__': main()
