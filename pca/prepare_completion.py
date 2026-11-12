#!/usr/bin/env python3
"""Prepare private real-panel candidates and full-generation accuracy budgets."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python/src'))
from gp_factor_miner.expressions import expression_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--panel-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int)
    p.add_argument('--candidates',type=int,default=512)
    p.add_argument('--target',type=float,default=1e-6)
    a=p.parse_args();out=a.out.resolve()
    if out.exists() or not out.is_relative_to((Path(__file__).resolve().parent/'runs').resolve()):p.error('new output under pca/runs required')
    manifest=json.loads(a.manifest.read_text())
    metrics=np.load(a.study/'metrics.npy',mmap_mode='r');sig=np.load(a.study/'signatures.npy',mmap_mode='r')
    complete=np.load(a.study/'completed.npy')
    count=a.count or len(manifest['records']);m,n=manifest['rows'],manifest['cols']
    if not(0<count<=len(manifest['records']) and len(complete)==len(manifest['records']) and complete.all()):p.error('complete same-input numerical study required')
    if sig.shape!=(len(complete),metrics.shape[1],2,metrics.shape[3],m) or not 0<a.target<1:p.error('invalid study shape/target')
    if not 0<a.candidates:p.error('candidates must be positive')
    out.mkdir(parents=True)
    candidates=[];seen=set()
    for line in a.scores.read_text().splitlines():
        row=json.loads(line);quality=row.get('abs_rank_ic')
        if not row.get('valid',True) or row.get('constant_R',False) or not row.get('candidate') or quality is None or not math.isfinite(quality):continue
        key=expression_key(row['expression'])
        if key in seen:continue
        seen.add(key);candidates.append(dict(index=row['index'],key=key,quality=quality,expression=row['expression']))
        if len(candidates)==a.candidates:break
    (out/'real_candidates.json').write_text(json.dumps(candidates,ensure_ascii=False)+'\n')
    plan=dict(manifest=str(a.manifest.resolve()),rows=m,cols=n,count=count,target=a.target,
              max_steps=metrics.shape[3],seed=0,methods={},budget_source='first serial SVD-validated step within target; max budget for censored factors')
    for method,name in enumerate(('power','lanczos')):
        errors=np.asarray(metrics[:count,0,method,:,0]);passed=errors<=a.target
        eligible=passed.any(axis=1)
        budgets=np.where(eligible,passed.argmax(axis=1)+1,errors.shape[1]).astype('<i4')
        selected=np.asarray(sig[np.arange(count),0,method,budgets-1,:],dtype='<f4')
        chosen=np.asarray(errors[np.arange(count),budgets-1],dtype='<f8')
        assert np.isfinite(selected).all() and np.isfinite(chosen).all()
        for suffix,values in (('steps.i4',budgets),('reference.f32',selected),('svd_error.f8',chosen)):
            values.tofile(out/f'{name}.{suffix}')
        plan['methods'][name]=dict(steps=str(out/f'{name}.steps.i4'),reference=str(out/f'{name}.reference.f32'),
            svd_error=str(out/f'{name}.svd_error.f8'),serial_target_reached=int(eligible.sum()))
    (out/'accuracy_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'real_source.json').write_text(json.dumps(dict(panel_dir=str(a.panel_dir.resolve()),
        candidates=str(out/'real_candidates.json'),initial=manifest['initial'],
        scope='Fixed full-universe factor panels; sample dates/subset stocks and rerank for each PCA configuration'),indent=2)+'\n')
    print(f'Prepared {len(candidates)} real candidates and accuracy budgets for {count} matrices; no market matrices copied.')


if __name__=='__main__':main()
