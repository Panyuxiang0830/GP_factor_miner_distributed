#!/usr/bin/env python3
"""Validate full-generation MPI signatures and compare the common certified cohort."""
import argparse
import json
import os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    load=lambda p:json.loads(p.read_text())
    plan=load(a.plan);B,m=plan['count'],plan['rows'];target=plan['target'];methods={};proof=[]
    assert load(a.run/'EXECUTED.json')['factors']==B
    for name in ('power','lanczos'):
        spec=plan['methods'][name];ref=np.fromfile(spec['reference'],dtype='<f4').reshape(B,m).astype(float)
        errors=np.fromfile(spec['svd_error'],dtype='<f8');budgets=np.fromfile(spec['steps'],dtype='<i4')
        repeats=[]
        for r in range(4):
            root=a.run/f'{name}_r{r}';metrics=load(root/'metrics.json');rows=load(root/'factors.json')
            assert len(rows)==B and metrics['factors']==B and metrics['target']==target
            sig=np.fromfile(root/'signatures.f32',dtype='<f4').reshape(B,m)
            x=sig.astype(float);norm=np.linalg.norm(x,axis=1)*np.linalg.norm(ref,axis=1)
            assert np.all(norm>0) and np.isfinite(x).all()
            err=np.maximum(0.,1-np.abs(np.sum(x*ref,axis=1)/norm))
            assert float(err.max())<=1e-6
            assert [row['index'] for row in rows]==list(range(B))
            assert np.array_equal(np.array([row['budget'] for row in rows]),budgets)
            for i,row in enumerate(rows):
                assert abs(row['serial_svd_error']-errors[i])<=1e-15
                assert abs(row['mpi_vs_serial_error']-err[i])<=1e-12
                same=np.array_equal(sig[i].astype(float),ref[i]);assert bool(row['identical_to_serial'])==same
                # Verify certification without relying on rounded arccos near one.
                # The recorded bound includes a conservative cosine rounding allowance.
                theta=np.arccos(np.clip(1-errors[i],0,1))
                delta=0. if same else np.arccos(np.clip(1-err[i]-5e-13,0,1))
                lower_check=1-np.cos(min(np.pi/2,theta+delta))
                assert row['svd_error_upper_bound']+1e-12>=lower_check
                assert row['allreduce_calls']>0 and row['logical_payload_bytes']>0
            if r>=2:assert (root/'signatures.f32').read_bytes()==(a.run/f'{name}_r1/signatures.f32').read_bytes()
            repeats.append((metrics,rows))
        certified=np.array([x['svd_error_upper_bound']<=target for x in repeats[1][1]]) & (errors<=target)
        methods[name]=dict(repeats=repeats,certified=certified,errors=errors,budgets=budgets)
    common=methods['power']['certified'] & methods['lanczos']['certified']
    assert common.any(),'no common target-certified factors'
    summary=[]
    for name in ('power','lanczos'):
        data=methods[name];row=dict(method=name,processed=B,common_target_certified=int(common.sum()),
            serial_budget_censored=int(np.sum(data['errors']>target)),mpi_target_certified=int(data['certified'].sum()))
        fields=('solve_seconds','communication_seconds','allreduce_calls','logical_payload_bytes','matvecs','steps')
        for mask,label in ((np.ones(B,dtype=bool),'all'),(common,'common')):
            for field in fields:
                values=[sum(x[field] for i,x in enumerate(records) if mask[i]) for _,records in data['repeats'][1:]]
                row[f'{label}_{field}']=float(np.median(values))
        row['total_seconds']=float(np.median([met['total_seconds'] for met,_ in data['repeats'][1:]]))
        summary.append(row)
    result=dict(verified=True,processed_factors=B,ranks=methods['power']['repeats'][0][0]['ranks'],target=target,
        max_steps=plan['max_steps'],common_target_certified=int(common.sum()),methods=summary,
        scope='All input matrices executed; same-accuracy costs compare only the identical jointly certified cohort. Offline SVD-selected budgets; capped nonconvergence reported separately. Per-factor reductions for both methods; no residual diagnostics.')
    (a.run/'SUCCESS.json').write_text(json.dumps(result,indent=2)+'\n');(a.run/'SUCCESS').touch()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
