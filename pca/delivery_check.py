#!/usr/bin/env python3
"""Small integration acceptance: seeded archives, continuation, numerics and failures."""
import argparse
import json
from pathlib import Path
import subprocess
import copy
import sys
from types import SimpleNamespace
import numpy as np
from archive_io import save_archive,load_archive
from validate import validate,angle_error


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mpiexec',default='mpirun');p.add_argument('--cuda',action='store_true');a=p.parse_args()
    root=a.out.resolve();runs=Path(__file__).resolve().parent/'runs'
    if root.exists() or not root.is_relative_to(runs.resolve()):p.error('new private pca/runs child required')
    root.mkdir(parents=True);rng=np.random.default_rng(9102);m,n,B=24,17,8
    initial=rng.normal(size=n);initial.astype('<f8').tofile(root/'initial.f8')
    matrices=rng.normal(size=(B,m,n));matrices-=matrices.mean(axis=2,keepdims=True)
    matrices[1]=matrices[0];matrices[2]=-matrices[0];matrices[5]=0
    matrices.astype('<f4').tofile(root/'R.f32')
    entries=[dict(key=f'factor_{i}',quality=.03+.004*i,candidate=i!=7,path=str(root/'R.f32'),slot=i) for i in range(B)]
    base=dict(rows=m,cols=n,initial=str(root/'initial.f8'),records=entries)
    manifest=root/'manifest.json';manifest.write_text(json.dumps(base))
    def check(run,source=manifest,svd=False):
        args=SimpleNamespace(manifest=source,run=run,svd_count=2 if svd else 0,svd_tolerance=1e-6 if svd else None,
            signature_tolerance=1e-6,power_reference=None,allow_legacy_reference=False,archive_input=None)
        report=validate(args);(run/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        assert report['passed'],(run,report)
        assert report['snapshot_passed'] and report['provenance_checked'] and report['independent_python_qd_agreement']
        return args
    reports=[]
    def execute(tag,source=manifest,extra=(),mpi=False,svd=False):
        run=root/tag;cmd=([a.mpiexec,'-np','4','--map-by','slot','--bind-to','core'] if mpi else [])
        cmd += [str(a.binary.resolve()),'--manifest',str(source),'--out',str(run),'--chunk','4',*extra]
        with (root/(tag+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        args=check(run,source,svd);reports.append(tag);return run,args
    power,power_args=execute('power')
    lanczos,lanczos_args=execute('lanczos',extra=('--method','lanczos','--steps','17'),svd=True)
    execute('lanczos_low_budget',extra=('--method','lanczos','--steps','1'))
    for method,k in (('power','4'),('lanczos','17')):
        for layout in ('date','transpose'):
            execute(f'{method}_{layout}',extra=('--method',method,'--steps',k,'--layout',layout),mpi=True,svd=method=='lanczos')
    # Fixed signatures isolate archive semantics; include seeded ties, same key,
    # strict quality margin, a threshold-near pair, invalid and constant rows.
    q,_=np.linalg.qr(rng.normal(size=(m,4)));q-=q.mean(axis=0);q,_=np.linalg.qr(q)
    x,y=q[:,0].astype('f4'),q[:,1].astype('f4')
    seed=root/'seed.json';save_archive(seed,{'a':(.03,x),'z':(.03,x)},m)
    signatures=np.array([x,x,y,y,.9*y+np.sqrt(1-.9**2)*q[:,2],np.ones(m),np.full(m,np.nan),-y],dtype='<f4')
    keys=('a','replacement','orth','orth','threshold','constant','nan','last')
    qualities=(.04,.05,.03,.03+.002,.031,.03,.03,.05)
    supplied=dict(base,records=[dict(key=k,quality=v,candidate=True) for k,v in zip(keys,qualities)])
    supplied_path=root/'replay.json';supplied_path.write_text(json.dumps(supplied));signatures.tofile(root/'provided.f32')
    prefix=('--signature-input',str(root/'provided.f32'),'--archive-input',str(seed))
    whole,_=execute('seeded_cpu',supplied_path,prefix)
    distributed,_=execute('seeded_mpi',supplied_path,(*prefix,'--archive','mpi'),mpi=True)
    first=dict(supplied,records=supplied['records'][:4]);second=dict(supplied,records=supplied['records'][4:])
    for tag,obj in (('first',first),('second',second)):(root/(tag+'.json')).write_text(json.dumps(obj))
    signatures[4:].tofile(root/'second.f32')
    one,_=execute('first_batch',root/'first.json',prefix)
    two,_=execute('second_batch',root/'second.json',('--signature-input',str(root/'second.f32'),'--archive-input',str(one/'archive.json'),'--archive','mpi'),mpi=True)
    expected=load_archive(whole/'archive.json',m)
    for other in (distributed,two):
        got=load_archive(other/'archive.json',m);assert sorted(expected)==sorted(got)
        assert all(expected[k][0]==got[k][0] and np.array_equal(expected[k][1],got[k][1]) for k in expected)
    # Regression for the independently reproduced false-positive validator.
    saved=(lanczos/'signatures.f32').read_bytes();np.ones((B,m),dtype='<f4').tofile(lanczos/'signatures.f32')
    try:
        try:bad=validate(lanczos_args);assert not bad['passed'],'wrong signatures passed'
        except ValueError:pass
    finally:(lanczos/'signatures.f32').write_bytes(saved)
    for bad in (np.array([np.nan,1]),np.array([0.,0.])):
        try:angle_error(bad,np.array([1.,0.]));raise AssertionError('invalid comparison passed')
        except ValueError:pass
    # Snapshot parse rejection and provenance binding must fail closed.
    malformed=json.loads(seed.read_text());malformed['records'].append(malformed['records'][0]);bad_seed=root/'bad_seed.json';bad_seed.write_text(json.dumps(malformed))
    try:load_archive(bad_seed,m);raise AssertionError('bad archive accepted')
    except ValueError:pass
    result=subprocess.run([str(a.binary.resolve()),'--manifest',str(supplied_path),'--out',str(root/'bad_output'),'--signature-input',str(root/'provided.f32'),'--archive-input',str(bad_seed)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert result.returncode!=0 and not (root/'bad_output').exists()
    original=manifest.read_text();manifest.write_text(original+' ')
    try:
        try:validate(power_args);raise AssertionError('changed input identity accepted')
        except ValueError:pass
    finally:manifest.write_text(original)
    cache=root/'cached_reference';cache.mkdir()
    for name in ('signatures.f32','provenance.json'):(cache/name).write_bytes((power/name).read_bytes())
    cached_args=copy.copy(power_args);cached_args.power_reference=cache/'signatures.f32'
    assert validate(cached_args)['passed']
    metadata=json.loads((cache/'provenance.json').read_text());metadata['initial_hash_fnv1a64']='wrong'
    (cache/'provenance.json').write_text(json.dumps(metadata))
    try:validate(cached_args);raise AssertionError('wrong cached initial vector accepted')
    except ValueError:pass
    if a.cuda:
        execute('seeded_gpu',supplied_path,(*prefix,'--device','cuda'))
        for precision in ('float','double'):
            execute('gpu_'+precision,extra=('--device','cuda','--precision',precision))
    result=dict(verified=True,scenarios=reports,seeded_archive_python_exact=True,continuation_matches_single_batch=True,
        final_snapshot_contents_exact=True,same_budget_python_numerics=True,svd_gate=True,negative_regressions=True,cuda=a.cuda)
    (root/'SUCCESS.json').write_text(json.dumps(result,indent=2)+'\n');(root/'SUCCESS').touch();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
