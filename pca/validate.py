#!/usr/bin/env python3
"""Separate same-budget numerics, exact-SVD convergence, QD, and snapshot checks."""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/src'))
from gp_factor_miner.pca_qd import PCAQDArchive,QDRecord,signature_similarity
from reference import power_signature,lanczos_signature
from archive_io import load_archive,fingerprint


def require(value,message):
    if not value:raise ValueError(message)


def angle_error(left,right):
    left,right=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    require(np.isfinite(left).all() and np.isfinite(right).all(),'nonfinite signature')
    a,b=np.linalg.norm(left),np.linalg.norm(right)
    if a==0 or b==0:
        require(a==b==0,'zero signature disagrees with reference');return 0.
    return float(max(0.,1-abs(left@right)/(a*b)))


def archive_replay(manifest,signatures,initial):
    archive=PCAQDArchive(.90,.002)
    archive._records={key:QDRecord(key,key,q,1,1,s) for key,(q,s) in initial.items()}
    indices=[];records=[]
    for i,(row,s) in enumerate(zip(manifest['records'],signatures)):
        if row['candidate'] and np.isfinite(signature_similarity(s,s)):
            indices.append(i);records.append(QDRecord(row['key'],row['key'],row['quality'],1,1,s))
    decisions=archive.consider_batch(records,backend='cpu')
    trace=[(i,r.key,d.action,d.closest_key,d.replaced_key) for i,r,d in zip(indices,records,decisions)]
    return archive,decisions,trace


def validate(args):
    manifest=json.loads(args.manifest.read_text());metrics=json.loads((args.run/'metrics.json').read_text())
    m,n,count=manifest['rows'],manifest['cols'],metrics['factors']
    require(0<count<=len(manifest['records']),'invalid factor count')
    provenance_path=args.run/'provenance.json';provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    initial_path=Path(provenance['initial']) if provenance else Path(manifest['initial'])
    if provenance:
        require(fingerprint(args.manifest)==provenance['manifest_hash_fnv1a64'],'manifest fingerprint changed')
        require(fingerprint(initial_path)==provenance['initial_hash_fnv1a64'],'initial vector fingerprint changed')
    initial=np.fromfile(initial_path,dtype='<f8');require(initial.shape==(n,) and np.isfinite(initial).all() and np.linalg.norm(initial)>0,'invalid initial vector')
    signatures=np.fromfile(args.run/'signatures.f32',dtype='<f4').reshape(count,m)
    supplied=bool(metrics.get('input_signatures',False));expected=None;svd_errors=[];residuals=[]
    if not supplied:require(np.isfinite(signatures).all(),'nonfinite output signatures')
    cached=args.power_reference
    if cached:
        require(not supplied,'provided-signature replay cannot use --power-reference')
        require(cached.resolve()!=(args.run/'signatures.f32').resolve(),'a run cannot be its own cached reference')
        require(metrics['method']=='power' and metrics['steps']==4,'--power-reference only supports four-step power')
        p=cached.parent/'provenance.json'
        if p.exists() and provenance:
            previous=json.loads(p.read_text())
            for field in ('manifest_hash_fnv1a64','initial_hash_fnv1a64','method','steps','rank_input','factors'):
                require(previous[field]==provenance[field],'cached reference configuration mismatch: '+field)
        else:require(args.allow_legacy_reference,'cached reference lacks provenance; explicitly use --allow-legacy-reference only for audited historical inputs')
        expected=np.load(cached) if cached.suffix=='.npy' else np.fromfile(cached,dtype='<f4').reshape(count,m)
        require(expected.shape==(count,m),'cached reference shape mismatch')
    elif supplied:
        if provenance and provenance['signature_input']:
            expected=np.fromfile(provenance['signature_input'],dtype='<f4')[:count*m].reshape(count,m)
            require(np.array_equal(signatures,expected,equal_nan=True),'provided signatures changed during replay')
    else:expected=np.zeros_like(signatures)
    require(args.svd_tolerance is None or args.svd_count>0,'SVD tolerance requires --svd-count > 0')
    require(not supplied or args.svd_count==0,'SVD check is not defined for supplied-signature replay')
    for i,row in enumerate(manifest['records'][:count]):
        if supplied:break
        if cached and i>=args.svd_count:continue
        R=np.memmap(row['path'],dtype='<f4',mode='r',offset=row['slot']*m*n*4,shape=(m,n)).astype(float)
        if metrics.get('rank_input',0):
            R=pd.DataFrame(R).rank(axis=1,method='average',pct=True).fillna(.5).to_numpy(float);R-=R.mean(axis=1,keepdims=True)
        require(np.isfinite(R).all(),'nonfinite preprocessed matrix')
        if not cached and np.any(R):expected[i]=(power_signature if metrics['method']=='power' else lanczos_signature)(R,initial,metrics['steps'])
        if i<args.svd_count:
            U,s,_=np.linalg.svd(R,full_matrices=False)
            require(s[0]>0 and np.linalg.norm(signatures[i])>0,'SVD sample has zero matrix/signature')
            svd_errors.append(angle_error(U[:,0],signatures[i]));v=signatures[i].astype(float)
            residuals.append(float(np.linalg.norm(R@(R.T@v)-s[0]**2*v)/(s[0]**2*np.linalg.norm(v))))
    numerical_checked=expected is not None
    errors=([0.]*count if supplied else [angle_error(x,y) for x,y in zip(signatures,expected)]) if numerical_checked else []
    max_error=max(errors) if errors else None
    numerical_passed=max_error<=args.signature_tolerance if numerical_checked else None
    archive_path=args.archive_input or (Path(provenance['archive_input']) if provenance and provenance['archive_input'] else None)
    if provenance:
        recorded=provenance['archive_input']
        require((not recorded and archive_path is None) or (archive_path is not None and Path(recorded).resolve()==archive_path.resolve()),'initial archive path differs from run')
        if archive_path:
            require(fingerprint(archive_path)==provenance['archive_hash_fnv1a64'],'initial archive metadata changed')
            meta=json.loads(archive_path.read_text())
            require(fingerprint(archive_path.parent/meta['signatures'])==provenance['archive_signatures_hash_fnv1a64'],'initial archive signatures changed')
    seeded=load_archive(archive_path,m) if archive_path else {}
    archive,decisions,trace=archive_replay(manifest,signatures,seeded)
    actual=json.loads((args.run/'decisions.json').read_text())
    actual_trace=[tuple(x[k] for k in ('signature_index','key','action','closest_key','replaced_key')) for x in actual]
    decision_passed=trace==actual_trace;similarity_error=0.
    require(len(actual)==len(decisions),'candidate filtering mismatch')
    for expected_decision,row in zip(decisions,actual):
        x,y=expected_decision.max_similarity,row['similarity']
        if np.isfinite(x):
            require(y is not None and np.isfinite(y),'finite similarity became nonfinite');similarity_error=max(similarity_error,abs(x-y))
        else:require(y is None,'nonfinite similarity mask mismatch')
    keys=json.loads((args.run/'archive_keys.json').read_text());key_agreement=keys==list(archive)
    independent_qd=None
    if numerical_checked:
        independent_archive,_,independent_trace=archive_replay(manifest,expected,seeded)
        independent_qd=independent_trace==actual_trace and list(independent_archive)==keys
    snapshot_checked=(args.run/'archive.json').exists();snapshot_passed=None
    if snapshot_checked:
        saved=load_archive(args.run/'archive.json',m)
        snapshot_passed=sorted(saved)==list(archive) and all(saved[k][0]==r.abs_rank_ic and np.array_equal(saved[k][1],r.signature) for k,r in archive._records.items())
    svd_checked=bool(svd_errors);svd_passed=None if args.svd_tolerance is None else bool(svd_checked and max(svd_errors)<=args.svd_tolerance and np.isfinite(residuals).all())
    passed=decision_passed and key_agreement and similarity_error<=1e-12 and numerical_passed is not False and independent_qd is not False and snapshot_passed is not False and svd_passed is not False
    if not supplied:passed=passed and numerical_checked
    return dict(passed=bool(passed),candidates=len(trace),initial_archive_size=len(seeded),decision_mismatches=[] if decision_passed else ['trace mismatch'],
        archive_key_agreement=key_agreement,numerical_checked=numerical_checked,numerical_passed=numerical_passed,
        numerical_reference=('provided signatures' if supplied else metrics['method']+' same-budget Python'),
        max_one_minus_abs_cosine_vs_reference=max_error,max_one_minus_abs_cosine_vs_power=max_error if metrics['method']=='power' else None,
        independent_python_qd_agreement=independent_qd,svd_checked=svd_checked,svd_passed=svd_passed,
        svd_one_minus_abs_cosine=svd_errors,svd_relative_residual=residuals,max_abs_similarity_error=similarity_error,
        snapshot_checked=snapshot_checked,snapshot_passed=snapshot_passed,provenance_checked=provenance is not None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--svd-count',type=int,default=0);p.add_argument('--svd-tolerance',type=float)
    p.add_argument('--signature-tolerance',type=float,default=1e-6);p.add_argument('--power-reference',type=Path)
    p.add_argument('--allow-legacy-reference',action='store_true');p.add_argument('--archive-input',type=Path)
    args=p.parse_args();require(args.svd_count>=0 and args.signature_tolerance>=0 and (args.svd_tolerance is None or args.svd_tolerance>=0),'invalid tolerance/count')
    try:report=validate(args)
    except (ValueError,KeyError,OSError,TypeError) as error:report=dict(passed=False,error=str(error))
    (args.run/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
