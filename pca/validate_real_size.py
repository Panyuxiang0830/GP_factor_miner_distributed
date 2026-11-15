#!/usr/bin/env python3
"""Check real subset reranking, numerical samples and full ordered QD for P1."""
import argparse
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/src'))
from gp_factor_miner.pca_qd import PCAQDArchive,QDRecord,signature_similarity
from reference import power_signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    load=lambda path:json.loads(path.read_text())
    source=load(a.run/'source.json');T,N=source['T'],source['N']
    raw=np.memmap(a.run/'raw_first2.f32',mode='r',dtype='<f4',shape=(2,T,N))
    executed=load(a.run/'EXECUTED.json');reports=[]
    for run in sorted(a.run.glob('*_r1')):
        m=load(run/'metrics.json');B,rows,n,k=m['factors'],m['rows'],m['cols'],m['steps']
        s=np.fromfile(run/'signatures.f32',dtype='<f4').reshape(B,rows)
        ranked=np.memmap(run/'R_first2.f64',mode='r',dtype='<f8',shape=(2,rows,n))
        # The initial vector follows the same fixed full-universe prefix.
        initial=np.fromfile(source['initial'],dtype='<f8')[:n] if 'initial' in source else None
        if initial is None:raise ValueError('source must preserve initial vector provenance')
        rank_error=0.;signature_error=0.
        for i in range(2):
            values=np.asarray(raw[i,::m['sample_step'],:n],dtype=float)
            expected=pd.DataFrame(values).rank(axis=1,method='average',pct=True).fillna(.5).to_numpy(float)
            expected-=expected.mean(axis=1,keepdims=True)
            rank_error=max(rank_error,float(np.max(np.abs(expected-ranked[i]))))
            assert rank_error<3e-15,(run,rank_error)
            ref=power_signature(expected,initial,k).astype(float);actual=s[i].astype(float)
            signature_error=max(signature_error,max(0.,1-abs(float(ref@actual))/(np.linalg.norm(ref)*np.linalg.norm(actual))))
        assert signature_error<=1e-6,(run,signature_error)
        records=[QDRecord(x['key'],x['key'],x['quality'],1,1,sig) for x,sig in zip(source['records'][:B],s)
                 if np.isfinite(signature_similarity(sig,sig))]
        archive=PCAQDArchive(.90,.002);oracle=archive.consider_batch(records,backend='cpu');actual=load(run/'decisions.json')
        assert len(oracle)==len(actual)
        for expected,got in zip(oracle,actual):
            assert (expected.action,expected.closest_key,expected.replaced_key)==tuple(got[x] for x in ('action','closest_key','replaced_key'))
            if np.isfinite(expected.max_similarity):assert got['similarity'] is not None and abs(expected.max_similarity-got['similarity'])<=1e-12
            else:assert got['similarity'] is None
        assert list(archive)==load(run/'archive_keys.json')
        for repeat in (2,3):
            other=run.with_name(run.name[:-1]+str(repeat))
            for f in ('signatures.f32','decisions.json','archive_keys.json'):assert (run/f).read_bytes()==(other/f).read_bytes()
        reports.append(dict(configuration=run.name,rank_error=rank_error,signature_error=signature_error,
            numerical_sample_count=2,full_qd_candidates=len(records),repetitions_byte_identical=True))
    assert len(reports)==executed['configurations']
    (a.run/'SUCCESS.json').write_text(json.dumps(dict(verified=True,real_market_panels=True,
        independent_subset_ranking=True,configurations=reports,
        scope='Full QD/repeats; first two true raw panels reranked in pandas and signatures independently recomputed'),indent=2)+'\n')
    (a.run/'SUCCESS').touch();print(f'Verified {len(reports)} real-input configurations.')


if __name__=='__main__':main()
