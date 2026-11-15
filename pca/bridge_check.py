#!/usr/bin/env python3
"""Live bridge acceptance against Python and the original complete GP loop."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/src'))
import numpy as np
import pandas as pd
from gp_factor_miner.cache import ExpressionCache
from gp_factor_miner.gp_engine import _evaluate_with_values
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
import gp_factor_miner.hierarchical_engine as gp
from gp_factor_miner.pca_qd import PCAQDArchive,QDRecord,factor_signature,signature_similarity
from gp_bridge import CppGPBackend
from validate import angle_error


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine-binary',type=Path,required=True);p.add_argument('--pca-binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ranks',type=int,default=4);p.add_argument('--device',default='cpu');p.add_argument('--mpiexec',default='mpirun');a=p.parse_args()
    out=a.out.resolve()
    if out.exists() or not out.is_relative_to((Path(__file__).resolve().parent/'runs').resolve()):p.error('new private run required')
    out.mkdir();panel=out/'panels';(panel/'fields').mkdir(parents=True)
    rng=np.random.default_rng(2026);T,N,F=201,8,6;data={f'f{i}':rng.normal(size=(T,N))+np.linspace(-.2,.2,N) for i in range(F)}
    target=(data['f0']-.4*data['f1']).astype('<f4');target.tofile(panel/'target.f32')
    for name,x in data.items():x.astype('<f4').tofile(panel/'fields'/f'{name}.f32')
    (panel/'meta.json').write_text(json.dumps(dict(T=T,N=N,dates=[str(i) for i in range(T)],assets=[str(i) for i in range(N)],fields=list(data))))
    config=replace(HierarchicalGPConfig().smoke(),population_size=8,gene_pool_size=6,parent_pool_size=6,warm_start_multiplier=2,
        generations_per_stage=1,progress_chunk_size=8,windows=(1,2,3),pca_sample_step=5)
    def backend(name):return CppGPBackend(panel,a.engine_binary,a.pca_binary,out/name,ranks=a.ranks,device=a.device,mpiexec=a.mpiexec)
    live=backend('fixed');frames,y=live.frames();live.validate_inputs(frames,y,config)
    expressions=['f0',('mean','f0',3),('std','f0',3),('lag','f1',2),('ema','f0',3),('+','f0','f1'),('correlation','f0','f1',3),('rank','f0')]
    seed=gp._numpy_pca_seed(2026);evaluation=gp._evaluation_config(config)
    got=live.evaluate(expressions,evaluation,config,seed);max_stats=0.;max_angle=0.;records=[]
    for expression,(candidate,sig) in zip(expressions,got):
        expected,values=_evaluate_with_values(expression,frames,y,evaluation,ExpressionCache())
        assert (candidate is None)==(expected is None),expression
        if candidate is None:continue
        for name in ('rank_ic','valid_ratio','missing_ratio','fitness','segment_rank_ic_std','same_sign_ratio'):
            delta=abs(getattr(candidate,name)-getattr(expected,name));max_stats=max(max_stats,delta);assert delta<2e-5,(expression,name,delta)
        try:reference=factor_signature(values,sample_step=config.pca_sample_step,power_iterations=config.pca_power_iterations,seed=seed)
        except ValueError:reference=None
        if reference is not None and np.isfinite(signature_similarity(reference,reference)):
            assert sig is not None;delta=angle_error(sig,reference);max_angle=max(max_angle,delta);assert delta<2e-5,(expression,delta)
        if sig is not None:records.append(QDRecord(str(expression),expression,candidate.abs_rank_ic,1,1,sig))
    oracle=PCAQDArchive(.9,.002);actual=PCAQDArchive(.9,.002)
    if records:
        first=records[0];seed_record=QDRecord('historical',first.expression,first.abs_rank_ic,1,1,first.signature)
        oracle._records['historical']=seed_record;actual._records['historical']=seed_record
    expected_decisions=oracle.consider_batch(records,backend='cpu');actual_decisions=live.consider(actual,records,config,seed)
    assert [(d.action,d.closest_key,d.replaced_key) for d in expected_decisions]==[(d.action,d.closest_key,d.replaced_key) for d in actual_decisions]
    assert list(oracle)==list(actual)
    profile=pd.DataFrame(dict(feature=list(frames),rank_ic=np.linspace(.08,-.04,F)))
    # Make any accidental use of the old numerical paths fail immediately.
    def run(checkpoint,impl,callback=None):
        with patch.object(gp,'_evaluate',side_effect=AssertionError('Python evaluator fallback')),patch.object(gp,'factor_signature',side_effect=AssertionError('Python PCA fallback')),patch.object(gp,'evaluate_expression',side_effect=AssertionError('Python panel fallback')),patch.object(PCAQDArchive,'consider_batch',side_effect=AssertionError('Python QD fallback')):
            return gp.run_hierarchical_evolution(frames,y,profile,config,2026,checkpoint_dir=checkpoint,batch_backend=impl,progress_callback=callback)
    full=run(out/'full_checkpoint',backend('full_backend'))
    generations=sum(row['generation']>0 for row in full.history)
    assert [x.depth for x in full.stages]==[1,2,3] and generations==3 and all(s.gene_pool for s in full.stages)
    def interrupt(event):
        if event['event']=='generation_completed':raise RuntimeError('intentional checkpoint interrupt')
    try:run(out/'resume_checkpoint',backend('resume_backend'),interrupt);raise AssertionError('interrupt not reached')
    except RuntimeError as error:assert 'intentional checkpoint interrupt' in str(error)
    resumed=run(out/'resume_checkpoint',backend('resume_backend'))
    assert [r.key for r in full.archive_records]==[r.key for r in resumed.archive_records]
    assert full.history==resumed.history
    assert all(np.array_equal(x.signature,y.signature) for x,y in zip(full.archive_records,resumed.archive_records))
    bad_identity=CppGPBackend(panel,a.engine_binary,a.pca_binary,out/'different_backend',ranks=1 if a.ranks!=1 else 2,device=a.device,mpiexec=a.mpiexec)
    try:run(out/'resume_checkpoint',bad_identity);raise AssertionError('different backend checkpoint accepted')
    except RuntimeError as error:assert 'backend' in str(error)
    report=dict(verified=True,live_cpp_engine=True,live_cpp_pca=True,live_cpp_qd=True,python_numerical_fallback_disabled=True,
        stages=[s.depth for s in full.stages],generations=generations,history_rows=len(full.history),checkpoint_resume_exact=True,changed_backend_rejected=True,
        fixed_expression_count=len(expressions),max_candidate_stat_error=max_stats,max_signature_angle_error=max_angle,
        final_archive=len(full.archive_records),ranks=a.ranks,device=a.device)
    (out/'SUCCESS.json').write_text(json.dumps(report,indent=2)+'\n');(out/'SUCCESS').touch();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
