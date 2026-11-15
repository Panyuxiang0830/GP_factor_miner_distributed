#!/usr/bin/env python3
"""Run the original three-stage GP with the live C++/MPI/CUDA batch backend."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/src'))
import numpy as np
from gp_factor_miner.hierarchical_config import HierarchicalGPConfig
from gp_factor_miner.hierarchical_engine import run_hierarchical_evolution
from gp_factor_miner.raw_features import profile_raw_panels
from gp_bridge import CppGPBackend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-dir',type=Path,required=True);p.add_argument('--engine-binary',type=Path,required=True);p.add_argument('--pca-binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--fields',nargs='+')
    p.add_argument('--dates',type=int);p.add_argument('--stocks',type=int);p.add_argument('--ranks',type=int,default=1);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--precision',choices=('float','double'),default='double');p.add_argument('--method',choices=('power','lanczos'),default='power')
    p.add_argument('--population',type=int,default=24);p.add_argument('--generations',type=int,default=1);p.add_argument('--gene-pool',type=int,default=12)
    p.add_argument('--warm-start-multiplier',type=int,default=2)
    p.add_argument('--sample-step',type=int,default=5);p.add_argument('--steps',type=int,default=4);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--windows',type=int,nargs='+',default=[1,5,10]);p.add_argument('--progress-chunk',type=int,default=24)
    p.add_argument('--engine-chunk',type=int,default=16);p.add_argument('--dag-chunk',type=int,default=32);p.add_argument('--H0',type=int,default=0)
    p.add_argument('--mpiexec',default='mpirun');p.add_argument('--mpi-map',default='slot');a=p.parse_args()
    out=a.out.resolve();runs=Path(__file__).resolve().parent/'runs'
    if not out.is_relative_to(runs.resolve()) or (out.exists() and not a.resume) or (a.resume and not out.exists()):p.error('use a new private pca/runs child, or explicitly --resume an existing run')
    if min(a.population,a.gene_pool,a.threads,a.progress_chunk,a.warm_start_multiplier)<1 or a.generations<0:p.error('invalid GP size')
    os.environ['OMP_NUM_THREADS']=str(a.threads);os.environ['OPENBLAS_NUM_THREADS']='1'
    out.mkdir(parents=True,exist_ok=a.resume)
    settings={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('resume','out')}
    config_path=out/'RUN_CONFIG.json'
    if config_path.exists() and json.loads(config_path.read_text())!=settings:p.error('resume settings changed')
    config_path.write_text(json.dumps(settings,indent=2)+'\n')
    backend=CppGPBackend(a.panel_dir,a.engine_binary,a.pca_binary,out/'backend',dates=a.dates,stocks=a.stocks,ranks=a.ranks,
        device=a.device,precision=a.precision,method=a.method,engine_chunk=a.engine_chunk,dag_chunk=a.dag_chunk,H0=a.H0,mpiexec=a.mpiexec,mpi_map=a.mpi_map)
    data,target=backend.frames(a.fields)
    config=replace(HierarchicalGPConfig(),population_size=a.population,generations_per_stage=a.generations,
        warm_start_multiplier=a.warm_start_multiplier,gene_pool_size=min(a.gene_pool,a.population),parent_pool_size=min(a.gene_pool,a.population),
        windows=tuple(a.windows),pca_sample_step=a.sample_step,pca_power_iterations=a.steps,progress_chunk_size=a.progress_chunk,max_workers=1,random_seed=a.seed)
    started=time.perf_counter();profile_path=out/'feature_profile.csv'
    if profile_path.exists():
        import pandas as pd
        profile=pd.read_csv(profile_path)
    else:profile=profile_raw_panels(data,target);profile.to_csv(profile_path,index=False)
    if set(profile['feature'])!=set(data):raise ValueError('feature profile does not match the active panels')
    setup_seconds=time.perf_counter()-started
    def progress(event):
        with (out/'progress.jsonl').open('a') as log:log.write(json.dumps(event,default=lambda x:x.item() if isinstance(x,np.generic) else list(x) if isinstance(x,(tuple,set)) else str(x))+'\n')
        if event['event'] in ('generation_completed','stage_completed'):print(event,flush=True)
    started=time.perf_counter()
    result=run_hierarchical_evolution(data,target,profile,config,a.seed,progress_callback=progress,checkpoint_dir=out/'checkpoint',batch_backend=backend)
    wall=time.perf_counter()-started
    from archive_io import save_archive
    dimension=(backend.T+a.sample_step-1)//a.sample_step
    save_archive(out/'archive.json',{r.key:(r.abs_rank_ic,r.signature) for r in result.archive_records},dimension)
    rows=[dict(depth=s.depth,gene_pool=len(s.gene_pool),generations=s.generations_completed) for s in result.stages]
    assert [x['depth'] for x in rows]==[1,2,3] and all(x['gene_pool'] for x in rows)
    report=dict(completed=True,live_cpp_engine=True,live_cpp_pca=True,live_cpp_qd=True,device=a.device,ranks=a.ranks,
        T=backend.T,N=backend.N,fields=list(data),stages=rows,archive_size=len(result.archive_records),evaluated=result.evaluated_count,
        generation_events=sum(row['generation']>0 for row in result.history),history_rows=len(result.history),setup_seconds=setup_seconds,evolution_wall_seconds=wall,backend_calls=len(backend.calls),
        scope='Actual original three-stage GP loop; subprocess launches, checkpointing and C++ engine/PCA/QD included in evolution wall. Size and generation budgets explicit; not a full-size production speedup claim.')
    (out/'SUCCESS.json').write_text(json.dumps(report,indent=2)+'\n');(out/'SUCCESS').touch();print(json.dumps(report,indent=2))


if __name__=='__main__':main()
