"""Opt-in live C++/MPI factor evaluation, PCA and QD for the original GP loop."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import os
import signal
import numpy as np
import pandas as pd
from archive_io import load_archive,save_archive
from gp_factor_miner.expressions import expression_key,normalize_expression
from gp_factor_miner.gp_engine import Candidate
from gp_factor_miner.pca_qd import QDRecord,QDDecision,signature_similarity


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CppGPBackend:
    """One bounded MPI batch per GP chunk; original Python owns evolution/checkpoints."""
    def __init__(self,panel_dir,engine_binary,pca_binary,work_dir,*,dates=None,stocks=None,ranks=1,
                 device='cpu',precision='double',method='power',engine_chunk=16,dag_chunk=32,H0=0,
                 mpiexec='mpirun',mpi_map='slot',timeout=1800):
        self.panel_dir=Path(panel_dir).resolve();self.meta=json.loads((self.panel_dir/'meta.json').read_text())
        self.T=dates or self.meta['T'];self.N=stocks or self.meta['N']
        if not 2<=self.T<=self.meta['T'] or not 2<=self.N<=self.meta['N'] or not 1<=ranks<=self.T:raise ValueError('invalid live backend shape/ranks')
        self.engine_binary=Path(engine_binary).resolve();self.pca_binary=Path(pca_binary).resolve()
        self.work_dir=Path(work_dir).resolve();self.work_dir.mkdir(parents=True,exist_ok=True)
        self.ranks=ranks;self.device=device;self.precision=precision;self.method=method
        if device not in ('cpu','cuda') or precision not in ('float','double') or method not in ('power','lanczos') or (device=='cuda' and method!='power'):raise ValueError('invalid backend solver/device')
        if engine_chunk<1 or dag_chunk<0 or H0<0 or timeout<1:raise ValueError('invalid backend limits')
        self.engine_chunk,self.dag_chunk,self.H0=engine_chunk,dag_chunk,H0
        self.mpiexec,self.mpi_map,self.timeout=mpiexec,mpi_map,timeout
        self.calls=json.loads((self.work_dir/'calls.json').read_text()) if (self.work_dir/'calls.json').exists() else []
        self.serial=0;self._initials={};self._config=None;self.threads=int(os.environ.get('OMP_NUM_THREADS','1'))
        self.identity=dict(kind='cpp-mpi-pca-qd-v1',panel_dir=str(self.panel_dir),meta_sha256=sha(self.panel_dir/'meta.json'),
            dates=self.T,stocks=self.N,engine_sha256=sha(self.engine_binary),pca_sha256=sha(self.pca_binary),
            ranks=ranks,threads=self.threads,device=device,precision=precision,method=method,engine_chunk=engine_chunk,dag_chunk=dag_chunk,H0=H0)
        # Exact panel consistency is checked once before any generation runs.
        identity_path=self.work_dir/'backend_identity.json'
        if identity_path.exists():
            previous=json.loads(identity_path.read_text());previous.pop('panel_files',None)
            if previous!=self.identity:raise ValueError('backend work directory belongs to different binaries/data/settings')
        else:identity_path.write_text(json.dumps(self.identity,indent=2)+'\n')

    def frames(self,fields=None):
        fields=list(fields or self.meta['fields']);unknown=set(fields)-set(self.meta['fields'])
        if unknown:raise ValueError('unknown input fields: '+str(sorted(unknown)))
        def frame(path):
            values=np.memmap(path,mode='r',dtype='<f4',shape=(self.meta['T'],self.meta['N']))
            return pd.DataFrame(values[:self.T,:self.N],index=self.meta['dates'][:self.T],columns=self.meta['assets'][:self.N],copy=False)
        return {name:frame(self.panel_dir/'fields'/f'{name}.f32') for name in fields},frame(self.panel_dir/'target.f32')

    def validate_inputs(self,data_dict,target,config):
        if config.pca_similarity_threshold!=.90 or config.qd_replacement_rankic_margin!=.002:raise ValueError('C++ QD currently requires thresholds 0.90 and 0.002')
        if config.pca_sample_step<1 or (self.T+config.pca_sample_step-1)//config.pca_sample_step<2:raise ValueError('not enough sampled dates')
        if self.device=='cuda' and self.N>8192:raise ValueError('GPU rank width exceeds 8192')
        expected,y=self.frames(data_dict.keys())
        for key,x in [*data_dict.items(),('__target__',target)]:
            ref=y if key=='__target__' else expected[key]
            if list(map(str,x.index))!=list(map(str,ref.index)) or list(map(str,x.columns))!=list(map(str,ref.columns)) or not np.array_equal(x.to_numpy(),ref.to_numpy(),equal_nan=True):
                raise ValueError('Python GP and C++ panels differ: '+key)
        files=[self.panel_dir/'target.f32',*(self.panel_dir/'fields'/f'{name}.f32' for name in sorted(data_dict))]
        self.identity['panel_files']={str(p.relative_to(self.panel_dir)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files}
        identity_path=self.work_dir/'backend_identity.json';previous=json.loads(identity_path.read_text())
        if 'panel_files' in previous and previous!=self.identity:raise ValueError('raw panel files changed since the previous backend run')
        identity_path.write_text(json.dumps(self.identity,indent=2)+'\n')
        self._config=config

    def _call(self,kind,binary,options,ranks):
        while True:
            self.serial+=1;run=self.work_dir/f'{kind}_{self.serial:06d}'
            if not run.exists() and not run.with_suffix('.log').exists():break
        mapping=self.mpi_map+(':PE='+str(self.threads) if self.threads>1 else '')
        prefix=[self.mpiexec,'-np',str(ranks),'--map-by',mapping,'--bind-to','core']
        cmd=prefix+[str(binary),*map(str,options),'--out',str(run)]
        started=time.perf_counter()
        with run.with_suffix('.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
            if code:raise subprocess.CalledProcessError(code,cmd)
        self.calls.append(dict(kind=kind,wall_seconds=time.perf_counter()-started,run=str(run),ranks=ranks))
        (self.work_dir/'calls.json').write_text(json.dumps(self.calls,indent=2)+'\n')
        return run

    def _initial(self,seed):
        if seed not in self._initials:
            path=self.work_dir/f'initial_{seed}.f8';values=np.random.default_rng(seed).normal(size=self.N).astype('<f8')
            if path.exists() and path.read_bytes()!=values.tobytes():raise ValueError('initial vector changed')
            if not path.exists():values.tofile(path)
            self._initials[seed]=path
        return self._initials[seed]

    def evaluate(self,expressions,evaluation_config,config,pca_seed):
        if not expressions:return []
        normalized=[normalize_expression(x) for x in expressions]
        request=dict(panel_dir=str(self.panel_dir),dates=self.T,stocks=self.N,sample_step=config.pca_sample_step,
            steps=config.pca_power_iterations,method=self.method,device=self.device,precision=self.precision,
            initial=str(self._initial(pca_seed)),engine_chunk=self.engine_chunk,dag_chunk=self.dag_chunk,H0=self.H0,
            fitness=asdict(evaluation_config),expressions=[dict(key=expression_key(x),expression=x) for x in normalized])
        request_path=self.work_dir/f'request_{self.serial+1:06d}_{time.time_ns()}.json';request_path.write_text(json.dumps(request,allow_nan=False)+'\n')
        run=self._call('evaluate',self.engine_binary,('--request',request_path),self.ranks)
        rows=json.loads((run/'candidates.json').read_text());m=(self.T+config.pca_sample_step-1)//config.pca_sample_step
        signatures=np.fromfile(run/'signatures.f32',dtype='<f4').reshape(len(normalized),m)
        if len(rows)!=len(normalized):raise RuntimeError('C++ candidate count mismatch')
        result=[]
        for i,(expression,row,s) in enumerate(zip(normalized,rows,signatures)):
            if row['index']!=i or row['key']!=expression_key(expression):raise RuntimeError('C++ candidate key/order mismatch')
            if not row['valid']:result.append((None,None));continue
            def number(key):
                x=row[key]
                if x is None or not np.isfinite(x):raise RuntimeError('nonfinite C++ candidate statistic: '+key)
                return float(x)
            c=Candidate(expression=expression,rank_ic=number('rank_ic'),abs_rank_ic=abs(number('rank_ic')),
                valid_ratio=number('valid_ratio'),missing_ratio=number('missing_ratio'),depth=int(row['depth']),nodes=int(row['nodes']),
                fitness=number('fitness'),segment_rank_ic_std=number('segment_rank_ic_std'),same_sign_ratio=number('same_sign_ratio'),
                segment_rank_ics=tuple(float(x) if x is not None else float('nan') for x in row['segment_rank_ics']))
            if not np.isfinite(s).all():raise RuntimeError('nonfinite C++ signature')
            result.append((c,s.copy() if np.isfinite(signature_similarity(s,s)) else None))
        return result

    def consider(self,archive,records,config,pca_seed):
        if not records:return []
        m=(self.T+config.pca_sample_step-1)//config.pca_sample_step
        while True:
            inputs=self.work_dir/f'qd_input_{self.serial+1:06d}_{time.time_ns()}'
            if not inputs.exists():inputs.mkdir();break
        seed=inputs/'archive.json';save_archive(seed,{key:(r.abs_rank_ic,r.signature) for key,r in archive._records.items()},m)
        np.asarray([r.signature for r in records],dtype='<f4').tofile(inputs/'signatures.f32')
        manifest=dict(rows=m,cols=self.N,initial=str(self._initial(pca_seed)),records=[dict(key=r.key,quality=r.abs_rank_ic,candidate=True) for r in records])
        path=inputs/'manifest.json';path.write_text(json.dumps(manifest,allow_nan=False)+'\n')
        # QD is sequential at rank 0; CUDA still performs batched similarity.
        run=self._call('qd',self.pca_binary,('--manifest',path,'--signature-input',inputs/'signatures.f32',
            '--archive-input',seed,'--device',self.device,'--precision',self.precision,'--chunk',config.pca_qd_block_size),1)
        decisions=json.loads((run/'decisions.json').read_text());saved=load_archive(run/'archive.json',m)
        if len(decisions)!=len(records):raise RuntimeError('C++ QD result count mismatch')
        current=dict(archive._records);answer=[]
        for i,(record,d) in enumerate(zip(records,decisions)):
            if d['signature_index']!=i or d['key']!=record.key or d['action'] not in ('admit','replace','reject'):raise RuntimeError('C++ QD order/action mismatch')
            if d['action']=='replace':
                if d['replaced_key'] not in current:raise RuntimeError('C++ QD replacement target absent')
                current.pop(d['replaced_key'])
            if d['action'] in ('admit','replace'):current[record.key]=record
            answer.append(QDDecision(d['action'],float(d['similarity']) if d['similarity'] is not None else float('nan'),d['closest_key'],d['replaced_key']))
        if sorted(current)!=sorted(saved) or any(saved[k][0]!=r.abs_rank_ic or not np.array_equal(saved[k][1],r.signature) for k,r in current.items()):raise RuntimeError('C++/Python final archive content mismatch')
        archive._records=current
        return answer
