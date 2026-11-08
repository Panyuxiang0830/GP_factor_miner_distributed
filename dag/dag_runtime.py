#!/usr/bin/env python3
"""Serial DAG-planner scaling with explicit V/E and isolated planning timings.
Times Python graph construction and logical scheduling; never factor kernels.
"""
import argparse
import csv
import json
import platform
from pathlib import Path
import statistics
import sys
import time
from memory_sim import graph, simulate


def graph_metrics(nodes, children):
    # Operands are references: a*a contributes two references, one distinct edge.
    return {'vertices':len(nodes),
            'edges_unique':sum(len(set(cs)) for cs in children),
            'operand_references':sum(len(cs) for cs in children),
            'operator_vertices':sum(n['op']!='leaf' for n in nodes)}


def timed(fn,repeat,warmup):
    for _ in range(warmup): fn()
    samples=[];value=None
    for _ in range(repeat):
        start=time.perf_counter();value=fn();samples.append(time.perf_counter()-start)
    return value,{'median':statistics.median(samples),'min':min(samples),'max':max(samples),'samples':samples}


def benchmark(batch, repeat=3, warmup=1, chunks=(64,256,1024)):
    if repeat<1 or warmup<0 or any(c<1 for c in chunks):raise ValueError('invalid repeat/warmup/chunks')
    built,build_t=timed(lambda:graph(batch),repeat,warmup)
    nodes,children,roots,accepted,rejected=built
    metrics=graph_metrics(nodes,children)
    results=[]
    configs=[('level_keep',256),('level_early',256)]+[('level_chunk',c) for c in chunks]+[('greedy',1)]
    for policy,chunk in configs:
        r,t=timed(lambda:simulate(nodes,children,roots,policy,chunk),repeat,warmup)
        # simulate allocates all per-run state anew; construction isn't in this scope.
        r.update(metrics)
        r.update(input_expressions=len(batch),accepted=accepted,rejected=rejected,
                 graph_build_seconds_median=build_t['median'],
                 schedule_seconds_median=t['median'],schedule_seconds_min=t['min'],
                 schedule_seconds_max=t['max'],schedule_seconds_samples=t['samples'],
                 repeat=repeat,warmup=warmup)
        results.append(r)
    return {'metrics':metrics,'graph_build_timing':build_t,'rows':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sizes',default='1000,5000,10000,30000')
    p.add_argument('--chunks',default='64,256,1024');p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1)
    args=p.parse_args();batch=json.loads(args.batch.read_text())
    if not isinstance(batch,list) or not batch:p.error('nonempty JSON list required')
    sizes=sorted(set([int(s) for s in args.sizes.split(',') if int(s)<=len(batch)]+[len(batch)]))
    chunks=tuple(int(s) for s in args.chunks.split(','))
    if args.repeat<1 or args.warmup<0 or min(sizes+list(chunks))<1:p.error('invalid positive parameters')
    reports=[];rows=[]
    for size in sizes:
        report=benchmark(batch[:size],args.repeat,args.warmup,chunks);reports.append(report)
        for r in report['rows']:
            rows.append(r)
            print(f"B={size} V={r['vertices']} E={r['edges_unique']} refs={r['operand_references']} "
                  f"{r['policy']} C={r['chunk_size']} peak={r['peak_live_panels']} "
                  f"build_s={r['graph_build_seconds_median']:.6f} plan_s={r['schedule_seconds_median']:.6f}")
    args.out.mkdir(parents=True,exist_ok=True)
    output={'schema_version':1,'environment':{'system':platform.system(),'release':platform.release(),
            'machine':platform.machine(),'python':platform.python_version()},
            'timing_scope':{'build':'Validation + Python normalization + DAG + dependency adjacency; excludes JSON IO.',
                            'schedule':'Fresh simulated scheduling + reference/peak tracking + correctness checks; excludes build, factor computation and MPI.'},
            'edge_definition':'E counts distinct child-parent pairs. Operand references count repeated edges such as a*a twice.',
            'reports':reports}
    (args.out/'dag_runtime.json').write_text(json.dumps(output,indent=2))
    fields=[k for k in rows[0] if k!='schedule_seconds_samples']
    with (args.out/'dag_runtime.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:r[k] for k in fields})
if __name__=='__main__':main()
