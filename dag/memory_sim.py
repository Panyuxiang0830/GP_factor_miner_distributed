#!/usr/bin/env python3
"""S2/G4: logical live-panel simulation. No factor evaluation or MPI execution."""
import argparse
from collections import Counter, defaultdict
import csv
import heapq
import json
from pathlib import Path
from dag_stats import build, expression_key


def graph(batch):
    model=build(batch)
    mapping,root_keys,accepted,rejected,*_=model
    keys=list(mapping); ids={k:i for i,k in enumerate(keys)}
    children=[]
    for key in keys:
        e=json.loads(key)
        children.append([] if isinstance(e,str) else
                        [ids[expression_key(c)] for c in e[1:] if isinstance(c,(str,list))])
    return list(mapping.values()),children,{ids[k] for k in root_keys},len(accepted),len(rejected)


def simulate(nodes,children,roots,policy,chunk=256,emit=None):
    n=len(nodes);pending=[x['parents'] for x in nodes]
    live=set();computed=set();peak=0;groups=0
    def execute(group,keep=False):
        nonlocal peak,groups
        for i in group:
            if i in computed or not all(c in live for c in children[i]):
                raise AssertionError('invalid schedule / released input')
        # All outputs allocated before the chunk computes, matching level engine allocation.
        if emit:
            emit({'op':'begin_group','nodes':list(group)})
            for i in group:
                emit({'op':'load' if not children[i] else 'compute','node':i})
            if not keep:
                for i in group:
                    if i in roots:emit({'op':'consume_root','node':i})
        live.update(group);peak=max(peak,len(live));computed.update(group)
        if any(children[i] for i in group):groups+=1
        for i in group:
            if pending[i]==0 and not (keep and i in roots):
                live.remove(i)
                if emit:emit({'op':'release','node':i})
        for i in group:
            for c in children[i]:
                pending[c]-=1
                if pending[c]<0:raise AssertionError('negative reference count')
                if pending[c]==0 and not (keep and c in roots):
                    live.remove(c)
                    if emit:emit({'op':'release','node':c})
        if emit:emit({'op':'end_group'})
    if policy in ('level_keep','level_early','level_chunk'):
        levels=defaultdict(list)
        for i,x in enumerate(nodes):levels[x['level']].append(i)
        for level,group in sorted(levels.items()):
            # Load all fields once for level policies. Chunk only operator layers.
            width=chunk if policy=='level_chunk' and level>0 else len(group)
            for start in range(0,len(group),width):execute(group[start:start+width],policy=='level_keep')
    elif policy=='greedy':
        parents=[set() for _ in nodes]
        for i,cs in enumerate(children):
            for c in cs:parents[c].add(i)
        indegree=[len(set(cs)) for cs in children]
        ready=set();heap=[];version=[0]*n
        def push(i):
            if i not in ready:return
            version[i]+=1
            freed=sum(pending[c]==multiplicity for c,multiplicity in Counter(children[i]).items())
            # Prefer largest immediate release, then an output with no future consumers.
            heapq.heappush(heap,(-freed,pending[i]>0,i,version[i]))
        for i,d in enumerate(indegree):
            if d==0:ready.add(i);push(i)
        while heap:
            _,_,i,v=heapq.heappop(heap)
            if i not in ready or v!=version[i]:continue
            ready.remove(i);execute([i])
            for p in parents[i]:
                indegree[p]-=1
                if indegree[p]==0:ready.add(p);push(p)
            # Consuming a shared input changes the score of other ready consumers.
            for c in set(children[i]):
                for p in parents[c]:push(p)
    else:raise ValueError(policy)
    if len(computed)!=n:raise AssertionError('incomplete execution')
    if policy=='level_keep':
        if live!=roots:raise AssertionError('baseline must retain exactly the roots')
        if emit:
            for i in sorted(roots):
                emit({'op':'consume_root','node':i})
                emit({'op':'release','node':i})
        live.clear()  # score retained roots, then release
    if live or any(pending):raise AssertionError('leaked panels or references')
    return {'policy':policy,'chunk_size':chunk if policy=='level_chunk' else None,
            'peak_live_panels':peak,'operator_groups_proxy':groups,'computed_nodes':n}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sizes',default='1000,5000,10000,30000')
    p.add_argument('--chunks',default='64,256,1024')
    p.add_argument('--ranks',type=int,default=1)
    p.add_argument('--dates',type=int,default=1461);p.add_argument('--stocks',type=int,default=5876)
    args=p.parse_args();batch=json.loads(args.batch.read_text())
    if not isinstance(batch,list) or not batch:p.error('nonempty JSON list required')
    sizes=sorted(set([int(s) for s in args.sizes.split(',') if int(s)<=len(batch)]+[len(batch)]))
    chunks=[int(c) for c in args.chunks.split(',')]
    if min(sizes+chunks+[args.ranks,args.dates,args.stocks])<1:p.error('positive parameters required')
    rows=[]
    for size in sizes:
        nodes,children,roots,accepted,rejected=graph(batch[:size])
        configs=[('level_keep',256),('level_early',256)]+[('level_chunk',c) for c in chunks]+[('greedy',1)]
        for policy,c in configs:
            r=simulate(nodes,children,roots,policy,c)
            r.update(input_expressions=size,accepted=accepted,rejected=rejected,ranks=args.ranks,
                     # Maximum own-date block under equal date partition; halo excluded.
                     estimated_own_panel_GiB_per_rank=r['peak_live_panels']*((args.dates+args.ranks-1)//args.ranks)*args.stocks*4/2**30)
            rows.append(r)
            print(f"B={size} {policy:12s} C={r['chunk_size']} peak={r['peak_live_panels']} "
                  f"panel_GiB/rank={r['estimated_own_panel_GiB_per_rank']:.3f} groups_proxy={r['operator_groups_proxy']}")
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'memory_sim.json').write_text(json.dumps({'schema_version':1,
        'model':'Equal-size float32 own-date panels. All nodes once. Root metrics consumed before release.',
        'exclusions':['halo bytes','pool-retained buffers','target ranks','operator scratch',
                      'actual MPI message counts and EMA communication','PCA workspaces','timing prediction'],
        'sync_caveat':'groups_proxy counts scheduled operator groups, NOT measured MPI synchronization or messages; greedy is a one-node-per-group conservative plan.',
        'rows':rows},indent=2))
    with (args.out/'memory_sim.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
