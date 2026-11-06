#!/usr/bin/env python3
"""S3: persistent panel cache simulation. Fixed-trace and subtree-pruned models.
Only structure is processed. Reported saved seconds are cost-model estimates.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import heapq
import json
import math
from pathlib import Path
from dag_stats import build, expression_key

POLICIES=('lru','lfu','cost','belady')
FALLBACKS={'rank':'std','zscore':'mean','delta':'lag'}


def load_costs(path,column='ns_per_cell_1thread'):
    costs={}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            value=float(row[column])
            if not math.isfinite(value) or value<0:raise ValueError('invalid operator cost')
            costs[row['op']]=value
    for op,source in FALLBACKS.items():
        if op not in costs:
            if source not in costs:raise ValueError('missing fallback source '+source)
            costs[op]=costs[source]
    return costs


def load_batches(paths):
    batches=[];hooks=[];seen={}
    for path in paths:
        path=Path(path)
        if path.suffix=='.jsonl':
            records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            values=json.loads(path.read_text())
            if not isinstance(values,list):raise ValueError('batch JSON must be a list')
            records=[{'type':'batch','members':values}]
        for record in records:
            # Do NOT collapse all events of a generation/phase: many are legitimate.
            if 'event_id' in record:
                token=str(record['event_id']);encoded=json.dumps(record,sort_keys=True)
                if token in seen:
                    if seen[token]!=encoded:raise ValueError('conflicting repeated event_id')
                    continue
                seen[token]=encoded
            kind=record.get('type')
            if kind=='batch':
                values=record.get('members')
                if not isinstance(values,list):raise ValueError('batch members must be a list')
                batches.append({'members':values,'metadata':{k:record.get(k) for k in ('depth','stage','generation','phase')}})
            elif kind in ('parents','population','gene_pool'):
                # Hooks retain ordering and membership for future policies, not used now.
                hooks.append({'before_batch':len(batches),'type':kind,'members':record.get('members',[]),
                              'metadata':{k:record.get(k) for k in ('depth','stage','generation','phase')}})
            else:raise ValueError('unsupported trace record type')
    if not batches:raise ValueError('no batches')
    return batches,hooks


def prepare(batches,costs):
    registry={};nodes=[];prepared=[]
    for item in batches:
        mapping,root_keys,accepted,rejected,*_=build(item['members'])
        local=[]
        for key,node in mapping.items():
            if key not in registry:
                e=json.loads(key)
                children=[] if isinstance(e,str) else [registry[expression_key(c)] for c in e[1:] if isinstance(c,(list,str))]
                closure=set()
                for c in children:closure.update(nodes[c]['closure'])
                idx=len(nodes);op=node['op']
                own=0.0 if op=='leaf' else costs[op]
                if op!='leaf':closure.add(idx)
                registry[key]=idx
                nodes.append({'op':op,'children':children,'own_ns':own,'closure':closure,
                              'subtree_ns':own+sum(nodes[j]['own_ns'] for j in closure if j!=idx)})
            local.append(registry[key])
        operators=[i for i in local if nodes[i]['op']!='leaf']
        prepared.append({'operators':operators,'roots':[registry[k] for k in root_keys],
                         'accepted':len(accepted),'rejected':len(rejected),
                         'input_expressions':len(item['members']),'metadata':item['metadata']})
    return nodes,prepared


class Cache:
    """Equal-size entries; deterministic LRU ties, frequency since admission."""
    def __init__(self,capacity,policy,nodes):
        self.capacity=capacity;self.policy=policy;self.nodes=nodes
        self.entries={};self.heap=[];self.clock=0;self.evictions=0
    def priority(self,key,entry):
        frequency,tick,future=entry
        if self.policy=='lru':value=tick
        elif self.policy=='lfu':value=frequency
        elif self.policy=='cost':value=self.nodes[key]['subtree_ns']*frequency
        elif self.policy=='belady':value=-future
        else:raise ValueError(self.policy)
        return value,tick,key
    def rebuild(self):
        self.heap=[self.priority(k,v) for k,v in self.entries.items()];heapq.heapify(self.heap)
    def refresh_future(self,fn):
        for k,(f,t,_) in list(self.entries.items()):self.entries[k]=(f,t,fn(k))
        self.rebuild()
    def touch(self,key,future=math.inf):
        if self.nodes[key]['op']=='leaf':raise AssertionError('leaf cache insertion')
        self.clock+=1
        if self.capacity==0:return
        if key not in self.entries and len(self.entries)>=self.capacity:
            while True:
                priority=heapq.heappop(self.heap);victim=priority[2]
                if victim in self.entries and self.priority(victim,self.entries[victim])==priority:
                    del self.entries[victim];self.evictions+=1;break
        frequency=self.entries.get(key,(0,0,0))[0]+1
        self.entries[key]=(frequency,self.clock,future)
        heapq.heappush(self.heap,self.priority(key,self.entries[key]))
        if len(self.heap)>4*max(1,len(self.entries))+32:self.rebuild()
        if len(self.entries)>self.capacity:raise AssertionError('cache budget exceeded')


def demands(roots,nodes,snapshot):
    """Unique requested hits and computed operators. Cache hit cuts recursion."""
    hits=set();computed=set();seen=set();stack=list(roots)
    while stack:
        i=stack.pop()
        if i in seen:continue
        seen.add(i)
        if nodes[i]['op']=='leaf':continue
        if i in snapshot:hits.add(i)
        else:
            computed.add(i);stack.extend(nodes[i]['children'])
    return hits,computed


def replay(nodes,batches,capacity,policy,mode,cell_scale):
    cache=Cache(capacity,policy,nodes);rows=[]
    flat=[i for b in batches for i in b['operators']]
    nexts=[math.inf]*len(flat);last={}
    for p in range(len(flat)-1,-1,-1):nexts[p]=last.get(flat[p],math.inf);last[flat[p]]=p
    batch_uses=defaultdict(list)
    for b,item in enumerate(batches):
        for i in item['operators']:batch_uses[i].append(b)
    offset=0
    for b,item in enumerate(batches):
        ops=item['operators'];base=sum(nodes[i]['own_ns'] for i in ops)
        start_evictions=cache.evictions
        if mode=='fixed_trace':
            hits=set();computed=set()
            for j,i in enumerate(ops):
                (hits if i in cache.entries else computed).add(i)
                cache.touch(i,nexts[offset+j])
        elif mode=='dag_pruned':
            # Snapshot values can be consumed during this batch even if persistent
            # entries are subsequently evicted at the boundary. Working memory
            # required for batch execution is not charged to persistent cache M.
            hits,computed=demands(item['roots'],nodes,set(cache.entries))
            def next_batch(i):
                uses=batch_uses[i];p=bisect_right(uses,b)
                return uses[p] if p<len(uses) else math.inf
            if policy=='belady':cache.refresh_future(next_batch)
            active=hits|computed
            for i in ops:
                if i in active:cache.touch(i,next_batch(i) if policy=='belady' else math.inf)
        else:raise ValueError(mode)
        actual=sum(nodes[i]['own_ns'] for i in computed)
        saved=base-actual
        if not computed.issubset(set(ops)) or saved < -1e-10*max(1.0,base):raise AssertionError('invalid modeled savings')
        saved=max(0.0,saved)
        requests=len(hits)+len(computed)
        rows.append({'batch_index':b+1,'input_expressions':item['input_expressions'],
            'accepted':item['accepted'],'rejected':item['rejected'],
            'model':mode,'policy':'future_batch_reference' if mode=='dag_pruned' and policy=='belady' else policy,
            'capacity_panels':capacity,'baseline_operator_nodes':len(ops),
            'cache_hits':len(hits),'computed_operator_nodes':len(computed),
            'pruned_descendant_nodes':len(ops)-requests,
            'request_hit_rate':len(hits)/requests if requests else 0.0,
            'avoided_operator_fraction':1-len(computed)/len(ops) if ops else 0.0,
            'baseline_cost_seconds':base*cell_scale/1e9,
            'computed_cost_seconds':actual*cell_scale/1e9,
            'saved_cost_seconds':saved*cell_scale/1e9,
            'saved_cost_fraction':saved/base if base else 0.0,
            'resident_panels_after_batch':len(cache.entries),
            'evictions_in_batch':cache.evictions-start_evictions})
        rows[-1].update({k:item['metadata'].get(k) for k in ('depth','stage','generation','phase')})
        offset+=len(ops)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batches',nargs='+',type=Path);p.add_argument('--costs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--budgets-gib',default='0,1,2,4,8,16')
    p.add_argument('--cost-column',default='ns_per_cell_1thread')
    p.add_argument('--dates',type=int,default=1461);p.add_argument('--stocks',type=int,default=5876)
    p.add_argument('--ranks',type=int,default=1);p.add_argument('--ideal-shards',action='store_true')
    p.add_argument('--models',default='fixed_trace,dag_pruned')
    args=p.parse_args();budgets=[float(s) for s in args.budgets_gib.split(',')];modes=args.models.split(',')
    if min(args.dates,args.stocks,args.ranks)<1 or args.ranks>args.dates:p.error('invalid dimensions/ranks')
    if any(not math.isfinite(x) or x<0 for x in budgets):p.error('budgets must be finite and nonnegative')
    if any(x not in ('fixed_trace','dag_pruned') for x in modes):p.error('unknown model')
    if args.ranks>1 and not args.ideal_shards:p.error('P>1 requires --ideal-shards; real halo/communication are not modeled')
    costs=load_costs(args.costs,args.cost_column)
    incoming,hooks=load_batches(args.batches);nodes,batches=prepare(incoming,costs)
    own_rows=(args.dates+args.ranks-1)//args.ranks;panel_bytes=own_rows*args.stocks*4
    results=[]
    for mode in modes:
        for budget in budgets:
            capacity=int(budget*2**30//panel_bytes)
            for policy in POLICIES:
                rows=replay(nodes,batches,capacity,policy,mode,own_rows*args.stocks)
                for r in rows:r.update(budget_gib=budget,ranks=args.ranks,panel_bytes=panel_bytes)
                results.extend(rows)
                baseline=sum(r['baseline_cost_seconds'] for r in rows)
                saved=sum(r['saved_cost_seconds'] for r in rows)
                print(f"{mode:11s} {rows[0]['policy']:22s} M={budget:g}GiB slots={capacity} "
                      f"saved_model_cost={100*saved/baseline if baseline else 0:.2f}% "
                      f"last_batch_hit_rate={rows[-1]['request_hit_rate']:.3f}")
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'cache_sim.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    report={'schema_version':1,'cold_start':True,'batch_count':len(batches),
            'hook_counts':{t:sum(h['type']==t for h in hooks) for t in ('parents','population','gene_pool')},
            'cost_fallbacks':FALLBACKS,'cost_column':args.cost_column,
            'scope':'Only persistent full panel cache. No panel values evaluated. Same dataset/date/stock order is assumed for all batches.',
            'fixed_trace':'All operator nodes requested once per batch in child-first order. Belady maximizes unweighted hits for THIS fixed equal-entry mandatory-insertion trace, not weighted time.',
            'dag_pruned':'Start-of-batch snapshot cuts hit subtrees. Children still needed by other roots are computed. Persistent cache updates occur at batch boundary; temporary live memory is excluded. future_batch_reference uses future unpruned batch membership and is NOT an optimal upper bound.',
            'cost_model':'Weighted operator cost proxy only; excludes IO, cache maintenance, RankIC, halo, MPI, threading speedup. Times from supplied source hardware are not measured on this machine.',
            'ideal_shards':args.ranks>1,'halo_validation':'Not implemented: P>1 is ideal capacity sensitivity only; never a real distributed correctness/speedup claim.',
            'trace_notes':'Only explicit identical event_id records are deduplicated. Population/gene-pool hooks retained by loader but not used by these policies.',
            'rows':results}
    (args.out/'cache_sim.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
