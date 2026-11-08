#!/usr/bin/env python3
"""S5 serial execution-plan export and structural validation, no numeric engine."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from dag_stats import build, expression_key
from memory_sim import graph, simulate


def require(ok,message):
    if not ok:raise ValueError(message)


def export_plan(batch,context,policy='level_chunk',chunk=256):
    nodes,children,roots,accepted,rejected=graph(batch)
    mapping,root_keys,accepted_ids,rejected_ids,*_=build(batch)
    keys=list(mapping);node_ids={k:i for i,k in enumerate(keys)}
    aliases={}
    for source_index,key in zip(accepted_ids,root_keys):aliases.setdefault(str(node_ids[key]),[]).append(source_index)
    events=[]
    result=simulate(nodes,children,roots,policy,chunk,emit=events.append)
    table=[{'id':i,'operator':n['op'],'inputs':children[i],
            'expression':json.loads(keys[i]),
            'expression_sha256':hashlib.sha256(keys[i].encode()).hexdigest(),
            'source':'load' if n['op']=='leaf' else 'compute'} for i,n in enumerate(nodes)]
    plan={'schema_version':1,'mode':'serial_full_panel','context':context,
          'normalizer':'handed-off Python expression_key (compact JSON)',
          'nodes':table,'root_outputs':aliases,'rejected_input_indices':rejected_ids,
          'input_count':len(batch),'initial_cache':[],
          'policy':policy,'chunk':chunk,'predicted_peak_live_panels':result['peak_live_panels'],
          'instructions':events}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require(plan.get('schema_version')==1 and plan.get('mode')=='serial_full_panel','unsupported version/mode')
    ctx=plan['context']
    require(all(ctx.get(k) for k in ('dataset_id','dates_id','assets_id','operator_semantics_id')),'missing data/semantic identity')
    require(ctx.get('ranks')==1 and ctx.get('rank')==0,'v1 supports full-panel P=1 only')
    require(ctx.get('dtype')=='float32' and ctx.get('layout')=='date_major','unsupported panel layout')
    require(isinstance(ctx.get('T'),int) and ctx['T']>0 and isinstance(ctx.get('N'),int) and ctx['N']>0,'invalid dimensions')
    nodes={n['id']:n for n in plan['nodes']};require(len(nodes)==len(plan['nodes']),'duplicate node id')
    for n in nodes.values():
        require(hashlib.sha256(expression_key(n['expression']).encode()).hexdigest()==n['expression_sha256'],'expression hash mismatch')
    outputs={int(k):v for k,v in plan['root_outputs'].items()}
    for node,indices in outputs.items():require(node in nodes and bool(indices),'invalid root mapping')
    all_output_indices=[j for indices in outputs.values() for j in indices]
    rejected=plan['rejected_input_indices']
    require(len(all_output_indices)==len(set(all_output_indices)),'duplicate input output')
    require(sorted(all_output_indices+rejected)==list(range(plan['input_count'])),'input/output mapping incomplete')
    pending=Counter()
    for i,n in nodes.items():
        require(n['source'] in ('load','compute','cache_get'),'invalid source')
        if n['source']=='compute':
            require(bool(n['inputs']) and all(c in nodes for c in n['inputs']),'unknown compute input')
            pending.update(n['inputs'])
        else:require(not n['inputs'],'load/cache node must have no execution dependencies')
    cache={e['key']:e for e in plan.get('initial_cache',[])}
    for e in cache.values():require(e['context']==ctx,'cache context mismatch')
    allocated=set();ready=set();executed=set();consumed=set();group=None;peak=0
    for step in plan['instructions']:
        op=step['op'];i=step.get('node')
        if op=='begin_group':
            require(group is None,'nested group');ids=step['nodes']
            require(len(ids)==len(set(ids)) and all(j in nodes and j not in executed and j not in allocated for j in ids),'invalid allocation')
            group=set(ids);allocated.update(ids);peak=max(peak,len(allocated))
        elif op in ('load','compute','cache_get'):
            require(group is not None and i in group and i not in executed,'node not allocated once in group')
            n=nodes[i];require(n['source']==op,'source opcode mismatch')
            if op=='compute':
                require(all(c in ready for c in n['inputs']),'input not ready or already released')
                pending.subtract(n['inputs'])
            elif op=='cache_get':
                e=cache.get(step['key']);require(e is not None,'cache miss: stop and replan')
                require(e['context']==ctx and e['key']==n['expression_sha256'],'cache identity mismatch')
            ready.add(i);executed.add(i)
        elif op=='consume_root':
            require(i in outputs and i in ready and i not in consumed,'root unavailable or consumed twice')
            consumed.add(i)
        elif op=='release':
            require(i in ready and pending[i]==0,'release before final consumer')
            require(i not in outputs or i in consumed,'root released before metrics')
            ready.remove(i);allocated.remove(i)
        elif op=='cache_put':
            require(i in ready and step['key']==nodes[i]['expression_sha256'],'invalid cache put')
            # Cache owns an immutable reference independently from batch handles.
            cache[step['key']]={'key':step['key'],'context':ctx}
        elif op=='cache_evict':
            require(step['key'] in cache,'unknown cached entry');del cache[step['key']]
        elif op=='end_group':
            require(group is not None and group.issubset(executed),'group incomplete');group=None
        else:raise ValueError('unknown opcode '+str(op))
    require(group is None and not allocated and not ready,'unfinished group or leaked batch handles')
    require(executed==set(nodes) and consumed==set(outputs),'incomplete plan')
    require(peak==plan['predicted_peak_live_panels'],'peak mismatch')
    return {'passed':True,'nodes':len(nodes),'outputs':len(all_output_indices),
            'peak_live_panels':peak,'instructions':len(plan['instructions'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('export');make.add_argument('batch',type=Path);make.add_argument('--out',type=Path,required=True)
    make.add_argument('--context',type=Path,required=True);make.add_argument('--chunk',type=int,default=256)
    make.add_argument('--policy',choices=['level_keep','level_early','level_chunk','greedy'],default='level_chunk')
    check=sub.add_parser('check');check.add_argument('plan',type=Path)
    args=p.parse_args()
    if args.command=='export':
        require(args.chunk>0,'chunk must be positive')
        plan=export_plan(json.loads(args.batch.read_text()),json.loads(args.context.read_text()),args.policy,args.chunk)
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(plan,indent=2))
    else:plan=json.loads(args.plan.read_text())
    print(json.dumps(validate_plan(plan)))
if __name__=='__main__':main()
