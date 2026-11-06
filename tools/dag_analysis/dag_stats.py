#!/usr/bin/env python3
"""G1: aggregate DAG statistics, using the original Python normalizer.
No factor panels are evaluated. No expressions or field names enter reports.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'python/src'))
from gp_factor_miner.expressions import normalize_expression, expression_key

UNARY = {'log', 'abs', 'rank', 'zscore'}
BINARY = {'+', '-', '*', '/'}
ROLL = {'mean','std','sum','max','min','median','ema','lag','delta','pct_change'}
PAIR = {'correlation','beta'}


def validate(e):
    if isinstance(e, str):
        return
    if not isinstance(e, (list, tuple)) or not e or not isinstance(e[0], str):
        raise ValueError('invalid expression')
    op = e[0]
    if op not in UNARY | BINARY | ROLL | PAIR:
        raise ValueError('unknown operator')
    windowed = op in ROLL | PAIR
    arity = 2 if op in BINARY | PAIR else 1
    if len(e) != 1 + arity + int(windowed):
        raise ValueError('invalid arity')
    if windowed:
        w = e[-1]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 1 or int(w) != w:
            raise ValueError('invalid window')
    for child in e[1:1+arity]:
        validate(child)


def tree_counts(e):
    if isinstance(e, str):
        return Counter({'leaf': 1})
    counts = Counter({e[0]: 1})
    for child in e[1:]:
        if isinstance(child, (str, list, tuple)):
            counts.update(tree_counts(child))
    return counts


def build(batch):
    nodes, roots, accepted, rejected = {}, [], [], []
    raw_ops, normalized_ops = Counter(), Counter()
    def visit(e):
        key = expression_key(e)
        if key in nodes:
            return key
        children = [] if isinstance(e,str) else [visit(c) for c in e[1:] if isinstance(c,(str,list,tuple))]
        level = 0 if not children else 1 + max(nodes[c]['level'] for c in children)
        nodes[key] = {'op': 'leaf' if isinstance(e,str) else e[0], 'level': level,
                      'tree_nodes': 1 + sum(nodes[c]['tree_nodes'] for c in children),
                      'parents': 0}
        for c in children:
            nodes[c]['parents'] += 1
        return key
    for i,e in enumerate(batch):
        try:
            validate(e)
            normalized = normalize_expression(e)
        except (ValueError, TypeError, OverflowError):
            rejected.append(i)
            continue
        raw_ops.update(tree_counts(e))
        normalized_ops.update(tree_counts(normalized))
        roots.append(visit(normalized))
        accepted.append(i)
    return nodes, roots, accepted, rejected, raw_ops, normalized_ops


def cpp_check(batch, model, compiler):
    nodes, roots, accepted, *_ = model
    with tempfile.TemporaryDirectory(prefix='gpe_dag_check_') as d:
        d=Path(d);binary=d/'probe';source=d/'batch.json'
        subprocess.run([compiler,'-std=c++17','-O2','-I'+str(REPO/'include'),
                        str(Path(__file__).with_name('cpp_probe.cpp')),
                        *[str(REPO/'src/core'/f) for f in ('dag.cpp','ops.cpp','json.cpp')],
                        '-o',str(binary)],check=True)
        source.write_text(json.dumps(batch))
        result=json.loads(subprocess.check_output([str(binary),str(source)],text=True))
    cpp_nodes={}
    cpp_keys=[]
    for n in result['nodes']:
        key=expression_key(n['expression']);cpp_keys.append(key)
        if key in cpp_nodes:
            raise AssertionError('C++ duplicate canonical node')
        cpp_nodes[key]={k:n[k] for k in ('level','tree_nodes','parents')}
    py_nodes={key:{k:n[k] for k in ('level','tree_nodes','parents')} for key,n in nodes.items()}
    checks={'node_keys_and_attributes': cpp_nodes==py_nodes,
            'accepted_indices':result['accepted_indices']==accepted,
            'root_sequence':[cpp_keys[i] for i in result['roots']]==roots}
    if not all(checks.values()):
        raise AssertionError('Python/C++ disagreement: '+str(checks))
    return checks


def analyze(batch,repeat,check,compiler):
    times=[]
    for _ in range(repeat):
        start=time.perf_counter();model=build(batch);times.append(time.perf_counter()-start)
    nodes,roots,accepted,rejected,raw,normalized=model
    levels=Counter(n['level'] for n in nodes.values())
    ops=Counter(n['op'] for n in nodes.values())
    raw_n=sum(raw.values());norm_n=sum(normalized.values());unique=len(nodes)
    result={'input_expressions':len(batch),'accepted':len(accepted),'rejected':len(rejected),
            'raw_tree_nodes_accepted':raw_n,'normalized_tree_nodes_accepted':norm_n,
            'unique_dag_nodes':unique,'unique_roots':len(set(roots)),
            'dag_to_raw_ratio':unique/raw_n if raw_n else None,
            'cse_reduction_vs_normalized':1-unique/norm_n if norm_n else None,
            'widest_level_nodes':max(levels.values(),default=0),
            'roots_by_depth':dict(sorted(Counter(nodes[r]['level'] for r in roots).items())),
            'nodes_by_level':dict(sorted(levels.items())),
            'nodes_by_operator':dict(sorted(ops.items())),
            'raw_nodes_by_operator':dict(sorted(raw.items())),
            'normalized_nodes_by_operator':dict(sorted(normalized.items())),
            'python_build_seconds_median':statistics.median(times),
            'timing_scope':'validation + normalization + DAG + counters; excludes JSON IO and C++ check',
            'repeat':repeat}
    if check:
        result['cpp_agreement']=cpp_check(batch,model,compiler)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch',type=Path)
    p.add_argument('--sizes',default='1000,5000,10000,30000')
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--check-cpp',action='store_true')
    p.add_argument('--cxx',default='g++')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    batch=json.loads(args.batch.read_text())
    if not isinstance(batch,list):
        p.error('expected JSON list of expressions')
    sizes=sorted(set([int(s) for s in args.sizes.split(',') if int(s)<=len(batch)]+[len(batch)]))
    if args.repeat<1 or any(s<1 for s in sizes):
        p.error('repeat and sizes must be positive; batch must be nonempty')
    reports=[]
    for size in sizes:
        r=analyze(batch[:size],args.repeat,args.check_cpp,args.cxx);reports.append(r)
        print(f"B={size} accepted={r['accepted']} raw={r['raw_tree_nodes_accepted']} "
              f"DAG={r['unique_dag_nodes']} widest={r['widest_level_nodes']} "
              f"build_s={r['python_build_seconds_median']:.6f}")
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'dag_stats.json').write_text(json.dumps({'schema_version':1,'reports':reports},indent=2))
    fields=['input_expressions','accepted','rejected','raw_tree_nodes_accepted',
            'normalized_tree_nodes_accepted','unique_dag_nodes','unique_roots',
            'dag_to_raw_ratio','cse_reduction_vs_normalized','widest_level_nodes',
            'python_build_seconds_median']
    with (args.out/'dag_stats.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in reports:w.writerow({k:r[k] for k in fields})

if __name__=='__main__':
    main()
