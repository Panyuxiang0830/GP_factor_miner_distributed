"""Compare all score fields except chunk/slot; never print expressions."""
import argparse
import json
from pathlib import Path


def load(path):
    return [{k:v for k,v in json.loads(line).items() if k not in ('chunk','slot')}
            for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    ref=load(args.directory/'c200_r1/scores.jsonl')
    if len(ref)!=200:raise ValueError('expected 200 score records')
    checks=[]
    for c in (16,64,200):
        for r in (1,2,3):
            rows=load(args.directory/f'c{c}_r{r}/scores.jsonl')
            if rows!=ref:raise ValueError(f'score mismatch: chunk={c}, repeat={r}')
            checks.append(dict(chunk=c,repeat=r,records=len(rows),differences=0))
    (args.directory/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
    print('PASS: all 9 score files agree (excluding chunk and slot)')

if __name__=='__main__':main()
