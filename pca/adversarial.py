#!/usr/bin/env python3
"""Generate private boundary/tie/duplicate/invalid-signature QD fixtures."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to((Path(__file__).resolve().parent/"runs").resolve()):
        parser.error("output must be new and inside runs/")
    args.out.mkdir(parents=True)
    n=3554;rng=np.random.default_rng(1)
    left=rng.normal(size=n);left-=left.mean();left/=np.linalg.norm(left)
    basis=rng.normal(size=n);basis-=basis.mean();basis-=left*np.dot(left,basis);basis/=np.linalg.norm(basis)
    right=.8999999758*left+np.sqrt(1-.8999999758**2)*basis
    cases={"threshold":([left,right],["old","new"],[.03,.031])}
    probe=np.random.default_rng(7).normal(size=n);probe-=probe.mean();probe/=np.linalg.norm(probe)
    archived=[]
    for seed,corr in ((8,.8),(9,.800000003)):
        orth=np.random.default_rng(seed).normal(size=n);orth-=orth.mean();orth-=probe*np.dot(probe,orth);orth/=np.linalg.norm(orth)
        archived.append(corr*probe+np.sqrt(1-corr*corr)*orth)
    cases["near_tie"]=(archived+[probe],["z","a","new"],[.03,.03,.04])
    cases["duplicate_margin"]=([left,left,left],["same","same","same"],[.03,.032,.033])
    cases["invalid"]=([left,np.ones(n),np.full(n,np.nan),probe],["good","constant","nan","other"],[.03]*4)
    for name,(signatures,keys,qualities) in cases.items():
        output=args.out/name;output.mkdir()
        np.asarray(signatures,dtype="<f4").tofile(output/"signatures.f32")
        np.ones(5,dtype="<f8").tofile(output/"initial.f8")
        records=[dict(index=i,score_line=i,key=key,quality=quality,candidate=True) for i,(key,quality) in enumerate(zip(keys,qualities))]
        (output/"manifest.json").write_text(json.dumps(dict(rows=n,cols=5,initial=str((output/"initial.f8").resolve()),records=records),indent=2)+"\n")


if __name__=="__main__":main()
