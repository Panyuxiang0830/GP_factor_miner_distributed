#!/usr/bin/env python3
"""Check all completed configurations against the fully verified Python trace."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import os

os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--oracle",type=Path,required=True,help="Python seed0_power_k04.json trace")
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/"runs"
    if args.out.exists() or not args.out.resolve().is_relative_to(root.resolve()):
        parser.error("output must be new and inside runs/")
    oracle=json.loads(args.oracle.read_text())
    baseline=np.fromfile(args.baseline,dtype="<f4").reshape(-1,293).astype(float)
    report=[]
    for parent in args.runs:
        if not (parent/"SUCCESS").exists(): raise ValueError(f"incomplete scaling job: {parent.name}")
        for run in sorted(parent.glob("*_r1")):
            metrics=json.loads((run/"metrics.json").read_text())
            decisions=json.loads((run/"decisions.json").read_text())
            trace={str(x["signature_index"]):[x["action"],x["closest_key"],x["replaced_key"]] for x in decisions}
            assert trace==oracle["trace"],run.name
            assert json.loads((run/"archive_keys.json").read_text())==oracle["archive_keys"],run.name
            signature=np.fromfile(run/"signatures.f32",dtype="<f4").reshape(baseline.shape).astype(float)
            norms=np.linalg.norm(baseline,axis=1)*np.linalg.norm(signature,axis=1)
            cosine=np.abs(np.sum(baseline*signature,axis=1)/norms)
            error=float(max(0,np.max(1-cosine)))
            assert error<1e-6,run.name
            bitwise=(run/"signatures.f32").read_bytes()==args.baseline.read_bytes()
            if metrics["layout"] in ("factor","transpose"):
                assert bitwise,(run.name,"factor layout changed same-input results")
            for repeat in (2,3):
                other=run.with_name(run.name[:-1]+str(repeat))
                for artifact in ("signatures.f32","decisions.json","archive_keys.json"):
                    assert (run/artifact).read_bytes()==(other/artifact).read_bytes(),(run.name,repeat,artifact)
            report.append(dict(run=run.name,ranks=metrics["ranks"],threads=metrics["threads"],
                factors=metrics["factors"],max_one_minus_abs_cosine=error,
                signature_bitwise_equal_serial=bitwise,python_oracle_decisions_equal=True,
                repetitions_byte_identical=True))
    args.out.mkdir(parents=True)
    (args.out/"SUCCESS.json").write_text(json.dumps(dict(verified=True,configurations=report),indent=2)+"\n")
    print(f"Validated {len(report)} configurations, including max ranks={max(x['ranks'] for x in report)}",flush=True)


if __name__=="__main__":main()
