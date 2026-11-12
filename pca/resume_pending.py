#!/usr/bin/env python3
"""Reconcile and submit only unacknowledged stages after PBS is reachable."""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import subprocess


def save(path,data):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(data,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def command(args):
    result=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parent/'runs/pending_submissions_all.json')
    parser.add_argument('--submit',action='store_true',help='submit missing stages; default checks status only')
    args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text())
    try:
        user=pwd.getpwuid(os.getuid()).pw_name
        command(['qstat','-u',user])
    except (KeyError,RuntimeError,subprocess.TimeoutExpired) as error:
        print(f'PBS/account service unavailable; no jobs submitted: {error}')
        return 2
    for request in catalog['requests']:
        if request.get('id'):
            print('Retaining acknowledged job:',request['id']);continue
        argv=request['argv'][:]
        if argv[0]!='qsub':raise ValueError('catalog may contain only qsub requests')
        variables=dict(item.split('=',1) for item in argv[argv.index('-v')+1].split(','))
        output=Path(variables['PCA_RUN'])
        if output.exists():
            print('Existing output retained; inspect its completion evidence:',output)
            continue
        match=re.search(r'raw_scaling_([124])node',request.get('run',''))
        name=f'pca_raw{match[1]}' if match else request['job_name']
        existing=command(['qselect','-u',user,'-N',name]).splitlines()
        if existing:
            if len(existing)!=1:raise RuntimeError(f'multiple existing jobs named {name}; inspect instead of duplicating')
            request.update(id=existing[0],status='recovered_existing_submission');save(args.catalog,catalog)
            print('Recovered job:',existing[0]);continue
        if not args.submit:
            print('Ready, not submitted:',name,output);continue
        request.update(job_name=name,status='submitting');save(args.catalog,catalog)
        try:
            answer=command(argv[:1]+['-N',name]+argv[1:])
            if not re.fullmatch(r'\d+(?:\.[\w.-]+)?',answer):raise RuntimeError('unexpected qsub response: '+answer)
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            request.update(status='unacknowledged; reconcile before retry',last_error=str(error));save(args.catalog,catalog)
            raise
        request.update(id=answer,status='submitted');save(args.catalog,catalog)
        print('Submitted:',name,answer)
    return 0


if __name__=='__main__':raise SystemExit(main())
