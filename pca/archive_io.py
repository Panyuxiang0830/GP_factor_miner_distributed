"""Private portable archive snapshots shared by validation and batch continuation."""
import json
from pathlib import Path
import numpy as np


def fingerprint(path):
    value=14695981039346656037
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):
            for byte in block:value=((value^byte)*1099511628211)&((1<<64)-1)
    return f'{value:016x}'


def load_archive(path,dimension):
    path=Path(path);meta=json.loads(path.read_text())
    if meta.get('version')!=1 or meta.get('dimension')!=dimension or not isinstance(meta.get('records'),list):raise ValueError('archive schema/dimension mismatch')
    binary=path.parent/meta['signatures'];values=np.fromfile(binary,dtype='<f4')
    if binary.stat().st_size!=len(meta['records'])*dimension*4:raise ValueError('archive binary size mismatch')
    if 'signature_hash_fnv1a64' in meta and fingerprint(binary)!=meta['signature_hash_fnv1a64']:raise ValueError('archive signature fingerprint mismatch')
    values=values.reshape(-1,dimension);answer={};slots=set()
    for row in meta['records']:
        key,quality,slot=row['key'],row['quality'],row['slot']
        if not isinstance(key,str) or not key or key in answer or type(quality) not in (int,float) or not np.isfinite(quality):raise ValueError('archive key/quality mismatch')
        if type(slot) is not int or not 0<=slot<len(values) or slot in slots:raise ValueError('archive slot mismatch')
        signature=values[slot].copy().astype(float)
        if not np.isfinite(signature).all() or not np.linalg.norm(signature-signature.mean())>0:raise ValueError('invalid archive signature')
        slots.add(slot);answer[key]=(float(quality),values[slot].copy())
    return answer


def save_archive(path,records,dimension):
    """records maps key to (quality, float32 signature); fixtures/tools only."""
    path=Path(path);binary=path.parent/(path.stem+'_signatures.f32')
    np.asarray([records[k][1] for k in sorted(records)],dtype='<f4').reshape(-1,dimension).tofile(binary)
    meta=dict(version=1,dimension=dimension,signatures=binary.name,
        records=[dict(key=k,quality=records[k][0],slot=i) for i,k in enumerate(sorted(records))],signature_hash_fnv1a64=fingerprint(binary))
    path.write_text(json.dumps(meta,allow_nan=False,indent=2)+'\n');load_archive(path,dimension)
