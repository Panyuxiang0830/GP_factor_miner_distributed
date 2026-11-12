#!/usr/bin/env python3
"""Original EOD CLI -> live C++ GP -> original out-of-sample screening."""
import argparse
from pathlib import Path
import json
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python/src'))
import numpy as np
import pandas as pd
from gp_factor_miner.hierarchical_cli import main as original_main
from gp_factor_miner.data import load_eod_panel_dataset,split_eod_panels
import gp_factor_miner.hierarchical_engine as engine
from gp_factor_miner.pca_qd import PCAQDArchive


def main():
    p=argparse.ArgumentParser();p.add_argument('--engine-binary',type=Path,required=True);p.add_argument('--pca-binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mpiexec',default='mpirun');a=p.parse_args()
    out=a.out.resolve()
    if out.exists() or not out.is_relative_to((Path(__file__).resolve().parent/'runs').resolve()):p.error('new private run required')
    out.mkdir();rng=np.random.default_rng(9104)
    dates=pd.Index([*pd.bdate_range('2016-01-04',periods=201).strftime('%Y%m%d'),*pd.bdate_range('2022-01-04',periods=64).strftime('%Y%m%d')])
    assets=pd.Index([f'{i:06d}' for i in range(1,9)])
    data={('close' if i==0 else f'f{i}'):pd.DataFrame(rng.normal(size=(len(dates),len(assets))),index=dates,columns=assets) for i in range(6)}
    target=data['close']-.4*data['f1'];pd.to_pickle(data,out/'eod.pkl');pd.to_pickle(target,out/'label.pkl')
    dataset=load_eod_panel_dataset(out/'eod.pkl',out/'label.pkl');evolution,y,_,_=split_eod_panels(dataset)
    panels=out/'panels';(panels/'fields').mkdir(parents=True)
    for name,x in evolution.items():x.to_numpy(dtype='<f4').tofile(panels/'fields'/f'{name}.f32')
    y.to_numpy(dtype='<f4').tofile(panels/'target.f32')
    (panels/'meta.json').write_text(json.dumps(dict(T=len(y),N=len(y.columns),dates=list(map(str,y.index)),assets=list(map(str,y.columns)),fields=list(evolution))))
    args=['--data-file',str(out/'eod.pkl'),'--label-file',str(out/'label.pkl'),'--output-dir',str(out/'result'),'--smoke',
          '--cpp-panel-dir',str(panels),'--cpp-engine-binary',str(a.engine_binary.resolve()),'--cpp-pca-binary',str(a.pca_binary.resolve()),'--cpp-ranks','4','--cpp-mpiexec',a.mpiexec]
    # Screening may use its original Python math; evolution must never fall back.
    with patch.object(engine,'_evaluate',side_effect=AssertionError('old evolution evaluator')),patch.object(engine,'factor_signature',side_effect=AssertionError('old evolution PCA')),patch.object(engine,'evaluate_expression',side_effect=AssertionError('old evolution panels')),patch.object(PCAQDArchive,'consider_batch',side_effect=AssertionError('old evolution QD')):
        assert original_main(args)==0
    result=out/'result';config=json.loads((result/'run_config.json').read_text())
    assert config['completed_depths']==[1,2,3] and config['batch_backends']
    required=('gene_pool_depth_01.json','gene_pool_depth_02.json','gene_pool_depth_03.json','pca_qd_archive.json','validation_rankic.csv','correlation_decisions.csv','selected_gp_factors.json')
    assert all((result/name).exists() for name in required)
    calls=json.loads((result/'eval/cpp_backend_01/calls.json').read_text());assert any(x['kind']=='evaluate' for x in calls) and any(x['kind']=='qd' for x in calls)
    # A completed run is reusable only with the same backend identity.
    assert original_main(args)==0
    (out/'SUCCESS.json').write_text(json.dumps(dict(verified=True,original_cli=True,cpp_evolution=True,original_oos_screening=True,
        completed_stages=[1,2,3],completed_resume=True,output_files=list(required),selected=len(json.loads((result/'selected_gp_factors.json').read_text()))),indent=2)+'\n')
    (out/'SUCCESS').touch()


if __name__=='__main__':main()
