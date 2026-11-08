import copy
import unittest
from plan_contract import export_plan,validate_plan
CTX={'dataset_id':'synthetic-demo-v1','dates_id':'demo-date-order','assets_id':'demo-stock-order',
     'operator_semantics_id':'handoff-2026-11-04','T':120,'N':300,'ranks':1,'rank':0,'dtype':'float32','layout':'date_major'}
BATCH=[['mean','x',5],['rank',['mean','x',5]],['*','x','x'],['mean','x',5]]
class PlanTests(unittest.TestCase):
    def test_all_policies(self):
        for policy in ('level_keep','level_early','level_chunk','greedy'):
            p=export_plan(BATCH,CTX,policy,1);self.assertEqual(validate_plan(p)['outputs'],4)
    def test_premature_release_rejected(self):
        p=export_plan(BATCH,CTX)
        i=next(n['id'] for n in p['nodes'] if n['operator']=='leaf')
        j=next(j for j,s in enumerate(p['instructions']) if s['op']=='load')
        p['instructions'].insert(j+1,{'op':'release','node':i})
        with self.assertRaises(ValueError):validate_plan(p)
    def test_missing_consume_rejected(self):
        p=export_plan(BATCH,CTX);p['instructions']=[s for s in p['instructions'] if s['op']!='consume_root']
        with self.assertRaises(ValueError):validate_plan(p)
    def test_cache_hit_and_miss(self):
        p=export_plan([['mean','x',5]],CTX);root=next(n for n in p['nodes'] if n['operator']=='mean')
        root['inputs']=[];root['source']='cache_get';key=root['expression_sha256'];p['nodes']=[root]
        p['initial_cache']=[{'key':key,'context':CTX}];p['predicted_peak_live_panels']=1
        p['instructions']=[{'op':'begin_group','nodes':[root['id']]},{'op':'cache_get','node':root['id'],'key':key},
                           {'op':'consume_root','node':root['id']},{'op':'release','node':root['id']},{'op':'end_group'}]
        self.assertTrue(validate_plan(p)['passed'])
        bad=copy.deepcopy(p);bad['initial_cache']=[]
        with self.assertRaises(ValueError):validate_plan(bad)
        bad=copy.deepcopy(p);bad['context']=dict(CTX,T=121)
        with self.assertRaises(ValueError):validate_plan(bad)
if __name__=='__main__':unittest.main()
