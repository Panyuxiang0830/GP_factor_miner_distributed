import itertools
import json
from pathlib import Path
import tempfile
import unittest
from functools import lru_cache
from cache_sim import Cache,prepare,replay,load_batches

COSTS={'mean':10.,'std':20.,'rank':30.,'abs':1.,'+':1.,'*':1.}
def batches(*items):return [{'members':x,'metadata':{}} for x in items]

class CacheTests(unittest.TestCase):
    def test_cached_root_prunes_entire_subtree(self):
        b=batches([['rank',['mean','x',5]]],[['rank',['mean','x',5]]])
        n,bs=prepare(b,COSTS)
        for policy in ('lru','lfu','cost','belady'):
            rows=replay(n,bs,1,policy,'dag_pruned',1)
            self.assertEqual(rows[0]['saved_cost_seconds'],0)
            self.assertEqual(rows[1]['computed_operator_nodes'],0)
            self.assertEqual(rows[1]['cache_hits'],1)
            self.assertEqual(rows[1]['pruned_descendant_nodes'],1)
            self.assertEqual(rows[1]['saved_cost_fraction'],1)

    def test_shared_descendant_still_required(self):
        a=['mean','x',5];root=['rank',a]
        n,bs=prepare(batches([root],[root,['abs',a]]),COSTS)
        r=replay(n,bs,1,'lru','dag_pruned',1)[1]
        self.assertEqual(r['baseline_operator_nodes'],3)
        self.assertEqual(r['cache_hits'],1)
        self.assertEqual(r['computed_operator_nodes'],2)
        self.assertEqual(r['pruned_descendant_nodes'],0)
        self.assertAlmostEqual(r['saved_cost_fraction'],30/41)

    def test_zero_and_large_capacity(self):
        a=['mean','x',5]
        n,bs=prepare(batches([a,['+',a,a]],[a,['+',a,a]]),COSTS)
        for policy,mode in itertools.product(('lru','lfu','cost','belady'),('fixed_trace','dag_pruned')):
            r=replay(n,bs,0,policy,mode,1)
            self.assertTrue(all(x['saved_cost_seconds']==0 for x in r))
            r=replay(n,bs,100,policy,mode,1)
            self.assertEqual(r[1]['saved_cost_fraction'],1)
            self.assertTrue(all(x['resident_panels_after_batch']<=100 for x in r))

    def test_belady_against_exhaustive_mandatory_insertion(self):
        # Brute force all feasible evictions for a small fixed request trace.
        nodes=[{'op':'mean','subtree_ns':1} for _ in range(3)]
        for sequence in itertools.product(range(3),repeat=5):
            @lru_cache(None)
            def optimal(pos,state):
                if pos==len(sequence):return 0
                x=sequence[pos];resident=set(state)
                if x in resident:return optimal(pos+1,state)
                if len(resident)<2:return 1+optimal(pos+1,tuple(sorted(resident|{x})))
                return 1+min(optimal(pos+1,tuple(sorted((resident-{v})|{x}))) for v in resident)
            cache=Cache(2,'belady',nodes);misses=0
            for pos,x in enumerate(sequence):
                misses+=x not in cache.entries
                future=next((i for i in range(pos+1,len(sequence)) if sequence[i]==x),float('inf'))
                cache.touch(x,future)
            self.assertEqual(misses,optimal(0,()))

    def test_trace_order_and_explicit_ids_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.jsonl'
            rows=[{'type':'population','members':[]},
                  {'type':'batch','phase':'offspring','generation':1,'members':[['mean','x',5]]},
                  {'type':'batch','phase':'offspring','generation':1,'members':[['std','x',5]]}]
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            b,h=load_batches([p]);self.assertEqual(len(b),2);self.assertEqual(len(h),1)
            rows[1]['event_id']='first';rows.append(rows[1])
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            b,h=load_batches([p]);self.assertEqual(len(b),2)
if __name__=='__main__':unittest.main()
