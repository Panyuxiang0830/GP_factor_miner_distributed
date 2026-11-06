import unittest
from memory_sim import graph,simulate

class MemoryTests(unittest.TestCase):
    def test_independent_roots(self):
        nodes,cs,roots,*_=graph([['mean','x',w] for w in (5,10,20)])
        self.assertEqual(simulate(nodes,cs,roots,'level_keep')['peak_live_panels'],4)
        self.assertEqual(simulate(nodes,cs,roots,'level_early')['peak_live_panels'],4)
        self.assertEqual(simulate(nodes,cs,roots,'level_chunk',1)['peak_live_panels'],2)
        self.assertEqual(simulate(nodes,cs,roots,'greedy')['peak_live_panels'],2)
        # Early release alone cannot fix peak of a single wide operator layer.
    def test_root_is_input(self):
        nodes,cs,roots,*_=graph([['mean','x',5],['rank',['mean','x',5]]])
        for policy in ['level_keep','level_early','level_chunk','greedy']:
            self.assertEqual(simulate(nodes,cs,roots,policy,1)['peak_live_panels'],2)
    def test_repeated_edge_and_disconnected(self):
        nodes,cs,roots,*_=graph([['*','a','a'],['mean','b',5],['rank',['mean','b',5]],'a'])
        for policy in ['level_keep','level_early','level_chunk','greedy']:
            r=simulate(nodes,cs,roots,policy,1)
            self.assertEqual(r['computed_nodes'],len(nodes))
    def test_empty_and_leaf(self):
        for b in [[],['a','a']]:
            nodes,cs,roots,*_=graph(b)
            for policy in ['level_keep','level_early','level_chunk','greedy']:
                r=simulate(nodes,cs,roots,policy,1)
                self.assertEqual(r['peak_live_panels'],int(bool(b)))
if __name__=='__main__':unittest.main()
