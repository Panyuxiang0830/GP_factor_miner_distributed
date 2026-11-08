import unittest
from dag_runtime import graph_metrics,benchmark
from memory_sim import graph

class RuntimeTests(unittest.TestCase):
    def test_vertices_edges_and_operand_slots(self):
        n,c,*_=graph([['*','a','a'],['+','a','b'],['+','b','a']])
        self.assertEqual(graph_metrics(n,c),dict(vertices=4,edges_unique=3,operand_references=4,operator_vertices=2))
    def test_timing_contract(self):
        result=benchmark([['mean','a',5],['rank',['mean','a',5]]],repeat=2,warmup=0,chunks=(1,))
        self.assertEqual(result['metrics']['vertices'],3)
        self.assertEqual(len(result['graph_build_timing']['samples']),2)
        for row in result['rows']:
            self.assertEqual(row['computed_nodes'],3)
            self.assertEqual(row['peak_live_panels'],2)
            self.assertEqual(len(row['schedule_seconds_samples']),2)
if __name__=='__main__':unittest.main()
