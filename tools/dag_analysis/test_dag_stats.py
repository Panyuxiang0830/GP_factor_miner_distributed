import json
import unittest
from pathlib import Path
from dag_stats import analyze, build, cpp_check, REPO

class DagStatsTests(unittest.TestCase):
    def test_shared_root_and_repeated_edge(self):
        batch=[['mean','close',5], ['rank',['mean','close',5]],
               ['+','a','b'], ['+','b','a'], ['*','a','a'],
               ['rank',['rank','a']], ['-', 'x','x']]
        result=analyze(batch,1,True,'g++')
        self.assertEqual(result['accepted'],6)
        self.assertEqual(result['rejected'],1)
        self.assertEqual(result['raw_tree_nodes_accepted'],17)
        self.assertEqual(result['normalized_tree_nodes_accepted'],16)
        self.assertEqual(result['unique_dag_nodes'],8)
        self.assertEqual(result['unique_roots'],5)
        self.assertEqual(result['widest_level_nodes'],4)
        self.assertTrue(all(result['cpp_agreement'].values()))

    def test_operator_coverage_against_cpp(self):
        # Synthetic cases only: no private normalization fixture is required.
        batch = ['a', 'a']
        for op in ('log', 'abs', 'rank', 'zscore'):
            batch.append([op, 'a'])
        for op in ('+', '-', '*', '/'):
            batch.extend([[op, 'a', 'b'], [op, 'b', 'a']])
        for op in ('mean', 'std', 'sum', 'max', 'min', 'median', 'ema',
                   'lag', 'delta', 'pct_change'):
            for window in (1, 5, 20):
                batch.append([op, 'a', window])
        for op in ('correlation', 'beta'):
            for window in (1, 5, 20):
                batch.append([op, 'a', 'b', window])
        batch += [['rank', ['rank', 'a']],
                  ['+', ['mean', 'a', 5], ['std', 'b', 10]],
                  ['+', ['std', 'b', 10], ['mean', 'a', 5]]]
        invalid = [['-', 'x', 'x'], ['/', ['+', 'a', 'b'], ['+', 'b', 'a']],
                   ['unknown', 'a'], ['mean', 'a', 0], ['mean', 'a'], []]
        model = build(batch + invalid)
        self.assertEqual(len(model[2]), len(batch))
        self.assertEqual(len(model[3]), len(invalid))
        self.assertTrue(all(cpp_check(batch + invalid, model, 'g++').values()))

if __name__=='__main__':
    unittest.main()
