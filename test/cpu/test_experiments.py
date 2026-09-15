import unittest
from bench.bench_memory import branch_case, external_fragmentation_case, summarize
from src.memory.simulation import simulate, Request


class ExperimentTests(unittest.TestCase):
    def test_summaries_reconcile_with_terminal_requests(self):
        for mode in ['contiguous','paged','swap']:
            run=simulate([Request(i,0,6,16) for i in range(4)],mode,frames=2,block_size=4,host_pages=6)
            row=summarize(run)
            self.assertEqual(row['submitted'],row['completed']+row['capacity_rejected'])
            self.assertEqual(len(run['records']),4)
            self.assertEqual({r['rid'] for r in run['records']},set(range(4)))
            self.assertEqual(run['final']['allocated_slots'],0)
        empty=summarize(simulate([], 'paged'))
        self.assertEqual(empty['weighted_internal_fragmentation'],0)
        self.assertEqual(empty['rejection_rate'],0)

    def test_branch_counts_and_external_fragmentation(self):
        plain,shared=branch_case(False),branch_case(True)
        self.assertEqual(plain['after_write']['allocated_slots'],108)
        self.assertEqual(shared['before_write']['allocated_slots'],12)
        self.assertEqual(shared['after_write']['allocated_slots'],44)
        self.assertEqual(shared['after_write']['cow_copies'],8)
        self.assertEqual(shared['final']['allocated_slots'],0)
        fragmented=external_fragmentation_case()
        self.assertTrue(fragmented['rejected'])
        self.assertEqual(fragmented['fragmented']['external_fragmentation'],.5)
        self.assertEqual(fragmented['after_coalesce']['largest_free_extent'],12)
