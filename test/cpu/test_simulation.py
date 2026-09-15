import unittest
from src.memory.simulation import Request, simulate, workload


class SimulationTests(unittest.TestCase):
    def test_reproducible_and_reclaimed(self):
        jobs = workload(42)
        for mode in ('paged', 'contiguous', 'swap'):
            a = simulate(jobs, mode, frames=8)
            b = simulate(jobs, mode, frames=8)
            self.assertEqual(a, b)
            self.assertEqual(a['completed'] + a['capacity_rejected'], len(jobs))
            self.assertEqual(a['final']['allocated_slots'], 0)

    def test_paging_uses_actual_length(self):
        jobs = [Request(i, 0, 2, 16) for i in range(4)]
        self.assertEqual(simulate(jobs, 'paged', frames=4)['completed'], 4)
        self.assertEqual(simulate(jobs, 'contiguous', frames=4)['completed'], 1)

    def test_empty_gaps_and_invalid(self):
        self.assertEqual(simulate([], 'paged')['completed'], 0)
        self.assertEqual(simulate([Request(1, 100, 1, 1)], 'paged')['completed'], 1)
        for req in [Request(1, -1, 2, 2), Request(1, 0, 3, 2)]:
            with self.assertRaises(ValueError):
                simulate([req], 'paged')
        with self.assertRaises(ValueError):
            simulate([Request(1, 0, 1, 1)] * 2, 'paged')
        with self.assertRaises(ValueError):
            simulate([], 'bad')
        with self.assertRaises(ValueError):
            simulate([], 'paged', frames=0)
        with self.assertRaises(ValueError):
            workload(count=0)

    def test_swap_improves_admission_but_not_working_set_limit(self):
        jobs = [Request(i, 0, 6, 16) for i in range(4)]
        paged = simulate(jobs, 'paged', frames=2, block_size=4)
        swapped = simulate(jobs, 'swap', frames=2, block_size=4, host_pages=6)
        self.assertEqual(swapped['completed'], 4)
        self.assertLess(paged['completed'], 4)
        self.assertGreater(swapped['final']['swap_in_pages'], 0)
        oversized = simulate([Request(1, 0, 9, 9)], 'swap', frames=2, block_size=4)
        self.assertEqual(oversized['capacity_rejected'], 1)
