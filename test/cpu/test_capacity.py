import argparse
import csv
import unittest
from pathlib import Path

from bench.bench_capacity import boundary_rows, parse_list, sweep

ROOT = Path(__file__).resolve().parents[2]
ARCHIVED = ROOT / 'experiments/task1/capacity'


class ParseListTests(unittest.TestCase):
    def test_accepts_spaces_and_rejects_bad_values(self):
        self.assertEqual(parse_list('256, 1024,4096'), [256, 1024, 4096])
        self.assertEqual(parse_list('8'), [8])
        for text in ('', 'abc', '1,,2', '0', '-4'):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_list(text)


class BoundaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for mode, completed in (('contiguous', 1), ('paged', 3)):
            for concurrency in (1, 2, 4):
                rejected = 0 if concurrency <= 2 else concurrency - completed
                rows.append({'mode': mode, 'tokens': 128, 'concurrency': concurrency,
                             'completed': completed if concurrency <= 2 else completed,
                             'capacity_rejected': rejected})
        return rows

    def test_boundary_picks_largest_admitted_concurrency(self):
        boundary = boundary_rows(self.rows())
        by_mode = {row['mode']: row for row in boundary}
        self.assertEqual(by_mode['contiguous']['max_concurrency_no_reject'], 2)
        self.assertEqual(by_mode['paged']['max_concurrency_no_reject'], 2)
        self.assertEqual(by_mode['contiguous']['max_tested_concurrency'], 4)
        self.assertEqual(by_mode['paged']['completed_at_max_tested'], 3)
        self.assertEqual([row['tokens'] for row in boundary], [128, 128])

    def test_boundary_reports_zero_when_every_run_rejects(self):
        rows = [{'mode': 'paged', 'tokens': 64, 'concurrency': count,
                 'completed': 0, 'capacity_rejected': count} for count in (1, 8)]
        boundary = boundary_rows(rows)
        self.assertEqual(boundary[0]['max_concurrency_no_reject'], 0)
        self.assertEqual(boundary[0]['completed_at_max_tested'], 0)
        self.assertEqual(boundary[0]['max_tested_concurrency'], 8)


class SweepTests(unittest.TestCase):
    def test_small_grid_reconciles_and_keeps_reports_small(self):
        rows, runs = sweep([64], [1, 2], frames=8, block_size=8, host_pages=8, max_tokens=64)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(runs), 6)
        for row in rows:
            self.assertEqual(row['submitted'], row['completed'] + row['capacity_rejected'])
            self.assertEqual(row['frames'], 8)
            self.assertEqual(row['block_size'], 8)
        for run in runs:
            self.assertNotIn('trace', run)
            self.assertEqual(run['final']['allocated_slots'], 0)
        boundary = boundary_rows(rows)
        self.assertEqual({row['mode'] for row in boundary}, {'contiguous', 'paged', 'swap'})
        self.assertEqual({row['tokens'] for row in boundary}, {64})

    def test_context_length_above_declared_maximum_is_rejected(self):
        with self.assertRaises(ValueError):
            sweep([128], [1], frames=8, block_size=8, host_pages=8, max_tokens=64)


class ArchivedCapacityTests(unittest.TestCase):
    def load(self, name):
        path = ARCHIVED / name
        if not path.is_file():
            self.skipTest('archived capacity experiment not present')
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle))

    def test_archived_boundary_matches_archived_runs(self):
        rows = self.load('summary.csv')
        archived = {(r['mode'], r['tokens']): r for r in self.load('boundary.csv')}
        for row in boundary_rows(rows):
            stored = archived[(row['mode'], str(row['tokens']))]
            self.assertEqual(row['max_concurrency_no_reject'],
                             int(stored['max_concurrency_no_reject']))
            self.assertEqual(row['completed_at_max_tested'],
                             int(stored['completed_at_max_tested']))

    def test_contiguous_capacity_is_locked_to_one_sequence(self):
        rows = [row for row in self.load('summary.csv') if row['mode'] == 'contiguous']
        self.assertTrue(rows)
        self.assertEqual({row['completed'] for row in rows}, {'1'})
        boundary = [row for row in self.load('boundary.csv') if row['mode'] == 'contiguous']
        self.assertEqual({row['max_concurrency_no_reject'] for row in boundary}, {'1'})


if __name__ == '__main__':
    unittest.main()
