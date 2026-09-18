import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from bench.bench_memory import summarize
from bench.workloads import (Geometry, LoadedTrace, dump_lengths, estimate_tokens,
                             load_azure_csv, load_lengths, load_longbench, load_sharegpt,
                             parse_timestamp)
from src.memory.simulation import Request, simulate

ROOT = Path(__file__).resolve().parents[2]
ARCHIVED = ROOT / 'experiments/task1/trace'

AZURE_CSV = """TIMESTAMP,ContextTokens,GeneratedTokens
2023-11-16 18:15:46.6805900,374,44
2023-11-16 18:15:50.9951690,396,109
2023-11-16 18:16:00.0000000,900,200
"""

SHAREGPT_JSON = [
    {'id': 'a', 'conversations': [
        {'from': 'human', 'value': 'Explain paging in one paragraph.'},
        {'from': 'gpt', 'value': 'Paging splits memory into fixed size pages.'}]},
    {'id': 'b', 'conversations': [
        {'from': 'human', 'value': '分页和分段有什么区别？'},
        {'from': 'gpt', 'value': '分页固定大小，分段按逻辑划分。'}]},
    {'id': 'c', 'conversations': [
        {'from': 'human', 'value': 'x' * 4000}]},
]

LONGBENCH_JSONL = '\n'.join([
    json.dumps({'input': 'Summarise the passage.', 'context': 'context ' * 40,
                'answers': ['a']}, ensure_ascii=False),
    json.dumps({'input': '问题', 'context': '长文本' * 50, 'answers': ['b']}, ensure_ascii=False),
])


def write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.write_text(text, encoding='utf-8')
    return path


class TimestampTests(unittest.TestCase):
    def test_seven_digit_fraction_is_accepted_on_python_310(self):
        stamp = parse_timestamp('2023-11-16 18:15:46.6805900')
        self.assertEqual((stamp.hour, stamp.minute, stamp.second, stamp.microsecond),
                         (18, 15, 46, 680590))


class AzureTraceTests(unittest.TestCase):
    def test_real_arrivals_lengths_and_max_tokens_filter(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'trace.csv', AZURE_CSV)
            loaded = load_azure_csv(path, tick_seconds=1.0, max_tokens=1024)
        self.assertEqual(loaded.kind, 'azure-llm-trace')
        self.assertEqual([r.arrival for r in loaded.requests], [0, 4])
        self.assertEqual([r.tokens for r in loaded.requests], [374 + 44, 396 + 109])
        self.assertEqual({r.maximum for r in loaded.requests}, {1024})
        self.assertEqual(loaded.dropped_too_long, 1)
        self.assertEqual([r.rid for r in loaded.requests], [0, 1])

    def test_missing_column_and_empty_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'bad.csv', 'TIME,ContextTokens,GeneratedTokens\n1,2,3\n')
            with self.assertRaises(ValueError):
                load_azure_csv(path)
            path = write(Path(raw), 'long.csv', AZURE_CSV)
            with self.assertRaises(ValueError):
                load_azure_csv(path, max_tokens=8)


class ConversationDatasetTests(unittest.TestCase):
    def test_sharegpt_joins_roles_estimates_tokens_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'sharegpt.json', json.dumps(SHAREGPT_JSON, ensure_ascii=False))
            first = load_sharegpt(path, max_tokens=512, arrival_rate=0.5, seed=7)
            second = load_sharegpt(path, max_tokens=512, arrival_rate=0.5, seed=7)
            other = load_sharegpt(path, max_tokens=512, arrival_rate=0.5, seed=8)
        expected = estimate_tokens('Explain paging in one paragraph.') + \
            estimate_tokens('Paging splits memory into fixed size pages.')
        self.assertEqual(len(first), 2)
        self.assertEqual(first.requests[0].tokens, expected)
        self.assertEqual(first.dropped_too_long, 1)
        self.assertEqual([r.arrival for r in first.requests], [r.arrival for r in second.requests])
        self.assertNotEqual([r.arrival for r in first.requests], [r.arrival for r in other.requests])
        cjk = estimate_tokens('分页和分段有什么区别？') + estimate_tokens('分页固定大小，分段按逻辑划分。')
        self.assertGreaterEqual(cjk, 20)
        self.assertEqual(first.requests[1].tokens, cjk)

    def test_sharegpt_rejects_malformed_entry(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'bad.json', json.dumps([{'id': 'a'}]))
            with self.assertRaises(ValueError):
                load_sharegpt(path)
            path = write(Path(raw), 'empty.json', json.dumps([{'conversations': [
                {'from': 'gpt', 'value': 'no prompt'}]}]))
            with self.assertRaises(ValueError):
                load_sharegpt(path)

    def test_longbench_uses_assumed_output_length(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'longbench.jsonl', LONGBENCH_JSONL)
            loaded = load_longbench(path, output_tokens=32, max_tokens=8192, arrival_rate=1.0)
        self.assertEqual(loaded.kind, 'longbench')
        self.assertEqual(len(loaded), 2)
        prompt = f"{'context ' * 40}\nSummarise the passage.".strip()
        self.assertEqual(loaded.requests[0].tokens, estimate_tokens(prompt) + 32)
        self.assertEqual(loaded.requests[0].maximum, 8192)

    def test_longbench_rejects_missing_fields_and_oversized_samples(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'bad.jsonl', json.dumps({'answers': ['a']}))
            with self.assertRaises(ValueError):
                load_longbench(path)
            path = write(Path(raw), 'longbench.jsonl', LONGBENCH_JSONL)
            with self.assertRaises(ValueError):
                load_longbench(path, output_tokens=0, max_tokens=4)


class LengthDistributionTests(unittest.TestCase):
    def test_dump_and_load_preserve_requests_and_geometry(self):
        trace_requests = [Request(0, 0, 40, 128), Request(1, 3, 100, 128)]
        loaded = LoadedTrace(kind='azure-llm-trace', source='unit', requests=trace_requests,
                             note='unit test')
        with tempfile.TemporaryDirectory() as raw:
            path = dump_lengths(loaded, Path(raw) / 'nested' / 'lengths.json',
                                origin='unit-source', geometry=Geometry(8, 4, 16))
            payload = json.loads(path.read_text(encoding='utf-8'))
            replay = load_lengths(path)
        self.assertEqual(payload['geometry'], {'frames': 8, 'block_size': 4, 'host_pages': 16})
        self.assertEqual(payload['origin'], 'unit-source')
        self.assertNotIn('value', json.dumps(payload))
        self.assertEqual([(r.arrival, r.tokens, r.maximum) for r in replay.requests],
                         [(0, 40, 128), (3, 100, 128)])
        self.assertEqual(replay.geometry, Geometry(8, 4, 16))
        self.assertIn('unit-source', replay.source)

    def test_invalid_lengths_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = write(Path(raw), 'empty.json', json.dumps({'entries': []}))
            with self.assertRaises(ValueError):
                load_lengths(path)


class ReplayPipelineTests(unittest.TestCase):
    def test_recorded_geometry_drives_the_replay(self):
        with tempfile.TemporaryDirectory() as raw:
            path = dump_lengths(
                LoadedTrace(kind='unit', source='unit',
                            requests=[Request(i, i // 2, 24, 64) for i in range(8)]),
                Path(raw) / 'lengths.json', geometry=Geometry(16, 4, 16))
            loaded = load_lengths(path)
        geometry = loaded.geometry
        self.assertEqual(geometry, Geometry(16, 4, 16))
        for mode in ('contiguous', 'paged', 'swap'):
            run = simulate(loaded.requests, mode, frames=geometry.frames,
                           block_size=geometry.block_size, host_pages=geometry.host_pages)
            row = summarize(run)
            self.assertEqual(row['submitted'], row['completed'] + row['capacity_rejected'])
            self.assertEqual(run['final']['allocated_slots'], 0)

    def test_archived_azure_sample_replays_contiguous_and_paged_rows(self):
        lengths = ARCHIVED / 'azure-sample-lengths.json'
        summary = ARCHIVED / 'summary.csv'
        if not lengths.is_file() or not summary.is_file():
            self.skipTest('archived trace experiment not present')
        loaded = load_lengths(lengths)
        with summary.open(newline='') as handle:
            archived = {row['mode']: row for row in csv.DictReader(handle)}
        for mode in ('contiguous', 'paged'):
            run = simulate(loaded.requests, mode, frames=loaded.geometry.frames,
                           block_size=loaded.geometry.block_size, host_pages=loaded.geometry.host_pages)
            row = summarize(run)
            self.assertEqual(row['completed'], int(archived[mode]['completed']), mode)
            self.assertEqual(row['capacity_rejected'], int(archived[mode]['capacity_rejected']), mode)
            self.assertEqual(row['peak_allocated_slots'], int(archived[mode]['peak_allocated_slots']), mode)

    @unittest.skipUnless(os.environ.get('KV_TRACE_SLOW_REPLAY') == '1',
                         'set KV_TRACE_SLOW_REPLAY=1 to replay the full 300-request swap run (~16 s)')
    def test_archived_azure_sample_replays_swap_row(self):
        loaded = load_lengths(ARCHIVED / 'azure-sample-lengths.json')
        with (ARCHIVED / 'summary.csv').open(newline='') as handle:
            archived = {row['mode']: row for row in csv.DictReader(handle)}
        run = simulate(loaded.requests, 'swap', frames=loaded.geometry.frames,
                       block_size=loaded.geometry.block_size, host_pages=loaded.geometry.host_pages)
        row = summarize(run)
        self.assertEqual(row['completed'], int(archived['swap']['completed']))
        self.assertEqual(row['swap_in_pages'], int(archived['swap']['swap_in_pages']))
        self.assertEqual(row['swap_out_pages'], int(archived['swap']['swap_out_pages']))


if __name__ == '__main__':
    unittest.main()
