import unittest
from src.memory import PagedMemory
from src.memory.runtime import MemoryRuntime


def execute_on_lists(memory):
    def execute(req, table, start, count, sample):
        # A tiny recurrent reference: stored values depend on all preceding tokens.
        for i in range(start, start + count):
            prev = memory.storage.read(table[(i-1)//memory.block_size], (i-1)%memory.block_size) if i else 0
            memory.storage.write(table[i//memory.block_size], i%memory.block_size, (prev + req.tokens[i]) % 101)
        value = memory.storage.read(table[(start+count-1)//memory.block_size], (start+count-1)%memory.block_size)
        return value if sample else None
    return execute


class RuntimeTests(unittest.TestCase):
    def test_reference_vs_swapping_and_fork(self):
        def run(frames, host):
            memory = PagedMemory(frames, 4, host_pages=host)
            runtime = MemoryRuntime(memory, execute_on_lists(memory), chunk_size=3)
            a = runtime.add([1,2,3,4,5], 4)
            runtime.step()  # Three valid cached tokens, unfinished prefill.
            b = runtime.fork(a)
            runtime.add([8,9,2,1], 4)
            out = runtime.run()
            self.assertEqual(out[a], out[b])
            memory.check_invariants()
            self.assertEqual(memory.metrics()['allocated_slots'], 0)
            return out, memory.metrics()
        reference, _ = run(12, 0)
        actual, metrics = run(2, 10)
        self.assertEqual(actual, reference)
        self.assertGreater(metrics['cow_copies'], 0)
        self.assertGreater(metrics['swap_out_pages'], 0)

    def test_zero_output_eos_cancel_and_rejection(self):
        mem = PagedMemory(1, 2, host_pages=4)
        rt = MemoryRuntime(mem, execute_on_lists(mem), chunk_size=1, context_len=20, eos_token_id=3)
        a = rt.add([1,2], 4)
        zero = rt.add([1,2], 0)
        too_big = rt.add([1,2,3], 1)
        cancelled = rt.add([1], 1)
        rt.cancel(cancelled)
        result = rt.run()
        self.assertEqual(result[a], [3])
        self.assertEqual(result[zero], [])
        self.assertEqual(rt.requests[too_big].status, 'capacity_rejected')
        self.assertEqual(rt.requests[cancelled].status, 'cancelled')
        with self.assertRaises(ValueError):
            rt.fork(a)
        with self.assertRaises(ValueError):
            rt.cancel(a)
        self.assertIsNone(rt.step())

    def test_errors_and_budget(self):
        mem = PagedMemory(4, 2)
        with self.assertRaises(ValueError):
            MemoryRuntime(mem, None, chunk_size=0)
        rt = MemoryRuntime(mem, lambda *args: None, context_len=4)
        for prompt, n in [([], 1), ([1], -1), ([1,2,3], 2)]:
            with self.assertRaises(ValueError):
                rt.add(prompt, n)
        rid = rt.add([1], 1)
        with self.assertRaises(RuntimeError):
            rt.run(0)
        with self.assertRaises(RuntimeError):
            rt.step()
        self.assertEqual(rt.requests[rid].status, 'execution_failed')
        mem.check_invariants()
        self.assertEqual(mem.metrics()['allocated_slots'], 0)
        rt2 = MemoryRuntime(mem, execute_on_lists(mem))
        rt2.add([1], 1)
        self.assertEqual(rt2.run(1), {0:[1]})

    def test_gpu_probe_unavailable_is_not_success(self):
        import importlib.util
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        if importlib.util.find_spec('torch'):
            import torch
            if torch.cuda.is_available():
                self.skipTest('this test exercises missing CUDA only')
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'probe.json'
            result = subprocess.run([sys.executable, 'test/validate_gpu.py', '--storage-only', '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(out.read_text())['status'], 'unavailable')
