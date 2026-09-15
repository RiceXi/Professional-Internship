import random
import subprocess
import sys
import unittest
from src.memory import AllocationError, ContiguousMemory, ListStorage, PagedMemory


class PagingTests(unittest.TestCase):
    def test_import_without_gpu_dependencies(self):
        code = 'import sys; from src.memory import PagedMemory; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_geometry(self):
        for n, b in [(0, 4), (4, 0), (-1, 2)]:
            with self.assertRaises(ValueError):
                PagedMemory(n, b)
        with self.assertRaises(ValueError):
            PagedMemory(2, 4, storage=ListStorage(1, 4))

    def test_empty_boundary_and_noncontiguous(self):
        mem = PagedMemory(4, 2)
        a, b = mem.new_sequence(), mem.new_sequence()
        self.assertEqual(mem.read_sequence(a), [])
        mem.append(a, 10)
        mem.append(b, 20)
        mem.append(a, 11)
        mem.append(a, 12)
        self.assertEqual(mem.translate(a, 2), (2, 0))
        self.assertEqual(mem.read_sequence(a), [10, 11, 12])
        for index in [-1, 3]:
            with self.assertRaises(IndexError):
                mem.read(a, index)
        mem.check_invariants()
        mem.free(a)
        mem.free(b)
        mem.check_invariants()
        self.assertEqual(mem.metrics()['free_pages'], 4)
        with self.assertRaises(KeyError):
            mem.free(a)

    def test_failure_is_atomic(self):
        mem = PagedMemory(2, 2)
        sid = mem.new_sequence()
        mem.append(sid, [1, 2])
        before = mem.metrics()
        with self.assertRaises(AllocationError):
            mem.reserve(sid, 5)
        self.assertEqual(before, mem.metrics())
        self.assertEqual(mem.read_sequence(sid), [[1, 2]])
        with self.assertRaises(ValueError):
            mem.reserve(sid, 0)
        value = mem.read(sid, 0)
        value.append(3)
        self.assertEqual(mem.read(sid, 0), [1, 2])
        mem.check_invariants()

    def test_metrics(self):
        mem = PagedMemory(4, 4)
        sid = mem.new_sequence()
        mem.reserve(sid, 5)
        metrics = mem.metrics()
        self.assertEqual(metrics['allocated_slots'], 8)
        self.assertEqual(metrics['useful_slots'], 5)
        self.assertEqual(metrics['internal_fragmentation'], 3 / 8)
        mem.free(sid)
        self.assertEqual(mem.metrics()['internal_fragmentation'], 0)

    def test_random_lifecycle(self):
        rng = random.Random(91)
        mem = PagedMemory(12, 3)
        expected = {}
        for _ in range(400):
            if not expected or rng.random() < .3:
                expected[mem.new_sequence()] = []
            sid = rng.choice(list(expected))
            if rng.random() < .2:
                mem.free(sid)
                del expected[sid]
            else:
                value = rng.randrange(1000)
                try:
                    mem.append(sid, value)
                    expected[sid].append(value)
                except AllocationError:
                    pass
                self.assertEqual(mem.read_sequence(sid), expected[sid])
            mem.check_invariants()
        for sid in list(expected):
            mem.free(sid)
        self.assertEqual(mem.metrics()['free_pages'], 12)


class ContiguousTests(unittest.TestCase):
    def test_split_coalesce_and_external_fragmentation(self):
        mem = ContiguousMemory(12)
        a, b, c = [mem.new_sequence(4) for _ in range(3)]
        mem.reserve(b, 2)
        self.assertEqual(mem.metrics()['internal_fragmentation'], 10 / 12)
        mem.free(a)
        mem.free(c)
        self.assertEqual(mem.metrics()['external_fragmentation'], .5)
        with self.assertRaises(AllocationError):
            mem.new_sequence(5)
        mem.check_invariants()
        mem.free(b)
        self.assertEqual(mem.holes, [(0, 12)])
        mem.check_invariants()

    def test_invalid_and_full_capacity(self):
        with self.assertRaises(ValueError):
            ContiguousMemory(0)
        mem = ContiguousMemory(4)
        with self.assertRaises(ValueError):
            mem.new_sequence(0)
        sid = mem.new_sequence(4)
        self.assertEqual(mem.metrics()['external_fragmentation'], 0)
        mem.reserve(sid, 3)
        with self.assertRaises(ValueError):
            mem.reserve(sid, 2)
        with self.assertRaises(AllocationError):
            mem.reserve(sid, 5)
        mem.free(sid)
        self.assertEqual(mem.metrics()['allocated_utilization'], 0)
