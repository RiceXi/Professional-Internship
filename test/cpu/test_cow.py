import random
import unittest
from src.memory import AllocationError, PagedMemory


class CowTests(unittest.TestCase):
    def test_partial_tail_and_release_orders(self):
        for parent_first in [False, True]:
            mem = PagedMemory(6, 4)
            parent = mem.new_sequence()
            for value in range(6):
                mem.append(parent, value)
            child = mem.fork(parent)
            self.assertEqual(mem.metrics()['shared_slots_saved'], 6)
            self.assertEqual(mem.metrics()['resident_pages'], 2)
            mem.append(child, 99)
            self.assertEqual(mem.metrics()['cow_copies'], 1)
            self.assertEqual(mem.read_sequence(parent), list(range(6)))
            self.assertEqual(mem.read_sequence(child), list(range(6)) + [99])
            self.assertEqual(mem.sequences[parent].pages[0], mem.sequences[child].pages[0])
            first, last = (parent, child) if parent_first else (child, parent)
            expected = mem.read_sequence(last)
            mem.free(first)
            self.assertEqual(mem.read_sequence(last), expected)
            mem.check_invariants()
            mem.free(last)
            self.assertEqual(mem.metrics()['free_pages'], 6)

    def test_full_page_write_and_append(self):
        mem = PagedMemory(6, 2)
        a = mem.new_sequence()
        mem.append(a, 1)
        mem.append(a, 2)
        b = mem.fork(a)
        mem.append(b, 3)
        self.assertEqual(mem.metrics()['cow_copies'], 0)
        mem.write(b, 0, 9)
        self.assertEqual(mem.read_sequence(a), [1, 2])
        self.assertEqual(mem.read_sequence(b), [9, 2, 3])
        self.assertEqual(mem.metrics()['cow_copies'], 1)
        mem.check_invariants()
        with self.assertRaises(IndexError):
            mem.write(b, 3, 1)

    def test_cow_capacity_failure_preserves_data(self):
        mem = PagedMemory(2, 4)
        a = mem.new_sequence()
        mem.append(a, 7)
        b = mem.fork(a)
        other = mem.new_sequence()
        mem.append(other, 8)
        before = mem.metrics()
        for fn in [lambda: mem.append(b, 9), lambda: mem.write(b, 0, 10), lambda: mem.reserve(b, 10)]:
            with self.assertRaises(AllocationError):
                fn()
            self.assertEqual(mem.metrics(), before)
            self.assertEqual(mem.read_sequence(a), [7])
            self.assertEqual(mem.read_sequence(b), [7])
            mem.check_invariants()
        mem.free(other)
        mem.append(b, 9)
        mem.check_invariants()

    def test_empty_fork_and_random_reference_model(self):
        rng = random.Random(2026)
        mem = PagedMemory(16, 3)
        a = mem.new_sequence()
        b = mem.fork(a)
        expected = {a: [], b: []}
        for _ in range(600):
            if not expected:
                expected[mem.new_sequence()] = []
            sid = rng.choice(list(expected))
            action = rng.randrange(4)
            if action == 0:
                expected[mem.fork(sid)] = list(expected[sid])
            elif action == 1:
                mem.free(sid)
                del expected[sid]
            elif action == 2 and expected[sid]:
                idx = rng.randrange(len(expected[sid]))
                val = rng.randrange(1000)
                try:
                    mem.write(sid, idx, val)
                    expected[sid][idx] = val
                except AllocationError:
                    pass
            else:
                val = rng.randrange(1000)
                try:
                    mem.append(sid, val)
                    expected[sid].append(val)
                except AllocationError:
                    pass
            mem.check_invariants()
            for key, value in expected.items():
                self.assertEqual(mem.read_sequence(key), value)
        for sid in expected:
            mem.free(sid)
        mem.check_invariants()
        self.assertEqual(mem.metrics()['free_pages'], 16)
