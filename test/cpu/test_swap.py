import random
import unittest
from src.memory import AllocationError, PagedMemory


class SwapTests(unittest.TestCase):
    def test_lru_content_and_full_host_exchange(self):
        mem = PagedMemory(2, 2, host_pages=2)
        ids = [mem.new_sequence() for _ in range(3)]
        for sid in ids[:2]:
            mem.append(sid, [sid, 'original'])
        mem.read(ids[0], 0)  # The second page is now least recently used.
        mem.append(ids[2], [2, 'new'])
        pid = mem.sequences[ids[1]].pages[0]
        self.assertIsNone(mem.pages[pid].frame)
        fourth = mem.new_sequence()
        mem.append(fourth, 4)
        self.assertEqual(mem.metrics()['free_host_pages'], 0)
        for _ in range(3):
            for sid in ids:
                self.assertEqual(mem.read(sid, 0), [sid, 'new' if sid == 2 else 'original'])
        self.assertGreater(mem.metrics()['swap_in_pages'], 0)
        mem.check_invariants()
        for sid in ids + [fourth]:
            mem.free(sid)
        mem.check_invariants()
        self.assertEqual(mem.metrics()['free_host_pages'], 2)

    def test_pins_capacity_and_exception_cleanup(self):
        mem = PagedMemory(2, 2, host_pages=4)
        a, b, c = [mem.new_sequence() for _ in range(3)]
        for sid in [a, b, c]:
            mem.append(sid, sid)
        with mem.pin_sequences([a, b]) as tables:
            self.assertEqual(len(set(tables[a] + tables[b])), 2)
            with self.assertRaises(AllocationError):
                mem.read(c, 0)
            with self.assertRaises(AllocationError):
                mem.free(a)
            with self.assertRaises(AllocationError):
                mem.swap_out(mem.sequences[a].pages[0])
            d = mem.new_sequence()
            with self.assertRaises(AllocationError):
                mem.reserve(d, 1)
            mem.free(d)
        with self.assertRaises(RuntimeError):
            with mem.pin_sequences([c]):
                raise RuntimeError('consumer failed')
        self.assertTrue(all(p.pins == 0 for p in mem.pages.values()))
        with self.assertRaises(AllocationError):
            with mem.pin_sequences([a, b, c]):
                pass
        mem.check_invariants()

    def test_shared_host_page_cow_single_frame(self):
        mem = PagedMemory(1, 4, host_pages=3)
        a = mem.new_sequence()
        mem.append(a, 1)
        b = mem.fork(a)
        pid = mem.sequences[a].pages[0]
        mem.swap_out(pid)
        mem.swap_out(pid)  # Already on host: no-op.
        mem.append(b, 2)
        self.assertEqual(mem.read_sequence(a), [1])
        self.assertEqual(mem.read_sequence(b), [1, 2])
        self.assertEqual(mem.metrics()['host_cow_reads'], 1)
        mem.check_invariants()

    def test_capacity_failure_no_half_mapping(self):
        mem = PagedMemory(1, 2, host_pages=1)
        a = mem.new_sequence()
        mem.append(a, 1)
        b = mem.fork(a)
        mem.append(b, 2)
        before = mem.metrics()
        with self.assertRaises(AllocationError):
            mem.reserve(b, 3)
        self.assertEqual(mem.metrics(), before)
        self.assertEqual(mem.read_sequence(a), [1])
        self.assertEqual(mem.read_sequence(b), [1, 2])
        with self.assertRaises(AllocationError):
            resident = next(pid for pid, p in mem.pages.items() if p.frame is not None)
            mem.swap_out(resident)
        mem.check_invariants()

    def test_cow_pinned_and_nested_pin(self):
        mem = PagedMemory(3, 2, host_pages=1)
        a = mem.new_sequence()
        mem.append(a, 1)
        b = mem.fork(a)
        with mem.pin_sequences([a]):
            with mem.pin_sequences([a, b]):
                self.assertEqual(mem.pages[mem.sequences[a].pages[0]].pins, 2)
                with self.assertRaises(AllocationError):
                    mem.append(b, 2)
        mem.check_invariants()

    def test_random_swapping_against_independent_lists(self):
        rng = random.Random(505)
        mem = PagedMemory(3, 3, host_pages=9)
        expected = {}
        for _ in range(400):
            if not expected:
                expected[mem.new_sequence()] = []
            sid = rng.choice(list(expected))
            action = rng.randrange(5)
            if action == 0:
                expected[mem.fork(sid)] = list(expected[sid])
            elif action == 1:
                mem.free(sid)
                del expected[sid]
            elif action == 2 and expected[sid]:
                i, value = rng.randrange(len(expected[sid])), rng.randrange(100)
                try:
                    mem.write(sid, i, value)
                    expected[sid][i] = value
                except AllocationError:
                    pass
            else:
                value = rng.randrange(100)
                try:
                    mem.append(sid, value)
                    expected[sid].append(value)
                except AllocationError:
                    pass
            mem.check_invariants()
            for s, values in expected.items():
                self.assertEqual(mem.read_sequence(s), values)
        for s in expected:
            mem.free(s)
        mem.check_invariants()

    def test_backend_copy_failure_releases_reserved_frame(self):
        mem = PagedMemory(2, 2)
        a = mem.new_sequence()
        mem.append(a, 1)
        b = mem.fork(a)
        def fail(*args):
            raise RuntimeError('copy failed')
        mem.storage.copy = fail
        with self.assertRaises(RuntimeError):
            mem.write(b, 0, 2)
        mem.check_invariants()
        self.assertEqual(mem.read_sequence(a), [1])
        self.assertEqual(mem.read_sequence(b), [1])

    def test_reserve_rollback_after_successful_cow_then_io_failure(self):
        mem = PagedMemory(4, 2, host_pages=2)
        a = mem.new_sequence()
        mem.append(a, 1)
        b = mem.fork(a)
        clear = mem.storage.clear
        calls = 0
        def fail_second(frame):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('host allocation failed')
            clear(frame)
        mem.storage.clear = fail_second
        with self.assertRaises(RuntimeError):
            mem.reserve(b, 6)
        mem.check_invariants()
        self.assertEqual(mem.read_sequence(a), [1])
        self.assertEqual(mem.read_sequence(b), [1])
        self.assertEqual(mem.metrics()['cow_copies'], 0)
