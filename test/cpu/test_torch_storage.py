import importlib.util
import unittest
from src.memory import PagedMemory


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional tensor backend requires torch')
class TensorStorageTests(unittest.TestCase):
    def test_copy_swap_and_cow_with_real_tensors(self):
        import torch
        from src.memory.torch_storage import TorchStorage
        storage = TorchStorage(torch.zeros(2, 2, 2, 4, 2, 3), host_pages=3)
        mem = PagedMemory(2, 4, host_pages=3, storage=storage)
        a = mem.new_sequence()
        mem.append(a, 3)
        b = mem.fork(a)
        mem.append(b, 7)
        c = mem.new_sequence()
        mem.append(c, 9)
        d = mem.new_sequence()
        mem.append(d, 11)
        e = mem.new_sequence()
        mem.append(e, 13)
        for sid, val in [(a,3),(b,3),(c,9),(d,11),(e,13)]:
            self.assertTrue(torch.equal(mem.read(sid,0), torch.full((2,2,2,3),float(val))))
        self.assertTrue((mem.read(b,1) == 7).all())
        mem.check_invariants()
        self.assertGreater(storage.metrics()['h2d_bytes'],0)
        for sid in [a,b,c,d,e]:
            mem.free(sid)
        self.assertTrue(all(p is None for p in storage.host))

    def test_invalid_geometry(self):
        import torch
        from src.memory.torch_storage import TorchStorage
        with self.assertRaises(ValueError):
            TorchStorage(torch.zeros(2,3))
        with self.assertRaises(ValueError):
            TorchStorage(torch.zeros(2,1,1,1,1,1),-1)

    def test_direct_host_roundtrip(self):
        import torch
        from src.memory.torch_storage import TorchStorage
        store = TorchStorage(torch.zeros(2,1,2,2,1,1), 1)
        mem = PagedMemory(2,2,host_pages=1,storage=store)
        sid = mem.new_sequence()
        mem.append(sid, 42)
        mem.swap_out(mem.sequences[sid].pages[0])
        self.assertTrue((mem.read(sid,0) == 42).all())
        self.assertEqual(store.h2d_bytes, store.page_bytes)
        mem.check_invariants()
