"""CPU regression: a preempted sequence must finish recomputing before sampling.

The scheduler used to decide "prompt finished" from ``num_cached_tokens`` versus
``num_prompt_tokens``. After preemption, KV allocation resets the cached count
but keeps the tokens generated so far, so a chunk that crosses the prompt
boundary during recompute looked like a completed prompt and appended a token
early. These cases drive the real scheduling loop with a fake KV layer, so they
run without torch, CUDA or a local model.
"""

from __future__ import annotations

import unittest

from src.control.scheduler import Scheduler
from src.data.sequence import Sequence, SequenceStatus
from src.sampling_params import SamplingParams


class _FakeKV:
    """Minimal KV interface used by ``Scheduler``; no tensor storage involved."""

    def __init__(self) -> None:
        self.deallocations = 0

    def can_allocate(self, seq: Sequence) -> bool:
        return True

    def allocate(self, seq: Sequence) -> None:
        seq.block_table = [0]

    def can_append(self, seq: Sequence) -> bool:
        return True

    def may_append(self, seq: Sequence) -> None:
        return None

    def sync_prefix(self, seq: Sequence) -> None:
        return None

    def deallocate(self, seq: Sequence, *, publish: bool = True) -> None:
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        self.deallocations += 1


def _sequence(prompt_length: int, max_tokens: int) -> Sequence:
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    return Sequence.from_prompt(
        list(range(prompt_length)), block_size=16, sampling=sampling
    )


class RecomputeBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kv = _FakeKV()
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=2,
            max_num_batched_tokens=1024,
            eos_token_id=-1,
        )
        # The design report's example: 1500-token prompt, 1500 tokens generated,
        # then preempted before max_tokens is reached.
        self.seq = _sequence(prompt_length=1500, max_tokens=2000)
        self.seq.token_ids.extend(range(10_000, 11_500))

    def _preempt(self) -> None:
        self.seq.status = SequenceStatus.RUNNING
        self.scheduler.preempt(self.seq)
        self.assertEqual(self.seq.num_cached_tokens, 0)
        self.assertEqual(self.seq.num_completion_tokens, 1500)

    def _step(self, token_id: int | None = None) -> int:
        batch = self.scheduler.schedule()
        self.assertTrue(batch.prefills)
        self.assertEqual(batch.decodes, [])
        seq = batch.prefills[0]
        before = len(seq.token_ids)
        self.scheduler.postprocess([seq], [token_id], is_prefill=True)
        return len(seq.token_ids) - before

    def test_partial_chunk_across_prompt_boundary_does_not_sample(self) -> None:
        self._preempt()

        # 1024 + 1024 crosses prompt length 1500 while 952 tokens are still
        # uncached. The second step must not append the sampled token.
        self.assertEqual(self._step(), 0)
        self.assertEqual(self.seq.num_cached_tokens, 1024)
        self.assertFalse(self.seq.need_sample)

        self.assertEqual(self._step(), 0)
        self.assertEqual(self.seq.num_cached_tokens, 2048)
        self.assertGreater(self.seq.num_cached_tokens, self.seq.num_prompt_tokens)
        self.assertFalse(self.seq.need_sample)
        self.assertEqual(len(self.seq.token_ids), 3000)
        self.assertEqual(self.seq.num_completion_tokens, 1500)

        # The final 952-token chunk covers the whole sequence and samples once.
        self.assertEqual(self._step(999), 1)
        self.assertEqual(self.seq.num_cached_tokens, 3000)
        self.assertTrue(self.seq.need_sample)
        self.assertEqual(self.seq.token_ids[-1], 999)
        self.assertEqual(self.seq.num_completion_tokens, 1501)
        self.assertIn(self.seq, self.scheduler.running)

    def test_fresh_prompt_chunk_boundary_is_unchanged(self) -> None:
        seq = _sequence(prompt_length=1500, max_tokens=4)
        self.scheduler.add(seq)

        self.assertEqual(self._step(), 0)
        self.assertEqual(seq.num_cached_tokens, 1024)
        self.assertEqual(self._step(777), 1)
        self.assertEqual(seq.num_cached_tokens, 1500)
        self.assertEqual(seq.num_completion_tokens, 1)


if __name__ == "__main__":
    unittest.main()
