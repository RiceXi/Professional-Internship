"""Real CUDA page round-trip and Qwen3 memory-policy validation.

python test/validate_gpu.py --model ~/huggingface/Qwen3-0.6B
Unavailable dependencies produce a machine-readable report and exit 2.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def page_roundtrip(torch):
    from src.memory import PagedMemory
    from src.memory.torch_storage import TorchStorage
    storage = TorchStorage(torch.zeros(2, 2, 2, 4, 2, 3, dtype=torch.float16, device='cuda'), 3)
    mem = PagedMemory(2, 4, host_pages=3, storage=storage)
    a = mem.new_sequence()
    mem.append(a, 3)
    b = mem.fork(a)
    mem.append(b, 7)
    ids = [a, b]
    for v in [9, 11, 13]:
        sid = mem.new_sequence()
        mem.append(sid, v)
        ids.append(sid)
    for sid, value in zip(ids, [3, 3, 9, 11, 13]):
        assert torch.equal(mem.read(sid, 0), torch.full((2, 2, 2, 3), value, dtype=torch.float16))
    assert (mem.read(b, 1) == 7).all().item()
    mem.check_invariants()
    assert mem.swap_in_pages > 0 and mem.cow_copies > 0
    result = {'status': 'passed', 'memory': mem.metrics(), 'storage': storage.metrics()}
    for sid in ids:
        mem.free(sid)
    mem.check_invariants()
    return result


def model_case(args, torch, frames, host_pages, shared=False):
    from src import EngineConfig
    from src.memory.gpu_engine import CourseEngine
    engine = None
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        config = EngineConfig(model=str(args.model.expanduser()), context_len=args.prompt_tokens + args.output_tokens,
            num_kvcache_blocks=frames, kvcache_block_size=256, max_num_seqs=1,
            max_num_batched_tokens=256, prefix_backend='none', enforce_eager=True,
            torch_compile=False, gpu_memory_utilization=.65, dtype='float16')
        engine = CourseEngine(config, host_pages=host_pages, chunk_size=256)
        init_s = time.perf_counter() - t0
        # Valid IDs come from the actual model vocabulary.
        vocab = config.hf_config.vocab_size
        prompt = [(i * 7919 + 100) % vocab for i in range(args.prompt_tokens)]
        # Warm up in the same engine, then remove request history and timing effects.
        warm = engine.add(prompt[:min(8, len(prompt))], 1)
        engine.runtime.run()
        if engine.runtime.requests[warm].status != 'completed':
            raise RuntimeError('warmup failed')
        engine.runtime.requests.clear()
        engine.runtime.history.clear()
        engine.runtime._next_id = 0
        engine.memory.cow_copies = engine.memory.swap_in_pages = engine.memory.swap_out_pages = 0
        engine.memory.host_cow_reads = 0
        engine.storage.d2h_bytes = engine.storage.h2d_bytes = engine.storage.d2d_bytes = 0
        engine.storage.transfer_seconds = 0.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if shared:
            parent = engine.add(prompt, args.output_tokens)
            while engine.runtime.requests[parent].cached < len(prompt):
                engine.runtime.step()
                if engine.runtime.requests[parent].status != 'running':
                    raise RuntimeError('parent completed or failed before fork')
            for _ in range(args.requests - 1):
                engine.runtime.fork(parent)
        else:
            for _ in range(args.requests):
                engine.add(prompt, args.output_tokens)
        outputs = engine.runtime.run()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        requests = [asdict(r) for r in engine.runtime.requests.values()]
        return {'frames': frames, 'host_pages': host_pages, 'shared': shared,
            'init_seconds': init_s, 'elapsed_seconds': elapsed,
            'output_tokens_per_second': sum(len(v) for v in outputs.values()) / elapsed,
            'outputs': outputs, 'requests': requests, 'trace': engine.runtime.history,
            'memory_final': engine.memory.metrics(), 'transfers': engine.storage.metrics(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'prompt_ids': prompt}
    finally:
        if engine is not None:
            engine.destroy()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('~/huggingface/Qwen3-0.6B'))
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--host-pages', type=int, default=16)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--prompt-tokens', type=int, default=513)
    parser.add_argument('--output-tokens', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--storage-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('experiments/local/gpu_validation.json'))
    args = parser.parse_args()
    if min(args.frames,args.requests,args.prompt_tokens) <= 0 or args.host_pages < 0 or args.output_tokens < 2:
        parser.error('positive sizes, nonnegative host pages, and at least 2 output tokens required')
    report = {'kind': 'gpu_validation', 'scope': 'storage_only' if args.storage_only else 'storage_and_qwen3',
              'status': 'unavailable', 'platform': platform.platform(),
              'python': sys.version, 'seed': args.seed, 'cases': {},
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [*sorted((ROOT / 'src/memory').glob('*.py')), Path(__file__)]}}
    code = 2
    try:
        report['commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()
        if importlib.util.find_spec('torch') is None:
            report['reason'] = 'PyTorch is not installed'
        else:
            import torch
            if not torch.cuda.is_available():
                report['reason'] = 'No CUDA device is available; CPU results are not GPU validation'
            else:
                report.update(gpu=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                              torch=torch.__version__, cuda=torch.version.cuda)
                report['storage_roundtrip'] = page_roundtrip(torch)
                if not args.storage_only:
                    missing = [m for m in ['transformers','flash_attn','flashinfer'] if importlib.util.find_spec(m) is None]
                    if missing or not args.model.expanduser().is_dir():
                        report['reason'] = f'Missing dependencies {missing} or local model {args.model}'
                        return code
                    report['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
                    needed = (args.prompt_tokens + args.output_tokens + 255)//256
                    ample = max(args.frames, needed * args.requests)
                    for name, n, host, shared in [
                        ('ample_reference', ample, 0, False),
                        ('fixed_no_swap', args.frames, 0, False),
                        ('fixed_swap', args.frames, args.host_pages, False),
                        ('fixed_swap_cow', args.frames, args.host_pages, True)]:
                        report['cases'][name] = model_case(args, torch, n, host, shared)
                    baseline = report['cases']['ample_reference']['outputs']
                    assert len(baseline) == args.requests, 'reference did not finish every request'
                    for name in ['fixed_swap','fixed_swap_cow']:
                        case = report['cases'][name]
                        assert case['outputs'] == baseline, f'{name}: token outputs differ or requests failed'
                    if needed * args.requests > args.frames:
                        assert report['cases']['fixed_swap']['memory_final']['swap_in_pages'] > 0, 'pressure test did not exercise swap-in'
                report['status'] = 'passed'
                code = 0
    except Exception as error:
        report.update(status='failed', reason=f'{type(error).__name__}: {error}')
        code = 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f"{report['status']}: {args.output}")
        if 'reason' in report:
            print(report['reason'])
    return code


if __name__ == '__main__':
    raise SystemExit(main())
