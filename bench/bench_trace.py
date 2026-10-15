"""Compare the three memory strategies on a real inference trace.

Unlike bench/bench_memory.py, which drives a synthetic load, this entry point
replays recorded arrivals and sequence lengths (Azure LLM trace, ShareGPT or
LongBench) or a committed length distribution, and writes the same summary
columns so both experiments can be read side by side.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.bench_memory import summarize  # noqa: E402
from bench.workloads import LOADERS, Geometry, dump_lengths  # noqa: E402
from src.memory.simulation import simulate  # noqa: E402

MODES = ('contiguous', 'paged', 'swap')


def percentiles(values: list[int], points=(50, 95, 99)) -> dict[str, int]:
    if not values:
        raise ValueError('percentiles require at least one value')
    ordered = sorted(values)
    return {f'p{point}': ordered[max(0, math.ceil(point / 100 * len(ordered)) - 1)] for point in points}


def make_token_counter(name_or_path: str):
    """Build an optional Hugging Face tokenizer-backed counter for text datasets."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit('--tokenizer requires the optional transformers package') from exc
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    count_tokens.__name__ = f'huggingface:{name_or_path}'
    return count_tokens


def git_output(*args: str) -> str:
    """Read Git metadata as UTF-8 on every platform, including Chinese paths on Windows."""
    return subprocess.check_output(
        ['git', *args], cwd=ROOT, text=True, encoding='utf-8', errors='replace').strip()


def load(trace: Path, fmt: str, args) -> 'object':
    if fmt == 'azure':
        return LOADERS['azure'](trace, tick_seconds=args.tick_seconds,
                                max_tokens=args.max_tokens, limit=args.limit)
    if fmt == 'lengths':
        return LOADERS['lengths'](trace)
    if fmt == 'sharegpt':
        return LOADERS['sharegpt'](trace, tick_seconds=args.tick_seconds, max_tokens=args.max_tokens,
                                   arrival_rate=args.arrival_rate, limit=args.limit, seed=args.seed,
                                   count_tokens=args.token_counter)
    if fmt == 'longbench':
        return LOADERS['longbench'](trace, tick_seconds=args.tick_seconds, max_tokens=args.max_tokens,
                                    output_tokens=args.output_tokens, arrival_rate=args.arrival_rate,
                                    limit=args.limit, seed=args.seed,
                                    count_tokens=args.token_counter)
    raise ValueError(f'unknown trace format: {fmt}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True, help='trace or dataset file')
    parser.add_argument('--format', choices=sorted(LOADERS), default='azure')
    parser.add_argument('--frames', type=int, default=None,
                        help='physical GPU pages; defaults to the geometry recorded in a lengths file')
    parser.add_argument('--block-size', type=int, default=None,
                        help='tokens per page; defaults to the geometry recorded in a lengths file')
    parser.add_argument('--host-pages', type=int, default=None,
                        help='host page pool for swap mode; defaults to the recorded geometry')
    parser.add_argument('--tick-seconds', type=float, default=0.05, help='modelled decode step duration')
    parser.add_argument('--max-tokens', type=int, default=2048, help='declared max_model_len')
    parser.add_argument('--arrival-rate', type=float, default=1.0, help='Poisson arrivals per tick')
    parser.add_argument('--output-tokens', type=int, default=128, help='assumed LongBench output length')
    parser.add_argument('--tokenizer', default=None,
                        help='optional Hugging Face tokenizer name/path for ShareGPT or LongBench')
    parser.add_argument('--limit', type=int, default=2000, help='keep the first N admitted requests')
    parser.add_argument('--seed', type=int, default=42, help='arrival seed for text datasets')
    parser.add_argument('--origin', default=None,
                        help='stable dataset label recorded instead of the local file path')
    parser.add_argument('--output', type=Path, default=Path('experiments/local/trace'))
    parser.add_argument('--export-lengths', type=Path, default=None,
                        help='also write the derived arrival/length distribution')
    args = parser.parse_args()
    args.token_counter = make_token_counter(args.tokenizer) if args.tokenizer else None

    loaded = load(args.trace, args.format, args)
    requests = loaded.requests
    lengths = [r.tokens for r in requests]
    prompt_lengths = [r.prompt_tokens for r in requests]
    generated_lengths = [r.generated_tokens for r in requests]
    length_stats = percentiles(lengths)
    prompt_stats = percentiles(prompt_lengths)
    generated_stats = percentiles(generated_lengths)
    recorded = loaded.geometry or Geometry(32, 4, 32)
    frames = args.frames if args.frames is not None else recorded.frames
    block_size = args.block_size if args.block_size is not None else recorded.block_size
    host_pages = args.host_pages if args.host_pages is not None else recorded.host_pages

    runs, rows = [], []
    for mode in MODES:
        run = simulate(requests, mode, frames=frames, block_size=block_size, host_pages=host_pages)
        run.update(scenario='trace', source=loaded.kind, seed=args.seed)
        rows.append(summarize(run, scenario='trace', seed=args.seed,
                              source=loaded.kind, trace_requests=len(requests),
                              prompt_p95=prompt_stats['p95'],
                              generated_p95=generated_stats['p95'],
                              total_p95=length_stats['p95'],
                              peak_active_requests=max((t['active'] for t in run['trace']), default=0)))
        run.pop('trace')  # per-tick state would dominate the report size
        runs.append(run)

    stats = {
        'source': args.origin or loaded.source,
        'kind': loaded.kind,
        'note': loaded.note,
        'requests': len(requests),
        'dropped_too_long': loaded.dropped_too_long,
        'arrival_span_ticks': max(r.arrival for r in requests) if requests else 0,
        'prompt_p50': prompt_stats['p50'],
        'prompt_p95': prompt_stats['p95'],
        'prompt_p99': prompt_stats['p99'],
        'generated_p50': generated_stats['p50'],
        'generated_p95': generated_stats['p95'],
        'generated_p99': generated_stats['p99'],
        'length_p50': length_stats['p50'],
        'length_p95': length_stats['p95'],
        'length_p99': length_stats['p99'],
        'length_max': max(lengths),
        'declared_max_tokens': requests[0].maximum,
    }
    sources = [*sorted((ROOT / 'src/memory').glob('*.py')), Path(__file__), ROOT / 'bench/workloads.py']
    report = {
        'kind': 'cpu_trace_simulation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'commit': git_output('rev-parse', 'HEAD'),
        'working_tree_dirty': bool(git_output('status', '--porcelain')),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'clock': (f'prompt KV allocated at arrival; then one generated token per iteration tick; '
                  f'{args.tick_seconds} s per decode step'),
        'admission': 'arrivals ordered by arrival/rid; capacity failures terminate request without retry',
        'trace': stats,
        'config': {'frames': frames, 'block_size': block_size, 'host_pages': host_pages,
                   'modes': list(MODES), 'per_tick_trace': 'omitted'},
        'runs': runs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'raw.json').write_text(
        json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n', newline='\n')
    with (args.output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'{args.format}: {len(requests)} requests, dropped {loaded.dropped_too_long}, '
          f'p95={stats["length_p95"]} tokens -> {args.output}')
    for row in rows:
        print(f"  {row['mode']:10s} completed={row['completed']:5d} "
              f"rejected={row['capacity_rejected']:5d} swap_in={row['swap_in_pages']}")
    if args.export_lengths:
        geometry = Geometry(frames, block_size, host_pages)
        print(f'-> {dump_lengths(loaded, args.export_lengths, origin=args.origin or loaded.source,
                               geometry=geometry)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
