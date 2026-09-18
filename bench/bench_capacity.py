"""Long-context capacity boundary: how many sequences can run at once.

Every request arrives at tick 0, so the sweep measures the simultaneous
working set rather than an arrival process. That is the CPU counterpart of
the long-context pressure test: a request that cannot keep its whole
attention working set resident is rejected instead of crashing with OOM.

The derived boundary table reports, per strategy and context length, the
largest concurrency that is admitted without any rejection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.bench_memory import summarize  # noqa: E402
from src.memory.simulation import Request, simulate  # noqa: E402

MODES = ('contiguous', 'paged', 'swap')


def parse_list(text: str) -> list[int]:
    parts = [part.strip() for part in text.split(',')]
    if not text.strip() or any(not part for part in parts):
        raise argparse.ArgumentTypeError(f'expected comma-separated integers, got {text!r}')
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'expected comma-separated integers, got {text!r}') from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError('values must be positive')
    return values


def sweep(lengths: list[int], concurrency: list[int], *, frames: int, block_size: int,
          host_pages: int, max_tokens: int, modes=MODES) -> tuple[list[dict], list[dict]]:
    """Run the grid and return (per-run rows, per-run summaries with traces)."""
    runs, rows = [], []
    for length in lengths:
        if length > max_tokens:
            raise ValueError(f'context length {length} exceeds max_tokens={max_tokens}')
        for count in concurrency:
            requests = [Request(i, 0, length, max_tokens) for i in range(count)]
            for mode in modes:
                run = simulate(requests, mode, frames=frames, block_size=block_size,
                               host_pages=host_pages)
                run.update(scenario='capacity_boundary', context_tokens=length, concurrency=count)
                rows.append(summarize(run, scenario='capacity_boundary', seed=0,
                                      concurrency=count, tokens=length))
                run.pop('trace')  # per-tick state would dominate the report size
                runs.append(run)
    return rows, runs


def boundary_rows(rows: list[dict]) -> list[dict]:
    """Largest concurrency admitted without rejection, per strategy and length.

    ``completed_at_max_tested`` reports behaviour at the most extreme grid
    point, where a strategy that drops the whole batch differs from one that
    still finishes a single request.
    """
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row['mode'], int(row['tokens'])), []).append(row)
    boundary = []
    for (mode, length), items in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ordered = sorted(items, key=lambda row: int(row['concurrency']))
        zero_reject = [int(r['concurrency']) for r in ordered if int(r['capacity_rejected']) == 0]
        largest = ordered[-1]
        boundary.append({
            'mode': mode,
            'tokens': length,
            'max_concurrency_no_reject': max(zero_reject, default=0),
            'max_tested_concurrency': int(largest['concurrency']),
            'completed_at_max_tested': int(largest['completed']),
        })
    return boundary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', type=parse_list, default=parse_list('256,1024,4096'),
                        help='context lengths in tokens, comma separated')
    parser.add_argument('--concurrency', type=parse_list, default=parse_list('1,4,16,64'),
                        help='numbers of simultaneously arriving requests')
    parser.add_argument('--frames', type=int, default=256, help='physical GPU pages')
    parser.add_argument('--block-size', type=int, default=16, help='tokens per page')
    parser.add_argument('--host-pages', type=int, default=256, help='host page pool for swap mode')
    parser.add_argument('--max-tokens', type=int, default=4096, help='declared max_model_len')
    parser.add_argument('--output', type=Path, default=Path('experiments/task1/capacity'))
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    rows, runs = sweep(args.lengths, args.concurrency, frames=args.frames,
                       block_size=args.block_size, host_pages=args.host_pages,
                       max_tokens=args.max_tokens)
    boundary = boundary_rows(rows)

    sources = [*sorted((ROOT / 'src/memory').glob('*.py')), Path(__file__)]
    report = {
        'kind': 'cpu_capacity_boundary',
        'created_utc': started.isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'clock': 'unit iteration ticks; all requests arrive at tick 0',
        'admission': 'capacity failures terminate request without retry',
        'config': {'frames': args.frames, 'block_size': args.block_size,
                   'host_pages': args.host_pages, 'max_tokens': args.max_tokens,
                   'lengths': args.lengths, 'concurrency': args.concurrency,
                   'modes': list(MODES), 'per_tick_trace': 'omitted'},
        'boundary': boundary,
        'runs': runs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'raw.json').write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
    write_csv(args.output / 'summary.csv', rows)
    write_csv(args.output / 'boundary.csv', boundary)

    print(f'{len(rows)} runs, physical pool {args.frames * args.block_size} tokens -> {args.output}')
    for row in boundary:
        print(f"  {row['mode']:10s} {row['tokens']:5d} tokens: "
              f"no-reject concurrency <= {row['max_concurrency_no_reject']:3d}, "
              f"completed {row['completed_at_max_tested']:3d} at concurrency "
              f"{row['max_tested_concurrency']:3d}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
