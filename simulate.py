"""Run a repeatable CPU memory experiment without torch or model downloads."""
import argparse
import json
from pathlib import Path
from src.memory.simulation import simulate, workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--host-pages', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('experiments/local/simulation.json'))
    args = parser.parse_args()
    requests = workload(args.seed, args.requests)
    results = [simulate(requests, mode, frames=args.frames, block_size=args.block_size, host_pages=args.host_pages)
               for mode in ('contiguous', 'paged', 'swap')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'seed': args.seed, 'results': results}, indent=2) + '\n')
    for r in results:
        print(f"{r['mode']}: completed={r['completed']}, capacity_rejected={r['capacity_rejected']}")
    print(args.output)


if __name__ == '__main__':
    main()
