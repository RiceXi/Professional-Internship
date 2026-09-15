"""Run independent GPU validation processes; preserve failures and boundaries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def classify(report, returncode, working_set_exceeds):
    if returncode == 2 and report.get('status') == 'unavailable':
        return 'unavailable'
    if returncode == 0 and report.get('status') == 'passed':
        return 'passed'
    rejected = report.get('cases', {}).get('fixed_swap', {}).get('requests', [])
    boundary_reasons = {'RuntimeError: parent completed or failed before fork',
                        'AssertionError: fixed_swap: token outputs differ or requests failed'}
    if (working_set_exceeds and report.get('reason') in boundary_reasons
        and report.get('storage_roundtrip', {}).get('status') == 'passed'
        and rejected and all(r['status'] == 'capacity_rejected' and
                            'working set exceeds' in (r.get('error') or '') for r in rejected)):
        return 'capacity_boundary'
    return 'failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('~/huggingface/Qwen3-0.6B'))
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--host-pages', type=int, default=64)
    parser.add_argument('--requests', type=int, nargs='+', default=[1, 3, 8])
    parser.add_argument('--prompt-tokens', type=int, nargs='+', default=[257, 513, 1025])
    parser.add_argument('--output-tokens', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/local/gpu')
    args = parser.parse_args()
    if (min(args.frames, args.timeout, *args.requests, *args.prompt_tokens) <= 0
        or args.host_pages < 0 or args.output_tokens < 2):
        parser.error('positive sizes, nonnegative host capacity and at least 2 output tokens required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for count in args.requests:
        for length in args.prompt_tokens:
            name = f'requests-{count}_prompt-{length}'
            path = args.output / (name + '.json')
            command = [sys.executable, str(ROOT / 'test/validate_gpu.py'), '--model', str(args.model.expanduser()),
                       '--frames', str(args.frames), '--host-pages', str(args.host_pages),
                       '--requests', str(count), '--prompt-tokens', str(length),
                       '--output-tokens', str(args.output_tokens), '--output', str(path)]
            # Do not interpret an old report if this run exits before writing one.
            if path.exists():
                path.unlink()
            try:
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=args.timeout)
                (args.output / (name + '.txt')).write_text(completed.stdout + completed.stderr)
                report = json.loads(path.read_text()) if path.exists() else {}
                status = classify(report, completed.returncode, length + args.output_tokens - 1 > args.frames * 256)
                results.append({'name': name, 'status': status, 'returncode': completed.returncode,
                                'report': path.name, 'reason': report.get('reason')})
            except subprocess.TimeoutExpired as error:
                output = (error.stdout or b'') + (error.stderr or b'')
                (args.output / (name + '.txt')).write_bytes(output)
                results.append({'name': name, 'status': 'failed', 'reason': 'timeout', 'report': None})
            print(f"{name}: {results[-1]['status']}", flush=True)
            (args.output / 'matrix.json').write_text(json.dumps({'scope': 'CUDA capacity matrix', 'results': results}, indent=2) + '\n')
            if results[-1]['status'] == 'unavailable':
                return 2  # Identical environment; avoid retrying every matrix cell.
    return 1 if any(r['status'] == 'failed' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
