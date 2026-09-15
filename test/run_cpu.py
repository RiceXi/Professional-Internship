"""CPU-only course tests: python test/run_cpu.py [--coverage]."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/local')
    args = parser.parse_args()
    cov = None
    if args.coverage:
        import coverage
        cov = coverage.Coverage(source=['src.memory'])
        cov.start()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'test/cpu'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    covered = 100
    if cov:
        cov.stop()
        cov.save()
        covered = cov.report(show_missing=True)
        args.output.mkdir(parents=True, exist_ok=True)
        cov.json_report(outfile=str(args.output / 'coverage.json'))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        'scope': 'CPU logic and optional CPU tensor backend; CUDA not exercised',
        'python': sys.version, 'platform': platform.platform(),
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
        'line_coverage_percent': covered if cov else None,
        'coverage_threshold': 70,
        'status': 'passed' if result.wasSuccessful() and covered >= 70 else 'failed',
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for directory in ('src/memory', 'test/cpu')
                          for p in sorted((ROOT / directory).glob('*.py'))},
    }
    (args.output / 'test_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return 0 if result.wasSuccessful() and covered >= 70 else 1


if __name__ == '__main__':
    raise SystemExit(main())
