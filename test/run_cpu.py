"""CPU-only course tests: python test/run_cpu.py [--coverage]."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', action='store_true')
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
        out = ROOT / 'experiments/local'
        out.mkdir(parents=True, exist_ok=True)
        cov.json_report(outfile=str(out / 'coverage.json'))
    return 0 if result.wasSuccessful() and covered >= 70 else 1


if __name__ == '__main__':
    raise SystemExit(main())
