"""Regenerate a paired report/summary/plot from the same raw benchmark CSV."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark import generate_artifacts


def generate_report(results_dir='results'):
    generate_artifacts(results_dir)
    return str(Path(results_dir) / 'report.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', required=True)
    args = p.parse_args()
    print(generate_report(args.results_dir))
