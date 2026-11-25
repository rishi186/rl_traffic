"""Plot verified paired raw CSV; unmatched legacy aggregate files are rejected."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.benchmark import generate_artifacts


def plot_paired(results_dir):
    generate_artifacts(results_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', required=True)
    args = p.parse_args()
    plot_paired(args.results_dir)
