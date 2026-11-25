"""Schema-v2 evaluation: explicit seconds, seeded demand, TripInfo and network metrics."""
import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_policy


def run_baseline(config_path, num_episodes=5, output_dir=None):
    config = yaml.safe_load(Path(config_path).read_text())
    seeds = range(config['experiment']['seed'], config['experiment']['seed'] + num_episodes)
    return evaluate_policy(config, 'fixed', seeds, output_dir or 'results/baseline-v2')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--output-dir')
    args = p.parse_args()
    run_baseline(args.config, args.episodes, args.output_dir)
