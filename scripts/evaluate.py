"""Schema-v2 evaluation: explicit seconds, seeded demand, TripInfo and network metrics."""
import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_policy


def evaluate(config_path, model_path, num_episodes=5, use_gui=False, output_dir=None, allow_unverified=False):
    config = yaml.safe_load(Path(config_path).read_text())
    config['sumo']['use_gui'] = use_gui
    seeds = range(config['experiment']['seed'], config['experiment']['seed'] + num_episodes)
    return evaluate_policy(config, config['training']['algorithm'], seeds,
                           output_dir or 'results/eval-v2', model_path, allow_unverified)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--output-dir')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--allow-unverified', action='store_true')
    args = p.parse_args()
    evaluate(args.config, args.model, args.episodes, args.gui, args.output_dir, args.allow_unverified)
