"""Experiment runner with hyperparameter sweep support.

Runs multiple training experiments with different hyperparameter configurations,
either from a sweep config file or CLI overrides.  Each run gets its own
subdirectory under the main log_dir.

Usage:
    # Single run with overrides
    python scripts/run_experiments.py --config config.yaml \\
        --override training.total_episodes=200 dqn.learning_rate=0.001

    # Sweep from YAML
    python scripts/run_experiments.py --config config.yaml \\
        --sweep configs/sweep.yaml

Sweep YAML format:
    algorithm: [dqn, ppo]
    training:
      total_episodes: [100, 200]
    dqn:
      learning_rate: [0.0001, 0.0005, 0.001]
      dueling: [true, false]
"""

import os
import sys
import copy
import json
import yaml
import argparse
import itertools
from typing import List, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_override(s: str) -> Tuple[str, Any]:
    """Parse a key=value override string.

    Supports nested keys with dot notation, and type inference
    (int, float, bool, string).

    Args:
        s: Override string like ``"dqn.learning_rate=0.001"``.

    Returns:
        Tuple of (dotted_key, typed_value).
    """
    key, _, val = s.partition("=")
    key = key.strip()
    val = val.strip()

    # Type inference
    if val.lower() in ("true", "false"):
        return key, val.lower() == "true"
    try:
        return key, int(val)
    except ValueError:
        pass
    try:
        return key, float(val)
    except ValueError:
        pass
    return key, val


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation.

    Args:
        config: Config dict to modify in-place.
        dotted_key: Dot-separated path like ``"dqn.learning_rate"``.
        value: Value to set.
    """
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _generate_sweep_configs(
    base_config: dict, sweep_config: dict
) -> List[Tuple[str, dict]]:
    """Generate all combinations from a sweep config.

    Args:
        base_config: Base config dict.
        sweep_config: Sweep spec with lists of values for each key.

    Returns:
        List of (run_name, config) tuples.
    """
    # Flatten sweep config into list of (dotted_key, values) pairs
    def flatten(d: dict, prefix: str = "") -> List[Tuple[str, List]]:
        result = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.extend(flatten(v, key))
            elif isinstance(v, list):
                result.append((key, v))
            else:
                # Single value — treat as list with one element
                result.append((key, [v]))
        return result

    flat = flatten(sweep_config)
    keys = [k for k, _ in flat]
    value_lists = [vs for _, vs in flat]

    configs = []
    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base_config)
        parts = []
        for k, v in zip(keys, combo):
            _set_nested(cfg, k, v)
            short = k.split(".")[-1]
            parts.append(f"{short}={v}")
        run_name = "_".join(parts)
        configs.append((run_name, cfg))

    return configs


def run_experiment(
    config: dict,
    run_name: str,
    base_log_dir: str,
) -> dict:
    """Run a single training experiment.

    Args:
        config: Full config dict for this run.
        run_name: Name for this run (used in log dir).
        base_log_dir: Base directory for all runs.

    Returns:
        Summary dict with run results.
    """
    run_log_dir = os.path.join(base_log_dir, run_name)
    config["experiment"]["log_dir"] = run_log_dir
    config["experiment"]["name"] = run_name

    logger.info("=" * 60)
    logger.info("Starting experiment: %s", run_name)
    logger.info("Log dir: %s", run_log_dir)
    logger.info("=" * 60)

    # Import and run training
    from scripts.train import train as run_train

    # Write temp config
    temp_config_path = os.path.join(run_log_dir, "config.yaml")
    os.makedirs(run_log_dir, exist_ok=True)
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    run_train(temp_config_path)

    return {"run_name": run_name, "log_dir": run_log_dir, "status": "completed"}


def run_experiments(
    config_path: str,
    sweep_path: str = None,
    overrides: List[str] = None,
    output_file: str = None,
) -> List[dict]:
    """Run one or more training experiments.

    Args:
        config_path: Path to base config YAML.
        sweep_path: Optional path to sweep config YAML.
        overrides: Optional list of key=value override strings.
        output_file: Optional path to save experiment results JSON.

    Returns:
        List of result dicts.
    """
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    base_log_dir = base_config["experiment"]["log_dir"]
    results = []

    if sweep_path:
        with open(sweep_path, "r") as f:
            sweep_config = yaml.safe_load(f)
        configs = _generate_sweep_configs(base_config, sweep_config)
        logger.info("Generated %d experiment configurations from sweep", len(configs))
    elif overrides:
        cfg = copy.deepcopy(base_config)
        name_parts = []
        for ov in overrides:
            key, val = _parse_override(ov)
            _set_nested(cfg, key, val)
            short = key.split(".")[-1]
            name_parts.append(f"{short}={val}")
        run_name = "_".join(name_parts)
        configs = [(run_name, cfg)]
    else:
        configs = [("default", base_config)]

    for run_name, cfg in configs:
        try:
            result = run_experiment(cfg, run_name, base_log_dir)
        except Exception as e:
            logger.error("Experiment '%s' failed: %s", run_name, e)
            result = {"run_name": run_name, "status": "failed", "error": str(e)}
        results.append(result)

    # Save results summary
    if output_file is None:
        output_file = os.path.join(base_log_dir, "experiment_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Experiment results saved to %s", output_file)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RL traffic experiments with optional hyperparameter sweeps"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to base config file")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Path to sweep config YAML (generates multiple runs)")
    parser.add_argument("--override", nargs="*", default=None,
                        help="Key=value overrides (e.g. dqn.learning_rate=0.001)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save experiment results JSON")
    args = parser.parse_args()

    run_experiments(args.config, args.sweep, args.override, args.output)
