"""Fixed-time baseline runner for traffic signal control.

Runs the SUMO simulation with default fixed-time signal programs (no RL)
and collects the same metrics used to evaluate RL agents.  Results are
saved as JSON + CSV for direct comparison.

Usage:
    python scripts/baseline_fixed.py --config config.yaml --episodes 5
"""

import os
import sys
import json
import csv
import yaml
import argparse
import numpy as np
import traci
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def run_baseline(config_path: str, num_episodes: int = 5, output_dir: str = None) -> dict:
    """Run SUMO with fixed-time signals and collect metrics.

    Args:
        config_path: Path to YAML config file.
        num_episodes: Number of simulation runs.
        output_dir: Directory for output files. Defaults to ``results/baseline/``.

    Returns:
        Dictionary of aggregate baseline metrics.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["experiment"]["log_dir"]
    if output_dir is None:
        output_dir = f"{log_dir}/baseline"
    os.makedirs(output_dir, exist_ok=True)

    setup_logger(log_dir=log_dir, log_level="INFO")

    sumo_cfg = config["sumo"]["cfg_file"]
    delta_time = config["sumo"]["delta_time"]
    max_steps = config["training"]["max_steps_per_episode"]

    logger.info("=" * 60)
    logger.info("Fixed-Time Baseline Evaluation")
    logger.info("Config: %s | Episodes: %d", config_path, num_episodes)
    logger.info("=" * 60)

    all_waiting_times = []
    all_queues = []
    all_throughputs = []
    episode_results = []

    for episode in range(num_episodes):
        # Start SUMO
        sumo_cmd = [
            "sumo",
            "-c", sumo_cfg,
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        traci.start(sumo_cmd)

        ts_ids = list(traci.trafficlight.getIDList())

        # Discover incoming lanes per traffic light
        ts_lanes = {}
        for ts_id in ts_ids:
            lanes = list(set(traci.trafficlight.getControlledLanes(ts_id)))
            ts_lanes[ts_id] = [
                lane for lane in lanes if not traci.lane.getEdgeID(lane).startswith(":")
            ]

        step_waiting = []
        step_queue = []
        vehicles_departed = 0
        vehicles_arrived = 0
        step = 0

        pbar = tqdm(
            total=max_steps,
            desc=f"Baseline Episode {episode + 1}/{num_episodes}",
            leave=True,
        )

        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Collect metrics every delta_time steps
            if step % delta_time == 0:
                total_wait = 0.0
                total_queue = 0
                for ts_id in ts_ids:
                    for lane in ts_lanes[ts_id]:
                        total_wait += traci.lane.getWaitingTime(lane)
                        total_queue += traci.lane.getLastStepHaltingNumber(lane)
                step_waiting.append(total_wait)
                step_queue.append(total_queue)

            vehicles_departed += traci.simulation.getDepartedNumber()
            vehicles_arrived += traci.simulation.getArrivedNumber()

            step += 1
            if step % delta_time == 0:
                pbar.update(1)
                pbar.set_postfix({
                    "wait": f"{step_waiting[-1]:.0f}" if step_waiting else "0",
                    "queue": f"{step_queue[-1]:.0f}" if step_queue else "0",
                })

        pbar.close()
        traci.close()

        avg_waiting = float(np.mean(step_waiting)) if step_waiting else 0.0
        avg_queue = float(np.mean(step_queue)) if step_queue else 0.0
        throughput = vehicles_arrived

        all_waiting_times.append(avg_waiting)
        all_queues.append(avg_queue)
        all_throughputs.append(throughput)

        ep_result = {
            "episode": episode + 1,
            "avg_waiting_time": avg_waiting,
            "avg_queue": avg_queue,
            "throughput": throughput,
            "steps": step,
        }
        episode_results.append(ep_result)

        logger.info(
            "Episode %d | AvgWait=%.1f | AvgQueue=%.1f | Throughput=%d",
            episode + 1, avg_waiting, avg_queue, throughput,
        )

    summary = {
        "type": "fixed-time-baseline",
        "num_episodes": num_episodes,
        "avg_waiting_time": float(np.mean(all_waiting_times)),
        "std_waiting_time": float(np.std(all_waiting_times)),
        "avg_queue": float(np.mean(all_queues)),
        "std_queue": float(np.std(all_queues)),
        "avg_throughput": float(np.mean(all_throughputs)),
        "std_throughput": float(np.std(all_throughputs)),
        "episodes": episode_results,
    }

    logger.info("=" * 60)
    logger.info("Baseline Summary:")
    logger.info("  Avg Waiting Time: %.2f +/- %.2f", summary["avg_waiting_time"], summary["std_waiting_time"])
    logger.info("  Avg Queue Length: %.2f +/- %.2f", summary["avg_queue"], summary["std_queue"])
    logger.info("  Avg Throughput:   %.0f +/- %.0f", summary["avg_throughput"], summary["std_throughput"])
    logger.info("=" * 60)

    # Save JSON
    json_path = os.path.join(output_dir, "baseline_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    csv_path = os.path.join(output_dir, "baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_results[0].keys())
        writer.writeheader()
        writer.writerows(episode_results)

    logger.info("Results saved to %s", output_dir)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fixed-time baseline")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of baseline episodes")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    run_baseline(args.config, args.episodes, args.output_dir)
