"""Evaluation script for trained DQN / PPO traffic-signal control agents.

Runs the trained model in deterministic mode and reports per-episode and
aggregate metrics.  Results are saved as JSON for downstream analysis.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger, get_logger
from src.environment.sumo_env import MultiAgentSumoEnv

logger = get_logger(__name__)


def _create_agent(algorithm: str, obs_dim: int, action_dim: int, config: dict, device: str):
    """Instantiate agent matching the training algorithm.

    Args:
        algorithm: ``"dqn"`` or ``"ppo"``.
        obs_dim: Observation vector length.
        action_dim: Number of discrete actions.
        config: Full config dict.
        device: Torch device (PPO only).

    Returns:
        Agent instance.
    """
    if algorithm == "dqn":
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent(obs_dim, action_dim, config)
    else:
        import torch
        from src.agents.ppo_agent import PPOAgent
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        return PPOAgent(obs_dim, action_dim, config, device)


def evaluate(
    config_path: str,
    model_path: str,
    num_episodes: int = 5,
    use_gui: bool = False,
    output_dir: str = None,
) -> dict:
    """Evaluate a trained model and return aggregate metrics.

    Args:
        config_path: Path to YAML config.
        model_path: Path to saved model weights.
        num_episodes: Episodes to run.
        use_gui: Whether to launch SUMO GUI.
        output_dir: Directory to write evaluation JSON. Defaults to ``results/eval/``.

    Returns:
        Dictionary of aggregate evaluation metrics.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["sumo"]["use_gui"] = use_gui
    algorithm = config["training"].get("algorithm", "dqn")
    device = config["experiment"]["device"]
    log_dir = config["experiment"]["log_dir"]

    if output_dir is None:
        output_dir = f"{log_dir}/eval"
    os.makedirs(output_dir, exist_ok=True)

    setup_logger(log_dir=log_dir, log_level="INFO")

    logger.info("=" * 60)
    logger.info("RL Traffic Signal Control — Evaluation")
    logger.info("Algorithm: %s | Model: %s", algorithm.upper(), model_path)
    logger.info("=" * 60)

    env = MultiAgentSumoEnv(config)
    observations, _ = env.reset()

    sample_obs = observations[list(observations.keys())[0]]
    obs_dim = len(sample_obs)
    action_dim = env.action_space.n

    agent = _create_agent(algorithm, obs_dim, action_dim, config, device)
    agent.load(model_path)
    logger.info("Model loaded from: %s", model_path)

    all_rewards = []
    all_waiting_times = []
    all_queues = []
    all_emergency_waits = []
    episode_results = []

    logger.info("Evaluating for %d episodes…", num_episodes)

    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = {ts_id: 0.0 for ts_id in env.ts_ids}
        episode_waiting_time = []
        episode_queue = []
        episode_emergency = []
        done = False
        step = 0

        pbar = tqdm(
            total=config["training"]["max_steps_per_episode"],
            desc=f"Episode {episode + 1}/{num_episodes}",
            leave=True,
        )

        while not done:
            actions = {}
            for ts_id in env.ts_ids:
                action, _, _ = agent.select_action(observations[ts_id], deterministic=True)
                actions[ts_id] = action

            next_observations, rewards, dones, truncateds, infos = env.step(actions)

            total_waiting = sum(infos[ts_id]["waiting_time"] for ts_id in env.ts_ids)
            total_queue = sum(infos[ts_id]["queue"] for ts_id in env.ts_ids)
            total_emergency = sum(infos[ts_id]["emergency_waiting"] for ts_id in env.ts_ids)
            episode_waiting_time.append(total_waiting)
            episode_queue.append(total_queue)
            episode_emergency.append(total_emergency)

            for ts_id in env.ts_ids:
                episode_reward[ts_id] += rewards[ts_id]

            observations = next_observations
            done = dones["__all__"]
            step += 1

            pbar.update(1)
            pbar.set_postfix({
                "reward": f"{np.mean(list(episode_reward.values())):.2f}",
                "wait": f"{total_waiting:.0f}",
                "queue": f"{total_queue:.0f}",
            })

        pbar.close()

        avg_reward = float(np.mean(list(episode_reward.values())))
        avg_waiting = float(np.mean(episode_waiting_time))
        avg_queue = float(np.mean(episode_queue))
        avg_emergency = float(np.mean(episode_emergency))

        all_rewards.append(avg_reward)
        all_waiting_times.append(avg_waiting)
        all_queues.append(avg_queue)
        all_emergency_waits.append(avg_emergency)

        ep_result = {
            "episode": episode + 1,
            "avg_reward": avg_reward,
            "avg_waiting_time": avg_waiting,
            "avg_queue": avg_queue,
            "avg_emergency_waiting": avg_emergency,
            "steps": step,
        }
        episode_results.append(ep_result)

        logger.info(
            "Episode %d | Reward=%.2f | AvgWait=%.1f | AvgQueue=%.1f | EmergWait=%.1f",
            episode + 1, avg_reward, avg_waiting, avg_queue, avg_emergency,
        )

    summary = {
        "algorithm": algorithm,
        "model_path": model_path,
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "avg_waiting_time": float(np.mean(all_waiting_times)),
        "std_waiting_time": float(np.std(all_waiting_times)),
        "avg_queue": float(np.mean(all_queues)),
        "std_queue": float(np.std(all_queues)),
        "avg_emergency_waiting": float(np.mean(all_emergency_waits)),
        "episodes": episode_results,
    }

    logger.info("=" * 60)
    logger.info("Evaluation Summary:")
    logger.info("  Avg Reward:       %.2f +/- %.2f", summary["avg_reward"], summary["std_reward"])
    logger.info("  Avg Waiting Time: %.2f +/- %.2f", summary["avg_waiting_time"], summary["std_waiting_time"])
    logger.info("  Avg Queue Length: %.2f +/- %.2f", summary["avg_queue"], summary["std_queue"])
    logger.info("  Avg Emergency WT: %.2f", summary["avg_emergency_waiting"])
    logger.info("=" * 60)

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", results_path)

    env.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN/PPO agent for traffic signal control"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for evaluation results")
    args = parser.parse_args()

    evaluate(args.config, args.model, args.episodes, args.gui, args.output_dir)
