"""Training script for DQN / PPO traffic-signal control agents.

Supports:
    - ``algorithm: dqn`` (PyTorch DQN with experience replay)
    - ``algorithm: ppo`` (PyTorch PPO)
    - Graceful shutdown on SIGINT (saves checkpoint before exit)
    - Checkpoint recovery via ``--resume``
    - TensorBoard logging for all key metrics
    - Per-agent TensorBoard metrics
    - Early stopping with patience
    - WandB integration (optional, behind config flag)
    - Config validation before training
"""

import os
import sys
import json
import signal
import yaml
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger, get_logger
from src.utils.config_validator import validate_config, ConfigError
from src.utils.wandb_logger import WandBLogger
from src.environment.sumo_env import MultiAgentSumoEnv
from src.training.curriculum import CurriculumScheduler
from src.training.early_stopping import EarlyStopping

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Shutdown requested (signal %s). Finishing current episode…", signum)


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _create_agent(algorithm: str, obs_dim: int, action_dim: int, config: dict, device: str):
    """Instantiate the correct agent based on config.

    Args:
        algorithm: ``"dqn"`` or ``"ppo"``.
        obs_dim: Observation vector length.
        action_dim: Number of discrete actions.
        config: Full experiment config dict.
        device: Torch device string (only used for PPO).

    Returns:
        Agent instance.
    """
    if algorithm == "dqn":
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent(obs_dim, action_dim, config)
    elif algorithm == "ppo":
        import torch
        from src.agents.ppo_agent import PPOAgent
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        return PPOAgent(obs_dim, action_dim, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Model file extension helper
# ---------------------------------------------------------------------------

def _model_ext(algorithm: str) -> str:
    return ".pth"


# ---------------------------------------------------------------------------
# DQN training loop
# ---------------------------------------------------------------------------


def _train_dqn(agent, env, config, writer, log_dir: str, start_episode: int = 0,
               wandb_logger: WandBLogger = None):
    """Training loop for DQN agent.

    Args:
        agent: DQNAgent instance.
        env: MultiAgentSumoEnv instance.
        config: Config dict.
        writer: TensorBoard SummaryWriter.
        log_dir: Directory for saving models.
        start_episode: Episode to resume from.
        wandb_logger: Optional WandB logger.
    """
    total_episodes = config["training"]["total_episodes"]
    save_freq = config["training"]["save_freq"]
    log_freq = config["training"]["log_freq"]
    ext = _model_ext("dqn")
    best_reward = -np.inf
    episode_rewards = []
    episode_metrics = []

    # Early stopping
    early_stopper = EarlyStopping.from_config(config)
    if early_stopper:
        logger.info("Early stopping enabled (patience=%d, mode=%s)",
                    early_stopper.patience, early_stopper.mode)

    # Curriculum learning
    curriculum = CurriculumScheduler.from_config(config)
    if curriculum:
        logger.info("Curriculum learning enabled: %s", curriculum.summary())

    # WandB: watch model gradients
    if wandb_logger and wandb_logger.is_active:
        wandb_logger.watch(agent.q_network, freq=100)

    for episode in range(start_episode, total_episodes):
        if _shutdown_requested:
            logger.info("Saving emergency checkpoint before exit…")
            agent.save(f"{log_dir}/models/emergency_checkpoint{ext}")
            break

        # Apply curriculum density
        if curriculum:
            env.density_multiplier = curriculum.get_density()

        observations, _ = env.reset()
        episode_reward = {ts_id: 0.0 for ts_id in env.ts_ids}
        episode_waiting = []
        episode_queue = []
        episode_length = 0
        done = False

        pbar = tqdm(
            total=config["training"]["max_steps_per_episode"],
            desc=f"Episode {episode + 1}/{total_episodes}",
            leave=False,
        )

        while not done:
            actions = {}
            for ts_id in env.ts_ids:
                action, _, _ = agent.select_action(observations[ts_id])
                actions[ts_id] = action

            next_observations, rewards, dones, truncateds, infos = env.step(actions)

            # Store transitions and update
            for ts_id in env.ts_ids:
                agent.store_transition(
                    observations[ts_id],
                    actions[ts_id],
                    rewards[ts_id],
                    next_observations[ts_id],
                    dones[ts_id],
                )
                episode_reward[ts_id] += rewards[ts_id]

            # Perform gradient update every step (if buffer is ready)
            update_info = agent.update()

            # Collect per-step metrics
            total_waiting = sum(infos[ts_id]["waiting_time"] for ts_id in env.ts_ids)
            total_queue = sum(infos[ts_id]["queue"] for ts_id in env.ts_ids)
            episode_waiting.append(total_waiting)
            episode_queue.append(total_queue)

            observations = next_observations
            done = dones["__all__"]
            episode_length += 1

            pbar.update(1)
            pbar.set_postfix({
                "avg_r": f"{np.mean(list(episode_reward.values())):.1f}",
                "eps": f"{agent.epsilon:.3f}",
                "buf": len(agent.replay_buffer),
            })

        pbar.close()

        avg_episode_reward = float(np.mean(list(episode_reward.values())))
        avg_waiting = float(np.mean(episode_waiting)) if episode_waiting else 0.0
        avg_queue = float(np.mean(episode_queue)) if episode_queue else 0.0
        episode_rewards.append(avg_episode_reward)
        episode_metrics.append({
            "episode": episode + 1,
            "reward": avg_episode_reward,
            "avg_waiting_time": avg_waiting,
            "avg_queue": avg_queue,
            "episode_length": episode_length,
            "epsilon": agent.epsilon,
        })

        # Logging
        if (episode + 1) % log_freq == 0:
            logger.info(
                "Episode %d/%d | Reward=%.2f | AvgWait=%.1f | AvgQueue=%.1f | "
                "Eps=%.3f | Steps=%d | Buffer=%d",
                episode + 1, total_episodes, avg_episode_reward,
                avg_waiting, avg_queue, agent.epsilon, episode_length,
                len(agent.replay_buffer),
            )
            if update_info:
                logger.info(
                    "  Loss=%.4f | AvgQ=%.2f",
                    update_info.get("loss", 0), update_info.get("avg_q", 0),
                )

            writer.add_scalar("Train/AverageReward", avg_episode_reward, episode)
            writer.add_scalar("Train/EpisodeLength", episode_length, episode)
            writer.add_scalar("Train/Epsilon", agent.epsilon, episode)
            writer.add_scalar("Train/AvgWaitingTime", avg_waiting, episode)
            writer.add_scalar("Train/AvgQueueLength", avg_queue, episode)
            writer.add_scalar("Train/BufferSize", len(agent.replay_buffer), episode)
            if update_info:
                writer.add_scalar("Train/Loss", update_info.get("loss", 0), episode)
                writer.add_scalar("Train/AvgQ", update_info.get("avg_q", 0), episode)
                if "lr" in update_info:
                    writer.add_scalar("Train/LearningRate", update_info["lr"], episode)

            # Per-agent TensorBoard metrics
            for ts_id in env.ts_ids:
                safe_id = ts_id.replace(":", "_")
                writer.add_scalar(f"PerAgent/{safe_id}/reward", episode_reward[ts_id], episode)
                writer.add_scalar(f"PerAgent/{safe_id}/waiting_time",
                                  float(np.mean(episode_waiting)) if episode_waiting else 0.0, episode)

            # WandB logging
            if wandb_logger and wandb_logger.is_active:
                wandb_metrics = {
                    "Train/AverageReward": avg_episode_reward,
                    "Train/EpisodeLength": episode_length,
                    "Train/Epsilon": agent.epsilon,
                    "Train/AvgWaitingTime": avg_waiting,
                    "Train/AvgQueueLength": avg_queue,
                    "Train/BufferSize": len(agent.replay_buffer),
                }
                if update_info:
                    wandb_metrics["Train/Loss"] = update_info.get("loss", 0)
                    wandb_metrics["Train/AvgQ"] = update_info.get("avg_q", 0)
                    if "lr" in update_info:
                        wandb_metrics["Train/LearningRate"] = update_info["lr"]
                wandb_logger.log(wandb_metrics, step=episode)

        # Best model
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            agent.save(f"{log_dir}/models/best_model{ext}")
            logger.info("  New best model saved! Reward: %.2f", best_reward)

        # Periodic checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f"{log_dir}/models/checkpoint_ep{episode + 1}{ext}")

        # Curriculum step
        if curriculum:
            curriculum.step()
            if (episode + 1) % log_freq == 0:
                writer.add_scalar("Train/CurriculumDensity", curriculum.get_density(), episode)

        # Early stopping check
        if early_stopper:
            if early_stopper.step(avg_episode_reward):
                logger.info("Early stopping at episode %d", episode + 1)
                break

    # Final save
    agent.save(f"{log_dir}/models/final_model{ext}")

    # Save training metrics to JSON
    metrics_path = f"{log_dir}/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(episode_metrics, f, indent=2)
    logger.info("Training metrics saved to %s", metrics_path)

    return best_reward, episode_rewards


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------


def _train_ppo(agent, env, config, writer, log_dir: str, start_episode: int = 0,
               wandb_logger: WandBLogger = None):
    """Training loop for PPO agent.

    Args:
        agent: PPOAgent instance.
        env: MultiAgentSumoEnv instance.
        config: Config dict.
        writer: TensorBoard SummaryWriter.
        log_dir: Directory for saving models.
        start_episode: Episode to resume from.
        wandb_logger: Optional WandB logger.
    """
    total_episodes = config["training"]["total_episodes"]
    save_freq = config["training"]["save_freq"]
    log_freq = config["training"]["log_freq"]
    ext = _model_ext("ppo")
    best_reward = -np.inf
    episode_metrics = []

    # Early stopping
    early_stopper = EarlyStopping.from_config(config)
    if early_stopper:
        logger.info("Early stopping enabled (patience=%d, mode=%s)",
                    early_stopper.patience, early_stopper.mode)

    for episode in range(start_episode, total_episodes):
        if _shutdown_requested:
            logger.info("Saving emergency checkpoint before exit…")
            agent.save(f"{log_dir}/models/emergency_checkpoint{ext}")
            break

        observations, _ = env.reset()
        episode_reward = {ts_id: 0.0 for ts_id in env.ts_ids}
        episode_waiting = []
        episode_queue = []
        episode_length = 0
        done = False

        pbar = tqdm(
            total=config["training"]["max_steps_per_episode"],
            desc=f"Episode {episode + 1}/{total_episodes}",
            leave=False,
        )

        while not done:
            actions = {}
            for ts_id in env.ts_ids:
                action, log_prob, value = agent.select_action(observations[ts_id])
                actions[ts_id] = action

                if episode_length > 0:
                    agent.storage[ts_id][-1]["log_prob"] = log_prob
                    agent.storage[ts_id][-1]["value"] = value

            next_observations, rewards, dones, truncateds, infos = env.step(actions)

            for ts_id in env.ts_ids:
                agent.store_transition(
                    ts_id,
                    observations[ts_id],
                    actions[ts_id],
                    rewards[ts_id],
                    0.0,
                    0.0,
                    dones[ts_id],
                )
                episode_reward[ts_id] += rewards[ts_id]

            total_waiting = sum(infos[ts_id]["waiting_time"] for ts_id in env.ts_ids)
            total_queue = sum(infos[ts_id]["queue"] for ts_id in env.ts_ids)
            episode_waiting.append(total_waiting)
            episode_queue.append(total_queue)

            observations = next_observations
            done = dones["__all__"]
            episode_length += 1

            pbar.update(1)
            pbar.set_postfix({
                "avg_r": f"{np.mean(list(episode_reward.values())):.1f}",
                "step": episode_length,
            })

        pbar.close()

        update_info = agent.update()

        avg_episode_reward = float(np.mean(list(episode_reward.values())))
        avg_waiting = float(np.mean(episode_waiting)) if episode_waiting else 0.0
        avg_queue = float(np.mean(episode_queue)) if episode_queue else 0.0
        episode_metrics.append({
            "episode": episode + 1,
            "reward": avg_episode_reward,
            "avg_waiting_time": avg_waiting,
            "avg_queue": avg_queue,
            "episode_length": episode_length,
        })

        if (episode + 1) % log_freq == 0:
            logger.info(
                "Episode %d/%d | Reward=%.2f | AvgWait=%.1f | AvgQueue=%.1f | Steps=%d",
                episode + 1, total_episodes, avg_episode_reward,
                avg_waiting, avg_queue, episode_length,
            )
            if update_info:
                logger.info(
                    "  PolicyLoss=%.4f | ValueLoss=%.4f | Entropy=%.4f",
                    update_info["policy_loss"], update_info["value_loss"], update_info["entropy"],
                )

            writer.add_scalar("Train/AverageReward", avg_episode_reward, episode)
            writer.add_scalar("Train/EpisodeLength", episode_length, episode)
            writer.add_scalar("Train/AvgWaitingTime", avg_waiting, episode)
            writer.add_scalar("Train/AvgQueueLength", avg_queue, episode)
            if update_info:
                writer.add_scalar("Train/PolicyLoss", update_info["policy_loss"], episode)
                writer.add_scalar("Train/ValueLoss", update_info["value_loss"], episode)
                writer.add_scalar("Train/Entropy", update_info["entropy"], episode)

            # Per-agent TensorBoard metrics
            for ts_id in env.ts_ids:
                safe_id = ts_id.replace(":", "_")
                writer.add_scalar(f"PerAgent/{safe_id}/reward", episode_reward[ts_id], episode)

            # WandB logging
            if wandb_logger and wandb_logger.is_active:
                wandb_metrics = {
                    "Train/AverageReward": avg_episode_reward,
                    "Train/EpisodeLength": episode_length,
                    "Train/AvgWaitingTime": avg_waiting,
                    "Train/AvgQueueLength": avg_queue,
                }
                if update_info:
                    wandb_metrics["Train/PolicyLoss"] = update_info["policy_loss"]
                    wandb_metrics["Train/ValueLoss"] = update_info["value_loss"]
                    wandb_metrics["Train/Entropy"] = update_info["entropy"]
                wandb_logger.log(wandb_metrics, step=episode)

        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            agent.save(f"{log_dir}/models/best_model{ext}")
            logger.info("  New best model saved! Reward: %.2f", best_reward)

        if (episode + 1) % save_freq == 0:
            agent.save(f"{log_dir}/models/checkpoint_ep{episode + 1}{ext}")

        # Early stopping check
        if early_stopper:
            if early_stopper.step(avg_episode_reward):
                logger.info("Early stopping at episode %d", episode + 1)
                break

    agent.save(f"{log_dir}/models/final_model{ext}")

    metrics_path = f"{log_dir}/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(episode_metrics, f, indent=2)
    logger.info("Training metrics saved to %s", metrics_path)

    return best_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(config_path: str, resume: str = None):
    """Main training entry point.

    Args:
        config_path: Path to YAML config file.
        resume: Optional path to a checkpoint to resume from.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate config before anything else
    try:
        validate_config(config)
    except ConfigError as e:
        print(f"Configuration error:\n{e}")
        return

    log_dir = config["experiment"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/models", exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)

    setup_logger(log_dir=log_dir, log_level="INFO")

    algorithm = config["training"].get("algorithm", "dqn")
    device = config["experiment"]["device"]

    logger.info("=" * 60)
    logger.info("RL Traffic Signal Control — Training")
    logger.info("Algorithm: %s | Config: %s", algorithm.upper(), config_path)
    logger.info("=" * 60)

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboard import SummaryWriter  # fallback
    tb_dir = f"{log_dir}/tensorboard/{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(tb_dir)
    logger.info("TensorBoard logs: %s", tb_dir)

    # WandB (optional)
    wandb_enabled = config.get("experiment", {}).get("wandb", {}).get("enabled", False)
    wandb_logger = WandBLogger(config, enabled=wandb_enabled)

    # Save config snapshot
    config_snapshot = f"{log_dir}/config_snapshot.yaml"
    with open(config_snapshot, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Environment
    logger.info("Creating environment…")
    env = MultiAgentSumoEnv(config)
    observations, _ = env.reset(seed=config["experiment"]["seed"])

    sample_obs = observations[list(observations.keys())[0]]
    obs_dim = len(sample_obs)
    action_dim = env.action_space.n

    logger.info("Observation dim: %d | Action dim: %d | Agents: %d",
                obs_dim, action_dim, len(env.ts_ids))

    # Agent
    agent = _create_agent(algorithm, obs_dim, action_dim, config, device)

    if resume:
        logger.info("Resuming from checkpoint: %s", resume)
        agent.load(resume)

    # Log agent config
    logger.info("Agent config: %s", agent.get_config_summary())

    # Train
    if algorithm == "dqn":
        best_reward, _ = _train_dqn(agent, env, config, writer, log_dir,
                                    wandb_logger=wandb_logger)
    else:
        best_reward = _train_ppo(agent, env, config, writer, log_dir,
                                 wandb_logger=wandb_logger)

    logger.info("=" * 60)
    logger.info("Training complete! Best reward: %.2f", best_reward)
    logger.info("Models saved in: %s/models/", log_dir)
    logger.info("=" * 60)

    env.close()
    writer.close()
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN/PPO agent for traffic signal control"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    train(args.config, args.resume)
