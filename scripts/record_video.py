"""Simulation video recording for SUMO traffic evaluation.

Records screenshots from SUMO GUI mode during evaluation episodes and
compiles them into a video using OpenCV or matplotlib animation.

Usage:
    python scripts/record_video.py --config config.yaml \\
        --model results/models/best_model.pth --output results/videos/eval.mp4
"""

import os
import sys
import yaml
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def record_simulation(
    config_path: str,
    model_path: str,
    output_path: str = "results/videos/simulation.mp4",
    max_steps: int = 500,
    fps: int = 10,
) -> str:
    """Run a SUMO simulation with GUI and record it as a video.

    Uses SUMO's built-in screenshot capability (--start --quit-after) or
    TraCI's GUI screenshot to capture frames, then compiles them into a video.

    Args:
        config_path: Path to config YAML.
        model_path: Path to trained model.
        output_path: Output video file path.
        max_steps: Maximum simulation steps to record.
        fps: Video frames per second.

    Returns:
        Path to the saved video file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_dir = config["experiment"]["log_dir"]
    setup_logger(log_dir=log_dir, log_level="INFO")

    # Force GUI mode for recording
    config["sumo"]["use_gui"] = True

    from src.environment.sumo_env import MultiAgentSumoEnv

    env = MultiAgentSumoEnv(config)
    observations, _ = env.reset(seed=config["experiment"]["seed"])

    obs_dim = len(observations[list(observations.keys())[0]])
    action_dim = env.action_space.n
    algorithm = config["training"].get("algorithm", "dqn")

    # Create agent
    if algorithm == "dqn":
        from src.agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim, action_dim, config)
    else:
        from src.agents.ppo_agent import PPOAgent
        device = config["experiment"].get("device", "cpu")
        agent = PPOAgent(obs_dim, action_dim, config, device)
    agent.load(model_path)

    logger.info("Recording simulation with %s agent (max %d steps)", algorithm.upper(), max_steps)

    # Try to use OpenCV for video writing
    try:
        import cv2
        has_opencv = True
    except ImportError:
        has_opencv = False
        logger.warning("OpenCV not available — will save frames as PNG images instead")

    frames_dir = os.path.join(os.path.dirname(output_path), "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    done = False

    try:
        while not done and frame_count < max_steps:
            actions = {}
            for ts_id in env.ts_ids:
                action, _, _ = agent.select_action(observations[ts_id], deterministic=True)
                actions[ts_id] = action

            observations, rewards, dones, _, infos = env.step(actions)
            done = dones["__all__"]

            # Capture screenshot via TraCI GUI
            try:
                import traci
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.png")
                traci.gui.screenshot("View #0", frame_path)
            except Exception:
                pass  # Screenshot not available in all SUMO versions

            frame_count += 1

    finally:
        env.close()

    logger.info("Captured %d frames to %s", frame_count, frames_dir)

    # Compile frames into video
    if has_opencv and frame_count > 0:
        first_frame = cv2.imread(os.path.join(frames_dir, f"frame_{0:05d}.png"))
        if first_frame is not None:
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for i in range(frame_count):
                frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    video.write(frame)

            video.release()
            logger.info("Video saved to %s", output_path)
            return output_path

    logger.info("Frames saved as PNGs in %s (compile with ffmpeg if needed)", frames_dir)
    return frames_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record SUMO simulation as video with trained RL agent"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--output", type=str, default="results/videos/simulation.mp4",
                        help="Output video file path")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum simulation steps to record")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video FPS")
    args = parser.parse_args()

    record_simulation(args.config, args.model, args.output, args.max_steps, args.fps)
