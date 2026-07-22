"""Plot training curves from the training metrics JSON.

Generates:
    - Reward curve over episodes
    - Loss curve (DQN)
    - Epsilon decay curve (DQN)
    - Average waiting time over episodes
    - Average queue length over episodes
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_training(
    metrics_path: str, output_dir: str = "results/figures", show: bool = False
) -> None:
    """Generate training curve plots from metrics JSON.

    Args:
        metrics_path: Path to ``training_metrics.json``.
        output_dir: Directory to save PNG figures.
        show: Whether to display plots interactively.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(metrics_path, "r") as f:
        data = json.load(f)

    episodes = [d["episode"] for d in data]
    rewards = [d["reward"] for d in data]
    avg_waiting = [d.get("avg_waiting_time", 0) for d in data]
    avg_queue = [d.get("avg_queue", 0) for d in data]
    epsilons = [d.get("epsilon", None) for d in data]

    fig_style = {"figure.figsize": (10, 6), "axes.grid": True, "grid.alpha": 0.3}
    plt.rcParams.update(fig_style)

    # --- Reward curve ---
    fig, ax = plt.subplots()
    ax.plot(episodes, rewards, color="#2196F3", linewidth=1.5, alpha=0.7)
    # Smoothed
    if len(rewards) > 10:
        window = min(10, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], smoothed, color="#1565C0", linewidth=2.5, label="Smoothed")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Training Reward Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_reward.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # --- Waiting time ---
    fig, ax = plt.subplots()
    ax.plot(episodes, avg_waiting, color="#FF9800", linewidth=1.5, alpha=0.7)
    if len(avg_waiting) > 10:
        window = min(10, len(avg_waiting) // 5)
        smoothed = np.convolve(avg_waiting, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], smoothed, color="#E65100", linewidth=2.5, label="Smoothed")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Waiting Time (s)")
    ax.set_title("Average Waiting Time Over Training")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_waiting_time.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # --- Queue length ---
    fig, ax = plt.subplots()
    ax.plot(episodes, avg_queue, color="#4CAF50", linewidth=1.5, alpha=0.7)
    if len(avg_queue) > 10:
        window = min(10, len(avg_queue) // 5)
        smoothed = np.convolve(avg_queue, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], smoothed, color="#1B5E20", linewidth=2.5, label="Smoothed")
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Queue Length")
    ax.set_title("Average Queue Length Over Training")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_queue.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    # --- Epsilon decay (DQN only) ---
    if epsilons[0] is not None:
        fig, ax = plt.subplots()
        ax.plot(episodes, epsilons, color="#9C27B0", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("Epsilon Decay Schedule")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_epsilon.png"), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    print(f"Training plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--metrics", type=str, default="results/training_metrics.json")
    parser.add_argument("--output", type=str, default="results/figures")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_training(args.metrics, args.output, args.show)
