"""Line chart showing RL agent performance across density multipliers.

Visualises how the trained policy generalises to unseen traffic densities.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_generalization(
    results_path: str,
    output_dir: str = "results/figures",
    show: bool = False,
) -> None:
    """Plot generalization performance across density multipliers.

    Args:
        results_path: Path to ``generalization_results.json``.
        output_dir: Directory for output PNGs.
        show: Whether to display interactively.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    multipliers = [r["multiplier"] for r in results]
    waiting_times = [r["avg_waiting_time"] for r in results]
    queues = [r["avg_queue"] for r in results]
    improvements = [r.get("improvement_pct", None) for r in results]
    algorithm = data.get("algorithm", "DQN").upper()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Waiting time across densities ---
    ax = axes[0]
    ax.plot(multipliers, waiting_times, "o-", color="#2196F3", linewidth=2,
            markersize=6, label=algorithm)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Training density")
    ax.set_xlabel("Density Multiplier")
    ax.set_ylabel("Avg Waiting Time (s)")
    ax.set_title("Waiting Time vs Traffic Density")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Improvement % ---
    ax = axes[1]
    valid_improvements = [(m, imp) for m, imp in zip(multipliers, improvements) if imp is not None]
    if valid_improvements:
        ms, imps = zip(*valid_improvements)
        colors = ["#4CAF50" if i >= 0 else "#F44336" for i in imps]
        ax.bar(ms, imps, width=0.08, color=colors, edgecolor="white", linewidth=1.2)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xlabel("Density Multiplier")
        ax.set_ylabel("Improvement over Baseline (%)")
        ax.set_title("Wait Time Improvement vs Fixed-Time Baseline")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No baseline data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Improvement (baseline data needed)")

    fig.suptitle(f"Generalization Across {len(multipliers)} Density Configurations",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "generalization.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Generalization plot saved to {output_dir}/generalization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot generalization results")
    parser.add_argument("--results", type=str,
                        default="results/generalization/generalization_results.json")
    parser.add_argument("--output", type=str, default="results/figures")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_generalization(args.results, args.output, args.show)
