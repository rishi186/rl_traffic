"""Bar charts comparing DQN agent vs fixed-time baseline.

Generates side-by-side comparison for waiting time, queue length,
and throughput metrics.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
    eval_path: str,
    baseline_path: str,
    output_dir: str = "results/figures",
    show: bool = False,
) -> None:
    """Generate comparison bar charts.

    Args:
        eval_path: Path to ``eval_results.json`` (RL agent).
        baseline_path: Path to ``baseline_results.json``.
        output_dir: Directory for output PNGs.
        show: Whether to display interactively.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)

    algorithm = eval_data.get("algorithm", "DQN").upper()

    metrics = ["avg_waiting_time", "avg_queue"]
    labels = ["Avg Waiting Time (s)", "Avg Queue Length"]
    colors_rl = ["#2196F3", "#4CAF50"]
    colors_bl = ["#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        rl_val = eval_data.get(metric, 0)
        rl_std = eval_data.get(f"std_{metric.replace('avg_', '')}", 0)
        bl_val = baseline_data.get(metric, 0)
        bl_std = baseline_data.get(f"std_{metric.replace('avg_', '')}", 0)

        x = np.arange(2)
        ax.bar(x, [bl_val, rl_val], yerr=[bl_std, rl_std],
               color=[colors_bl[i], colors_rl[i]], capsize=5, width=0.5,
               edgecolor="white", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Fixed-Time", algorithm])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

        # Annotate improvement
        if bl_val > 0:
            improvement = (bl_val - rl_val) / bl_val * 100
            ax.annotate(
                f"{improvement:+.1f}%",
                xy=(1, rl_val), xytext=(1.25, (bl_val + rl_val) / 2),
                fontsize=12, fontweight="bold", color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0"),
            )

    fig.suptitle(f"{algorithm} vs Fixed-Time Baseline", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_baseline.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Comparison plot saved to {output_dir}/comparison_baseline.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot baseline comparison")
    parser.add_argument("--eval", type=str, default="results/eval/eval_results.json")
    parser.add_argument("--baseline", type=str, default="results/baseline/baseline_results.json")
    parser.add_argument("--output", type=str, default="results/figures")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plot_comparison(args.eval, args.baseline, args.output, args.show)
