"""Auto-generate a Markdown performance report.

Combines training metrics, evaluation results, baseline comparison, and
generalization data into a single ``results/report.md`` with embedded
figure references.

Usage:
    python scripts/generate_report.py --results-dir results
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def generate_report(results_dir: str = "results") -> str:
    """Build a Markdown report from available results.

    Args:
        results_dir: Root results directory.

    Returns:
        Path to the generated report file.
    """
    lines = []

    lines.append("# RL Traffic Signal Control — Performance Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # --- Config snapshot ---
    config_path = os.path.join(results_dir, "config_snapshot.yaml")
    if os.path.exists(config_path):
        lines.append("## Experiment Configuration\n")
        lines.append("```yaml")
        with open(config_path, "r") as f:
            lines.append(f.read().strip())
        lines.append("```\n")

    # --- Training summary ---
    metrics_path = os.path.join(results_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        lines.append("## Training Summary\n")
        lines.append(f"- **Episodes**: {len(metrics)}")
        rewards = [m["reward"] for m in metrics]
        lines.append(f"- **Final Reward**: {rewards[-1]:.2f}")
        lines.append(f"- **Best Reward**: {max(rewards):.2f}")
        lines.append(f"- **Mean Reward (last 10)**: {sum(rewards[-10:]) / min(10, len(rewards)):.2f}")
        if "epsilon" in metrics[-1]:
            lines.append(f"- **Final Epsilon**: {metrics[-1]['epsilon']:.4f}")
        lines.append("")

        fig_path = "figures/training_reward.png"
        if os.path.exists(os.path.join(results_dir, fig_path)):
            lines.append(f"![Training Reward]({fig_path})\n")
        fig_path = "figures/training_waiting_time.png"
        if os.path.exists(os.path.join(results_dir, fig_path)):
            lines.append(f"![Waiting Time]({fig_path})\n")

    # --- Evaluation results ---
    eval_path = os.path.join(results_dir, "eval", "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        lines.append("## Evaluation Results\n")
        lines.append(f"- **Algorithm**: {eval_data.get('algorithm', 'N/A').upper()}")
        lines.append(f"- **Episodes**: {eval_data.get('num_episodes', 'N/A')}")
        lines.append(f"- **Avg Reward**: {eval_data['avg_reward']:.2f} ± {eval_data['std_reward']:.2f}")
        lines.append(f"- **Avg Waiting Time**: {eval_data['avg_waiting_time']:.2f} ± {eval_data['std_waiting_time']:.2f}")
        lines.append(f"- **Avg Queue Length**: {eval_data['avg_queue']:.2f} ± {eval_data['std_queue']:.2f}")
        lines.append(f"- **Avg Emergency WT**: {eval_data.get('avg_emergency_waiting', 0):.2f}")
        lines.append("")

    # --- Baseline comparison ---
    baseline_path = os.path.join(results_dir, "baseline", "baseline_results.json")
    if os.path.exists(baseline_path) and os.path.exists(eval_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        lines.append("## Baseline Comparison\n")
        lines.append("| Metric | Fixed-Time | DQN Agent | Improvement |")
        lines.append("|--------|-----------|-----------|-------------|")

        bl_wait = baseline_data["avg_waiting_time"]
        rl_wait = eval_data["avg_waiting_time"]
        wait_imp = (bl_wait - rl_wait) / bl_wait * 100 if bl_wait > 0 else 0
        lines.append(f"| Avg Waiting Time | {bl_wait:.1f} | {rl_wait:.1f} | **{wait_imp:+.1f}%** |")

        bl_queue = baseline_data["avg_queue"]
        rl_queue = eval_data["avg_queue"]
        queue_imp = (bl_queue - rl_queue) / bl_queue * 100 if bl_queue > 0 else 0
        lines.append(f"| Avg Queue Length | {bl_queue:.1f} | {rl_queue:.1f} | **{queue_imp:+.1f}%** |")

        if "avg_throughput" in baseline_data:
            lines.append(f"| Throughput | {baseline_data['avg_throughput']:.0f} | — | — |")
        lines.append("")

        fig_path = "figures/comparison_baseline.png"
        if os.path.exists(os.path.join(results_dir, fig_path)):
            lines.append(f"![Baseline Comparison]({fig_path})\n")

    # --- Generalization ---
    gen_path = os.path.join(results_dir, "generalization", "generalization_results.json")
    if os.path.exists(gen_path):
        with open(gen_path, "r") as f:
            gen_data = json.load(f)
        results = gen_data.get("results", [])
        lines.append(f"## Generalization Across {len(results)} Density Configurations\n")
        lines.append("| Density | Avg Wait Time | Avg Queue | Reward | Improvement |")
        lines.append("|---------|--------------|-----------|--------|-------------|")
        for r in results:
            imp = f"{r['improvement_pct']:+.1f}%" if "improvement_pct" in r else "N/A"
            lines.append(
                f"| {r['tag']} | {r['avg_waiting_time']:.1f} | {r['avg_queue']:.1f} "
                f"| {r['avg_reward']:.2f} | {imp} |"
            )
        lines.append("")

        fig_path = "figures/generalization.png"
        if os.path.exists(os.path.join(results_dir, fig_path)):
            lines.append(f"![Generalization]({fig_path})\n")

    # Write report
    report_content = "\n".join(lines)
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    args = parser.parse_args()
    generate_report(args.results_dir)
