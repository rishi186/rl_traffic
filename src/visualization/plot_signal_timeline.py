"""Per-intersection signal phase timeline visualisation.

Requires data collected during evaluation (phase log).  Generates a
Gantt-style chart showing green/yellow/red phases over time for each
traffic light.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional


def plot_signal_timeline(
    phase_log: List[Dict],
    ts_ids: Optional[List[str]] = None,
    max_steps: int = 200,
    output_dir: str = "results/figures",
    show: bool = False,
) -> None:
    """Plot signal phase timelines.

    Args:
        phase_log: List of dicts with keys ``step``, ``ts_id``, ``phase_state``.
                   ``phase_state`` should be a SUMO phase string like ``"GGrrGGrr"``.
        ts_ids: Subset of traffic lights to plot. If None, plots all.
        max_steps: Maximum steps to include.
        output_dir: Directory for output PNGs.
        show: Whether to display interactively.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not phase_log:
        print("No phase log data provided. Skipping signal timeline plot.")
        return

    # Organise by ts_id
    timeline: Dict[str, List] = {}
    for entry in phase_log:
        if entry["step"] > max_steps:
            break
        tid = entry["ts_id"]
        if ts_ids and tid not in ts_ids:
            continue
        if tid not in timeline:
            timeline[tid] = []
        timeline[tid].append(entry)

    if not timeline:
        print("No matching traffic lights in phase log.")
        return

    ids = sorted(timeline.keys())[:8]  # Limit to 8 for readability


    fig, ax = plt.subplots(figsize=(14, max(3, len(ids) * 0.8)))

    for row, tid in enumerate(ids):
        entries = timeline[tid]
        for entry in entries:
            step = entry["step"]
            state = entry["phase_state"]
            # Determine dominant colour
            greens = sum(1 for c in state if c in "Gg")
            yellows = sum(1 for c in state if c in "Yy")
            total = len(state) if state else 1
            if greens / total > 0.3:
                color = "#4CAF50"
            elif yellows / total > 0.1:
                color = "#FFC107"
            else:
                color = "#F44336"
            ax.barh(row, 1, left=step, height=0.6, color=color, edgecolor="none")

    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xlabel("Simulation Step")
    ax.set_title("Traffic Signal Phase Timeline")
    ax.invert_yaxis()

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Green"),
        mpatches.Patch(color="#FFC107", label="Yellow"),
        mpatches.Patch(color="#F44336", label="Red"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "signal_timeline.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Signal timeline saved to {output_dir}/signal_timeline.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot signal phase timeline")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to phase_log.json")
    parser.add_argument("--output", type=str, default="results/figures")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    with open(args.log, "r") as f:
        log_data = json.load(f)
    plot_signal_timeline(log_data, max_steps=args.max_steps,
                         output_dir=args.output, show=args.show)
