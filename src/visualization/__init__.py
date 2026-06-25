"""Visualization utilities for RL traffic signal control."""

__all__ = [
    "plot_training",
    "plot_comparison",
    "plot_generalization",
    "plot_signal_timeline",
]


def __getattr__(name: str):
    if name == "plot_training":
        from src.visualization.plot_training import plot_training
        return plot_training
    if name == "plot_comparison":
        from src.visualization.plot_comparison import plot_comparison
        return plot_comparison
    if name == "plot_generalization":
        from src.visualization.plot_generalization import plot_generalization
        return plot_generalization
    if name == "plot_signal_timeline":
        from src.visualization.plot_signal_timeline import plot_signal_timeline
        return plot_signal_timeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
