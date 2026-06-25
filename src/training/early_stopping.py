"""Early stopping utility for RL training loops.

Monitors a metric (e.g. average reward) and stops training if no improvement
is seen for a configurable number of episodes.
"""

from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping monitor with patience-based triggering.

    Args:
        patience: Number of episodes without improvement before stopping.
        min_delta: Minimum change in metric to qualify as an improvement.
        mode: ``"max"`` (higher is better) or ``"min"`` (lower is better).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = -float("inf") if mode == "max" else float("inf")
        self.counter: int = 0
        self.stopped: bool = False

    @classmethod
    def from_config(cls, config: dict) -> Optional["EarlyStopping"]:
        """Create from config dict, or None if disabled.

        Args:
            config: Full experiment config dict.

        Returns:
            EarlyStopping instance or None.
        """
        es_cfg = config.get("training", {}).get("early_stopping", {})
        if not es_cfg.get("enabled", False):
            return None
        return cls(
            patience=es_cfg.get("patience", 15),
            min_delta=es_cfg.get("min_delta", 0.0),
            mode=es_cfg.get("mode", "max"),
        )

    def step(self, metric: float) -> bool:
        """Feed a new metric value.  Returns True if training should stop.

        Args:
            metric: The metric value for the current episode.

        Returns:
            True if early stopping is triggered, False otherwise.
        """
        if self.stopped:
            return True

        improved = (
            (metric > self.best + self.min_delta)
            if self.mode == "max"
            else (metric < self.best - self.min_delta)
        )

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                logger.info(
                    "Early stopping triggered (no improvement for %d episodes, "
                    "best=%.4f, mode=%s)",
                    self.patience, self.best, self.mode,
                )
                return True
        return False

    @property
    def is_triggered(self) -> bool:
        """Whether early stopping has been triggered."""
        return self.stopped
