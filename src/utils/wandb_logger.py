"""Weights & Biases (WandB) integration for experiment tracking.

Provides a thin wrapper that is a no-op when WandB is not installed or
disabled in config, so the rest of the codebase never needs to guard imports.
"""

from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


class WandBLogger:
    """Wrapper around wandb that degrades gracefully when unavailable.

    Args:
        config: Full experiment config dict.
        enabled: Force-enable even if wandb is installed.
    """

    def __init__(self, config: dict, enabled: bool = True) -> None:
        self.enabled = enabled and _WANDB_AVAILABLE
        self.run = None

        if not self.enabled:
            if enabled and not _WANDB_AVAILABLE:
                logger.warning("WandB requested but not installed — metrics will not be logged to WandB")
            return

        wandb_cfg = config.get("experiment", {}).get("wandb", {})
        project = wandb_cfg.get("project", "rl-traffic")
        entity = wandb_cfg.get("entity", None)
        name = config.get("experiment", {}).get("name", "experiment")
        tags = wandb_cfg.get("tags", [])

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
        )
        logger.info("WandB run initialised: project=%s, name=%s", project, name)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics.

        Args:
            metrics: Metric name -> value mapping.
            step: Optional step number.
        """
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step)

    def log_per_agent(
        self,
        prefix: str,
        agent_metrics: Dict[str, Dict[str, float]],
        step: Optional[int] = None,
    ) -> None:
        """Log per-agent metrics with agent-specific prefixes.

        Args:
            prefix: Metric group prefix (e.g. "Train").
            agent_metrics: Dict mapping agent ID to its metric dict.
            step: Optional global step.
        """
        if not self.enabled or self.run is None:
            return
        flat: Dict[str, Any] = {}
        for agent_id, metrics in agent_metrics.items():
            safe_id = agent_id.replace("/", "_").replace(":", "_")
            for key, val in metrics.items():
                flat[f"{prefix}/{safe_id}/{key}"] = val
        wandb.log(flat, step=step)

    def watch(self, model, freq: int = 100) -> None:
        """Watch model gradients and parameters.

        Args:
            model: PyTorch model.
            freq: Gradient log frequency (in optimizer steps).
        """
        if self.enabled and self.run is not None:
            wandb.watch(model, log="all", log_freq=freq)

    def finish(self) -> None:
        """Finish the WandB run."""
        if self.enabled and self.run is not None:
            wandb.finish()
            self.run = None

    @property
    def is_active(self) -> bool:
        """Whether WandB logging is active."""
        return self.enabled and self.run is not None
