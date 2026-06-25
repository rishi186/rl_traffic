"""Curriculum learning scheduler for traffic density.

Starts training with easier (lower density) traffic and progressively
increases to harder (higher density) scenarios.  This stabilises early
learning and leads to more robust policies.
"""

from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CurriculumScheduler:
    """Density curriculum that advances through levels every N episodes.

    Args:
        start_density: Initial density multiplier (easy).
        end_density: Final density multiplier (hard).
        step_size: Density increment per curriculum level.
        episodes_per_level: Episodes at each density before advancing.
    """

    def __init__(
        self,
        start_density: float = 0.5,
        end_density: float = 1.5,
        step_size: float = 0.1,
        episodes_per_level: int = 10,
    ) -> None:
        self.start_density = start_density
        self.end_density = end_density
        self.step_size = step_size
        self.episodes_per_level = episodes_per_level

        self.current_density = start_density
        self.level = 0
        self._episodes_at_level = 0

        # Pre-compute all levels
        self.levels = []
        d = start_density
        while d <= end_density + 1e-6:
            self.levels.append(round(d, 4))
            d += step_size
        if not self.levels:
            self.levels = [1.0]

        logger.info(
            "Curriculum: %d levels from %.2f to %.2f (step=%.2f, episodes/level=%d)",
            len(self.levels), start_density, end_density,
            step_size, episodes_per_level,
        )

    @classmethod
    def from_config(cls, config: dict) -> Optional["CurriculumScheduler"]:
        """Create scheduler from config, or None if disabled.

        Args:
            config: Full experiment config dict.

        Returns:
            CurriculumScheduler or None.
        """
        curr_cfg = config.get("curriculum", {})
        if not curr_cfg.get("enabled", False):
            return None
        return cls(
            start_density=curr_cfg.get("start_density", 0.5),
            end_density=curr_cfg.get("end_density", 1.5),
            step_size=curr_cfg.get("step_size", 0.1),
            episodes_per_level=curr_cfg.get("episodes_per_level", 10),
        )

    def get_density(self) -> float:
        """Return the current density multiplier."""
        return self.current_density

    def step(self) -> bool:
        """Call after each episode.  Advances level if threshold reached.

        Returns:
            True if the density level changed.
        """
        self._episodes_at_level += 1
        if (
            self._episodes_at_level >= self.episodes_per_level
            and self.level < len(self.levels) - 1
        ):
            self.level += 1
            self.current_density = self.levels[self.level]
            self._episodes_at_level = 0
            logger.info(
                "Curriculum advanced to level %d / %d (density=%.2f)",
                self.level + 1, len(self.levels), self.current_density,
            )
            return True
        return False

    @property
    def progress(self) -> float:
        """Fraction of curriculum completed (0.0 to 1.0)."""
        if len(self.levels) <= 1:
            return 1.0
        return self.level / (len(self.levels) - 1)

    @property
    def is_complete(self) -> bool:
        """Whether the final density level has been reached."""
        return self.level >= len(self.levels) - 1

    def summary(self) -> dict:
        """Return curriculum state for logging."""
        return {
            "level": self.level + 1,
            "total_levels": len(self.levels),
            "current_density": self.current_density,
            "progress": f"{self.progress:.1%}",
            "is_complete": self.is_complete,
        }
