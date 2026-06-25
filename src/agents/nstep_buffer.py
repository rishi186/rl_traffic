"""N-Step Return wrapper for experience replay buffers.

Accumulates n-step bootstrapped returns before pushing transitions to the
underlying replay buffer.  This speeds up credit assignment and improves
sample efficiency.
"""

import numpy as np
from collections import deque
from typing import Any


class NStepCollector:
    """Collects transitions and computes n-step returns before forwarding
    to a base replay buffer.

    Args:
        base_buffer: Underlying replay buffer (uniform or PER).
        n_step: Number of steps for return computation.
        gamma: Discount factor.
    """

    def __init__(self, base_buffer: Any, n_step: int = 3, gamma: float = 0.99) -> None:
        self.base_buffer = base_buffer
        self.n_step = n_step
        self.gamma = gamma
        self._buffer: deque = deque(maxlen=n_step)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single-step transition.  Once n transitions have accumulated,
        compute the n-step return and push the resulting transition to the
        base buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Immediate reward.
            next_state: Next observation.
            done: Episode termination flag.
        """
        self._buffer.append((state, action, reward, next_state, done))

        if done:
            # Flush all remaining transitions in the deque
            while len(self._buffer) > 0:
                self._flush_one()
        elif len(self._buffer) == self.n_step:
            self._flush_one()

    def _flush_one(self) -> None:
        """Compute n-step return for the oldest transition and push it."""
        if len(self._buffer) == 0:
            return

        state, action, _, _, _ = self._buffer[0]

        # Compute discounted n-step return
        n_step_return = 0.0
        for i in range(len(self._buffer)):
            _, _, r, _, d = self._buffer[i]
            n_step_return += (self.gamma ** i) * r
            if d:
                break

        # The "next_state" is the state after n steps
        _, _, _, last_next_state, last_done = self._buffer[-1]

        self.base_buffer.push(state, action, n_step_return, last_next_state, last_done)
        self._buffer.popleft()

    def reset(self) -> None:
        """Clear the n-step accumulator (call at episode boundaries)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Delegate common buffer interface to base_buffer
    # ------------------------------------------------------------------

    def sample(self, *args, **kwargs):
        return self.base_buffer.sample(*args, **kwargs)

    def is_ready(self, batch_size: int) -> bool:
        return self.base_buffer.is_ready(batch_size)

    def update_priorities(self, *args, **kwargs):
        if hasattr(self.base_buffer, "update_priorities"):
            return self.base_buffer.update_priorities(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.base_buffer)

    @property
    def capacity(self) -> int:
        return self.base_buffer.capacity

    def clear(self) -> None:
        self._buffer.clear()
        self.base_buffer.clear()

    def get_stats(self):
        stats = self.base_buffer.get_stats()
        stats["n_step"] = self.n_step
        return stats
