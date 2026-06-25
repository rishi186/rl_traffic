"""Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and supports
uniform random sampling for mini-batch updates.
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, Any


class ReplayBuffer:
    """Fixed-capacity experience replay buffer with uniform sampling.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimensionality of the observation vector.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            state: Current observation vector.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation vector.
            done: Whether the episode terminated.
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.

        Raises:
            ValueError: If batch_size exceeds buffer length.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size {batch_size} exceeds buffer length {len(self.buffer)}"
            )

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        actions = np.array([self.buffer[i][1] for i in indices], dtype=np.int64)
        rewards = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        next_states = np.array([self.buffer[i][3] for i in indices], dtype=np.float32)
        dones = np.array([self.buffer[i][4] for i in indices], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough samples for a batch.

        Args:
            batch_size: Required batch size.

        Returns:
            True if buffer length >= batch_size.
        """
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Remove all transitions from the buffer."""
        self.buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics.

        Returns:
            Dictionary with size, capacity, and utilization.
        """
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity if self.capacity > 0 else 0.0,
        }
