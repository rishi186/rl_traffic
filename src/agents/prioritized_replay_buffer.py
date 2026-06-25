"""Prioritized Experience Replay (PER) buffer with SumTree.

Implements proportional prioritization (Schaul et al., 2015) using a binary
sum-tree for O(log n) sampling.  Higher TD-error transitions are sampled
more frequently, with importance-sampling weights to correct the bias.
"""

import numpy as np
from typing import Tuple, Dict, Any


class SumTree:
    """Binary tree where each leaf holds a priority and parent nodes store
    the sum of their children.  Enables O(log n) proportional sampling.

    Args:
        capacity: Maximum number of leaf nodes (transitions).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Update parent nodes after a priority change."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find the leaf index corresponding to *value* in [0, total)."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        """Sum of all priorities."""
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> None:
        """Insert or overwrite a transition with the given priority."""
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        """Set priority of a specific tree node and propagate."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float, Any]:
        """Sample one leaf proportional to stored priorities.

        Args:
            value: Uniform random sample in [0, total).

        Returns:
            (tree_index, priority, data)
        """
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def max_priority(self) -> float:
        """Highest priority currently stored (among populated leaves)."""
        start = self.capacity - 1
        end = start + self.size
        if self.size == 0:
            return 1.0
        return float(np.max(self.tree[start:end]))

    @property
    def min_priority(self) -> float:
        """Lowest priority currently stored (among populated leaves)."""
        start = self.capacity - 1
        end = start + self.size
        if self.size == 0:
            return 1.0
        leaves = self.tree[start:end]
        nonzero = leaves[leaves > 0]
        if len(nonzero) == 0:
            return 1.0
        return float(np.min(nonzero))


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Args:
        capacity: Maximum transitions.
        obs_dim: Observation vector dimensionality.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta_start: Initial importance-sampling exponent.
        beta_frames: Number of frames over which beta is annealed to 1.0.
        epsilon: Small constant added to TD-error to ensure non-zero priority.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0

        self.tree = SumTree(capacity)

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (annealed linearly to 1)."""
        frac = min(self.frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum current priority."""
        priority = self.tree.max_priority ** self.alpha
        if priority == 0.0:
            priority = 1.0
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )
        self.tree.add(priority, transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a prioritized mini-batch.

        Args:
            batch_size: Number of transitions.

        Returns:
            (states, actions, rewards, next_states, dones, is_weights, tree_indices)
            where is_weights are importance-sampling correction weights.
        """
        self.frame += 1

        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        transitions = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx, prio, data = self.tree.get(value)
            indices[i] = idx
            priorities[i] = prio
            transitions.append(data)

        # Importance-sampling weights
        total = self.tree.total
        min_prob = self.tree.min_priority / total if total > 0 else 1e-8
        max_weight = (min_prob * len(self)) ** (-self.beta)

        probs = priorities / total
        is_weights = (probs * len(self)) ** (-self.beta)
        is_weights /= max_weight  # normalise so max weight = 1

        states = np.array([t[0] for t in transitions], dtype=np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        dones = np.array([t[4] for t in transitions], dtype=np.float32)

        return (
            states, actions, rewards, next_states, dones,
            is_weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new TD-errors.

        Args:
            tree_indices: SumTree indices returned by sample().
            td_errors: Absolute TD-errors for each sampled transition.
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, prio in zip(tree_indices, priorities):
            self.tree.update(int(idx), float(prio))

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self) >= batch_size

    def __len__(self) -> int:
        return self.tree.size

    def clear(self) -> None:
        """Remove all transitions."""
        self.tree = SumTree(self.capacity)
        self.frame = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        return {
            "size": len(self),
            "capacity": self.capacity,
            "utilization": len(self) / self.capacity if self.capacity > 0 else 0.0,
            "beta": self.beta,
            "total_priority": self.tree.total,
        }
