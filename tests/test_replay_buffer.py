"""Tests for the experience replay buffer."""

import numpy as np
import pytest

from src.agents.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    """Unit tests for ReplayBuffer."""

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100, obs_dim=4)
        assert len(buf) == 0

        buf.push(np.zeros(4), 0, 1.0, np.ones(4), False)
        assert len(buf) == 1

        for i in range(50):
            buf.push(np.random.randn(4), i % 2, float(i), np.random.randn(4), False)
        assert len(buf) == 51

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10, obs_dim=2)
        for i in range(25):
            buf.push(np.array([i, i]), 0, 0.0, np.array([i, i]), False)
        assert len(buf) == 10

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, obs_dim=5)
        for _ in range(20):
            buf.push(np.random.randn(5), 1, 0.5, np.random.randn(5), True)

        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 5)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 5)
        assert dones.shape == (8,)

    def test_sample_raises_on_insufficient_data(self):
        buf = ReplayBuffer(capacity=100, obs_dim=3)
        buf.push(np.zeros(3), 0, 0.0, np.zeros(3), False)
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_is_ready(self):
        buf = ReplayBuffer(capacity=100, obs_dim=2)
        assert not buf.is_ready(1)
        buf.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
        assert buf.is_ready(1)
        assert not buf.is_ready(2)

    def test_clear(self):
        buf = ReplayBuffer(capacity=100, obs_dim=2)
        for _ in range(10):
            buf.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
        buf.clear()
        assert len(buf) == 0

    def test_get_stats(self):
        buf = ReplayBuffer(capacity=50, obs_dim=2)
        for _ in range(25):
            buf.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
        stats = buf.get_stats()
        assert stats["size"] == 25
        assert stats["capacity"] == 50
        assert abs(stats["utilization"] - 0.5) < 1e-6

    def test_sample_dtypes(self):
        buf = ReplayBuffer(capacity=100, obs_dim=3)
        for _ in range(10):
            buf.push(np.random.randn(3), 1, 0.5, np.random.randn(3), True)
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert states.dtype == np.float32
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32
