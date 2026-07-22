"""Tests for the DQN agent."""

import os
import tempfile
import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent


def _make_config(dueling=False):
    """Create a minimal config dict for testing."""
    return {
        "training": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "batch_size": 4,
            "dqn": {
                "learning_rate": 1e-3,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.99,
                "target_update_freq": 5,
                "replay_buffer_size": 100,
                "hidden_dims": [32, 32],
                "dueling": dueling,
            },
        },
    }


class TestDQNAgent:
    """Unit tests for DQNAgent."""

    def test_init_standard(self):
        agent = DQNAgent(obs_dim=10, action_dim=2, config=_make_config())
        assert agent.obs_dim == 10
        assert agent.action_dim == 2
        assert agent.dueling is False

    def test_init_dueling(self):
        agent = DQNAgent(obs_dim=10, action_dim=4, config=_make_config(dueling=True))
        assert agent.dueling is True

    def test_select_action_shape(self):
        agent = DQNAgent(obs_dim=5, action_dim=3, config=_make_config())
        obs = np.random.randn(5).astype(np.float32)
        action, q_val, max_q = agent.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 3
        assert isinstance(q_val, float)
        assert isinstance(max_q, float)

    def test_select_action_deterministic(self):
        agent = DQNAgent(obs_dim=5, action_dim=3, config=_make_config())
        agent.epsilon = 0.0  # Force greedy
        obs = np.random.randn(5).astype(np.float32)
        actions = [agent.select_action(obs, deterministic=True)[0] for _ in range(20)]
        # All actions should be the same when deterministic
        assert len(set(actions)) == 1

    def test_store_transition(self):
        agent = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
        agent.store_transition(np.zeros(4), 0, 1.0, np.ones(4), False)
        assert len(agent.replay_buffer) == 1
        assert agent.total_steps == 1

    def test_update_returns_empty_when_buffer_small(self):
        agent = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
        agent.store_transition(np.zeros(4), 0, 1.0, np.ones(4), False)
        result = agent.update()
        assert result == {}

    def test_update_returns_metrics(self):
        agent = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
        # Fill buffer above batch_size
        for _ in range(10):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randint(2),
                np.random.randn(),
                np.random.randn(4).astype(np.float32),
                False,
            )
        result = agent.update()
        assert "loss" in result
        assert "epsilon" in result
        assert "avg_q" in result
        assert "buffer_size" in result

    def test_epsilon_decay(self):
        agent = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
        initial_eps = agent.epsilon
        for _ in range(10):
            agent.store_transition(
                np.random.randn(4).astype(np.float32), 0, 0.0,
                np.random.randn(4).astype(np.float32), False,
            )
        agent.update()
        assert agent.epsilon < initial_eps

    def test_save_load_roundtrip(self):
        agent1 = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
        obs = np.random.randn(4).astype(np.float32)
        action1, q1, _ = agent1.select_action(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            agent1.save(path)

            agent2 = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
            agent2.load(path)

            action2, q2, _ = agent2.select_action(obs, deterministic=True)
            assert action1 == action2
            assert abs(q1 - q2) < 1e-5

    def test_network_output_shape(self):
        agent = DQNAgent(obs_dim=8, action_dim=3, config=_make_config())
        obs = torch.tensor(np.random.randn(1, 8).astype(np.float32))
        agent.q_network.eval()
        with torch.no_grad():
            q_values = agent.q_network(obs)
        assert q_values.shape == (1, 3)

    def test_get_config_summary(self):
        agent = DQNAgent(obs_dim=4, action_dim=2, config=_make_config())
        summary = agent.get_config_summary()
        assert "algorithm" in summary
        assert "obs_dim" in summary
        assert summary["obs_dim"] == 4
