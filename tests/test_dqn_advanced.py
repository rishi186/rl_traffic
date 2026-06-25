"""Tests for new DQN features: soft target updates, LR scheduler, gradient clipping, attention."""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent, QNetwork


def _make_config(**dqn_overrides):
    """Create a minimal config dict for testing with DQN overrides."""
    dqn_cfg = {
        "learning_rate": 1e-3,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99,
        "target_update_freq": 5,
        "replay_buffer_size": 100,
        "hidden_dims": [32, 32],
        "dueling": False,
        "double_dqn": True,
        "per": False,
        "n_step": 1,
        "noisy_net": False,
        "noisy_sigma": 0.5,
        "soft_update_tau": 0.0,
        "grad_clip": 0.0,
        "lr_scheduler": "none",
        "use_attention": False,
        "num_lanes": 0,
        "num_lane_features": 5,
        "num_agents": 0,
    }
    dqn_cfg.update(dqn_overrides)
    return {
        "training": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "batch_size": 4,
            "total_episodes": 10,
            "max_steps_per_episode": 100,
            "dqn": dqn_cfg,
        },
    }


class TestSoftTargetUpdate:
    """Tests for Polyak averaging (soft target update)."""

    def test_soft_update_config(self):
        config = _make_config(soft_update_tau=0.01)
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)
        assert agent.use_soft_update is True
        assert agent.soft_update_tau == 0.01

    def test_soft_update_changes_target(self):
        config = _make_config(soft_update_tau=0.5)
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)

        # Get initial target weights
        initial_target = {k: v.clone() for k, v in agent.target_network.state_dict().items()}

        # Modify q_network weights significantly
        with torch.no_grad():
            for p in agent.q_network.parameters():
                p.add_(torch.randn_like(p) * 1.0)

        # Trigger soft update
        agent._soft_update_target()

        # Target should have changed
        updated_target = agent.target_network.state_dict()
        changed = False
        for key in initial_target:
            if not torch.allclose(initial_target[key], updated_target[key], atol=1e-6):
                changed = True
                break
        assert changed, "Target network weights should change after soft update"

    def test_hard_update_when_tau_zero(self):
        config = _make_config(soft_update_tau=0.0)
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)
        assert agent.use_soft_update is False


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_grad_clip_config(self):
        config = _make_config(grad_clip=5.0)
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)
        assert agent.grad_clip == 5.0

    def test_grad_clip_no_error(self):
        config = _make_config(grad_clip=1.0)
        agent = DQNAgent(obs_dim=4, action_dim=2, config=config)
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


class TestLRScheduler:
    """Tests for learning rate scheduler."""

    def test_cosine_scheduler(self):
        config = _make_config(lr_scheduler="cosine", lr_scheduler_eta_min=1e-6)
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)
        assert agent.lr_scheduler is not None
        assert isinstance(agent.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_scheduler(self):
        config = _make_config(lr_scheduler="step", lr_scheduler_step_size=5, lr_scheduler_gamma=0.9)
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)
        assert agent.lr_scheduler is not None
        assert isinstance(agent.lr_scheduler, torch.optim.lr_scheduler.StepLR)

    def test_no_scheduler(self):
        config = _make_config(lr_scheduler="none")
        agent = DQNAgent(obs_dim=10, action_dim=2, config=config)
        assert agent.lr_scheduler is None

    def test_lr_in_update_result(self):
        config = _make_config(lr_scheduler="cosine")
        agent = DQNAgent(obs_dim=4, action_dim=2, config=config)
        for _ in range(10):
            agent.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randint(2),
                np.random.randn(),
                np.random.randn(4).astype(np.float32),
                False,
            )
        result = agent.update()
        assert "lr" in result


class TestAttentionQNetwork:
    """Tests for attention-based Q-Network."""

    def test_attention_network_forward(self):
        num_lanes = 4
        num_lane_features = 5
        lane_dim = num_lanes * num_lane_features
        global_dim = 10  # phase one-hot + elapsed time
        obs_dim = lane_dim + global_dim

        net = QNetwork(
            obs_dim=obs_dim,
            action_dim=2,
            hidden_dims=[32, 32],
            dueling=False,
            noisy_net=False,
            noisy_sigma=0.5,
            use_attention=True,
            num_lanes=num_lanes,
            num_lane_features=num_lane_features,
        )
        obs = torch.randn(3, obs_dim)
        out = net(obs)
        assert out.shape == (3, 2)

    def test_attention_with_dueling(self):
        num_lanes = 3
        num_lane_features = 5
        obs_dim = num_lanes * num_lane_features + 8

        net = QNetwork(
            obs_dim=obs_dim,
            action_dim=3,
            hidden_dims=[32],
            dueling=True,
            noisy_net=False,
            noisy_sigma=0.5,
            use_attention=True,
            num_lanes=num_lanes,
            num_lane_features=num_lane_features,
        )
        obs = torch.randn(2, obs_dim)
        out = net(obs)
        assert out.shape == (2, 3)


class TestParameterSharing:
    """Tests for multi-agent parameter sharing."""

    def test_agent_id_embedding(self):
        num_agents = 4
        obs_dim = 20

        net = QNetwork(
            obs_dim=obs_dim,
            action_dim=2,
            hidden_dims=[32],
            dueling=False,
            noisy_net=False,
            noisy_sigma=0.5,
            num_agents=num_agents,
        )
        # Observation includes agent ID one-hot at the end
        obs = torch.randn(3, obs_dim + num_agents)
        # Append one-hot
        one_hot = torch.zeros(3, num_agents)
        one_hot[0, 0] = 1.0
        one_hot[1, 1] = 1.0
        one_hot[2, 2] = 1.0
        full_obs = torch.cat([obs[:, :obs_dim], one_hot], dim=-1)
        out = net(full_obs)
        assert out.shape == (3, 2)


class TestSaveLoadWithScheduler:
    """Test save/load preserves scheduler state."""

    def test_save_load_with_scheduler(self):
        config = _make_config(lr_scheduler="cosine")
        agent1 = DQNAgent(obs_dim=4, action_dim=2, config=config)
        obs = np.random.randn(4).astype(np.float32)

        # Do some updates to advance scheduler
        for _ in range(10):
            agent1.store_transition(
                np.random.randn(4).astype(np.float32),
                np.random.randint(2),
                np.random.randn(),
                np.random.randn(4).astype(np.float32),
                False,
            )
        agent1.update()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            agent1.save(path)

            agent2 = DQNAgent(obs_dim=4, action_dim=2, config=config)
            agent2.load(path)

            action1, q1, _ = agent1.select_action(obs, deterministic=True)
            action2, q2, _ = agent2.select_action(obs, deterministic=True)
            assert action1 == action2
            assert abs(q1 - q2) < 1e-5
