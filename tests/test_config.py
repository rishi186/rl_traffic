"""Tests for configuration parsing and validation."""

import os
import yaml
import pytest


@pytest.fixture
def config():
    """Load the project config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TestConfigParsing:
    """Validate config structure and values."""

    def test_experiment_section_exists(self, config):
        assert "experiment" in config
        assert "name" in config["experiment"]
        assert "seed" in config["experiment"]
        assert "log_dir" in config["experiment"]

    def test_sumo_section_exists(self, config):
        assert "sumo" in config
        assert "cfg_file" in config["sumo"]
        assert "delta_time" in config["sumo"]
        assert "yellow_time" in config["sumo"]
        assert "min_green" in config["sumo"]
        assert "max_green" in config["sumo"]

    def test_density_multiplier_default(self, config):
        mult = config["sumo"].get("density_multiplier", 1.0)
        assert 0.0 < mult <= 10.0

    def test_environment_section(self, config):
        assert "environment" in config
        assert "reward_type" in config["environment"]
        valid_types = ["diff-waiting-time", "queue", "pressure", "custom-shaped"]
        assert config["environment"]["reward_type"] in valid_types

    def test_reward_weights(self, config):
        rw = config["environment"].get("reward_weights", {})
        assert "queue_weight" in rw
        assert "waiting_time_weight" in rw
        assert "emergency_weight" in rw
        # Weights should be positive
        assert rw["queue_weight"] > 0
        assert rw["waiting_time_weight"] > 0
        assert rw["emergency_weight"] >= 0

    def test_training_section(self, config):
        t = config["training"]
        assert "algorithm" in t
        assert t["algorithm"] in ["dqn", "ppo"]
        assert "total_episodes" in t
        assert "batch_size" in t
        assert "gamma" in t
        assert 0 < t["gamma"] <= 1.0

    def test_dqn_config(self, config):
        dqn = config["training"].get("dqn", {})
        assert "epsilon_start" in dqn
        assert "epsilon_end" in dqn
        assert "epsilon_decay" in dqn
        assert "target_update_freq" in dqn
        assert "replay_buffer_size" in dqn
        assert "hidden_dims" in dqn
        assert isinstance(dqn["hidden_dims"], list)
        assert all(isinstance(d, int) for d in dqn["hidden_dims"])
        assert dqn["epsilon_start"] >= dqn["epsilon_end"]

    def test_evaluation_section(self, config):
        assert "evaluation" in config
        assert "num_episodes" in config["evaluation"]

    def test_ppo_config(self, config):
        ppo = config["training"].get("ppo", {})
        assert "gae_lambda" in ppo
        assert "clip_epsilon" in ppo
        assert "entropy_coef" in ppo
        assert "ppo_epochs" in ppo
        assert 0 < ppo["clip_epsilon"] < 1.0
        assert 0 < ppo["gae_lambda"] <= 1.0

    def test_density_variants(self, config):
        dv = config.get("density_variants", {})
        multipliers = dv.get("multipliers", [])
        assert len(multipliers) >= 12
        assert 1.0 in multipliers
        assert all(m > 0 for m in multipliers)
