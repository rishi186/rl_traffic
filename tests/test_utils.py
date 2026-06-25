"""Tests for config validation, early stopping, and WandB logger."""

import os
import yaml
import pytest

from src.utils.config_validator import validate_config, ConfigError
from src.training.early_stopping import EarlyStopping


@pytest.fixture
def valid_config():
    """Load the project config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TestConfigValidator:
    """Tests for config validation module."""

    def test_valid_config_passes(self, valid_config):
        warnings = validate_config(valid_config)
        assert isinstance(warnings, list)

    def test_missing_section_raises(self):
        config = {"experiment": {"name": "test", "seed": 42, "device": "cpu", "log_dir": "results"}}
        with pytest.raises(ConfigError, match="Missing required section"):
            validate_config(config)

    def test_invalid_algorithm_raises(self, valid_config):
        valid_config["training"]["algorithm"] = "invalid"
        with pytest.raises(ConfigError, match="algorithm"):
            validate_config(valid_config)

    def test_invalid_gamma_raises(self, valid_config):
        valid_config["training"]["gamma"] = 1.5
        with pytest.raises(ConfigError, match="gamma"):
            validate_config(valid_config)

    def test_invalid_reward_type_raises(self, valid_config):
        valid_config["environment"]["reward_type"] = "unknown"
        with pytest.raises(ConfigError, match="reward_type"):
            validate_config(valid_config)

    def test_soft_update_tau_range(self, valid_config):
        valid_config["training"]["dqn"]["soft_update_tau"] = 1.5
        with pytest.raises(ConfigError, match="soft_update_tau"):
            validate_config(valid_config)

    def test_lr_scheduler_valid(self, valid_config):
        valid_config["training"]["dqn"]["lr_scheduler"] = "cosine"
        validate_config(valid_config)  # should not raise

    def test_lr_scheduler_invalid(self, valid_config):
        valid_config["training"]["dqn"]["lr_scheduler"] = "invalid"
        with pytest.raises(ConfigError, match="lr_scheduler"):
            validate_config(valid_config)

    def test_max_green_geq_min_green(self, valid_config):
        valid_config["sumo"]["max_green"] = 3
        valid_config["sumo"]["min_green"] = 5
        with pytest.raises(ConfigError, match="max_green"):
            validate_config(valid_config)


class TestEarlyStopping:
    """Tests for early stopping utility."""

    def test_no_trigger_when_improving(self):
        es = EarlyStopping(patience=3, mode="max")
        assert es.step(1.0) is False
        assert es.step(2.0) is False
        assert es.step(3.0) is False

    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(10.0)  # best=10
        assert es.step(9.0) is False  # counter=1
        assert es.step(8.0) is True   # counter=2, triggered

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(10.0)
        es.step(9.0)  # counter=1
        es.step(11.0)  # improvement, counter=0
        assert es.counter == 0
        assert es.step(10.0) is False  # counter=1

    def test_min_mode(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(5.0)  # best=5
        assert es.step(4.0) is False  # improvement
        assert es.step(5.0) is False  # counter=1
        assert es.step(6.0) is True   # counter=2, triggered

    def test_from_config_disabled(self):
        config = {"training": {"early_stopping": {"enabled": False}}}
        assert EarlyStopping.from_config(config) is None

    def test_from_config_enabled(self):
        config = {"training": {"early_stopping": {"enabled": True, "patience": 10}}}
        es = EarlyStopping.from_config(config)
        assert es is not None
        assert es.patience == 10

    def test_min_delta(self):
        es = EarlyStopping(patience=1, mode="max", min_delta=0.5)
        es.step(10.0)
        assert es.step(10.3) is True  # 10.3 - 10.0 = 0.3 < min_delta=0.5


class TestWandBLogger:
    """Tests for WandB logger wrapper."""

    def test_disabled_when_not_installed(self):
        config = {"experiment": {"name": "test", "wandb": {"enabled": True}}}
        from src.utils.wandb_logger import WandBLogger, _WANDB_AVAILABLE
        logger = WandBLogger(config, enabled=True)
        if not _WANDB_AVAILABLE:
            assert logger.is_active is False
        else:
            # If wandb is available, just check it doesn't crash
            assert logger is not None

    def test_disabled_by_config(self):
        config = {"experiment": {"name": "test"}}
        from src.utils.wandb_logger import WandBLogger
        logger = WandBLogger(config, enabled=False)
        assert logger.is_active is False

    def test_log_no_op_when_disabled(self):
        config = {"experiment": {"name": "test"}}
        from src.utils.wandb_logger import WandBLogger
        logger = WandBLogger(config, enabled=False)
        logger.log({"metric": 1.0}, step=0)  # should not raise
