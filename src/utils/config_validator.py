"""Configuration validation for the RL Traffic project.

Validates the experiment config.yaml against expected schema, value ranges,
and cross-field constraints.  Raises ``ConfigError`` with a descriptive
message when validation fails.
"""

from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigError(Exception):
    """Raised when the experiment configuration is invalid."""


# -----------------------------------------------------------------------
# Schema: section -> required keys and their types
# -----------------------------------------------------------------------

_REQUIRED_SCHEMA = {
    "experiment": {
        "name": str,
        "seed": int,
        "device": str,
        "log_dir": str,
    },
    "sumo": {
        "cfg_file": str,
        "use_gui": bool,
        "num_seconds": int,
        "delta_time": int,
        "yellow_time": int,
        "min_green": int,
        "max_green": int,
        "begin_time": int,
        "density_multiplier": (int, float),
    },
    "environment": {
        "reward_type": str,
        "observation_type": str,
        "add_system_info": bool,
        "add_per_agent_info": bool,
        "normalize_observations": bool,
    },
    "training": {
        "algorithm": str,
        "total_episodes": int,
        "max_steps_per_episode": int,
        "batch_size": int,
        "learning_rate": (int, float),
        "gamma": (int, float),
        "save_freq": int,
        "eval_freq": int,
        "log_freq": int,
    },
}

_VALID_ALGORITHMS = {"dqn", "ppo"}
_VALID_REWARD_TYPES = {"diff-waiting-time", "queue", "pressure", "custom-shaped"}
_VALID_DEVICES = {"cpu", "cuda", "mps"}
_VALID_LR_SCHEDULERS = {"none", "cosine", "step"}


def validate_config(config: dict) -> List[str]:
    """Validate the experiment configuration.

    Args:
        config: Full config dict loaded from YAML.

    Returns:
        List of warning strings (non-fatal issues).

    Raises:
        ConfigError: If any required field is missing or has an invalid value.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # --- Check required sections exist ---
    for section, keys in _REQUIRED_SCHEMA.items():
        if section not in config:
            errors.append(f"Missing required section: '{section}'")
            continue
        for key, expected_type in keys.items():
            if key not in config[section]:
                errors.append(f"Missing required key: '{section}.{key}'")
            elif not isinstance(config[section][key], expected_type):
                actual = type(config[section][key]).__name__
                expected = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else " | ".join(t.__name__ for t in expected_type)
                )
                errors.append(
                    f"Invalid type for '{section}.{key}': "
                    f"expected {expected}, got {actual}"
                )

    if errors:
        raise ConfigError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    # --- Value range checks ---
    c = config

    if c["training"]["algorithm"] not in _VALID_ALGORITHMS:
        errors.append(
            f"training.algorithm must be one of {_VALID_ALGORITHMS}, "
            f"got '{c['training']['algorithm']}'"
        )

    if c["experiment"]["device"] not in _VALID_DEVICES:
        warnings.append(
            f"experiment.device='{c['experiment']['device']}' is not in {_VALID_DEVICES}; "
            "will fall back to 'cpu'"
        )

    if c["environment"]["reward_type"] not in _VALID_REWARD_TYPES:
        errors.append(
            f"environment.reward_type must be one of {_VALID_REWARD_TYPES}, "
            f"got '{c['environment']['reward_type']}'"
        )

    if not (0 < c["training"]["gamma"] <= 1.0):
        errors.append(
            f"training.gamma must be in (0, 1], got {c['training']['gamma']}"
        )

    if c["training"]["total_episodes"] <= 0:
        errors.append("training.total_episodes must be > 0")

    if c["training"]["batch_size"] <= 0:
        errors.append("training.batch_size must be > 0")

    if c["training"]["learning_rate"] <= 0:
        errors.append("training.learning_rate must be > 0")

    if c["sumo"]["delta_time"] <= 0:
        errors.append("sumo.delta_time must be > 0")

    if c["sumo"]["min_green"] <= 0:
        errors.append("sumo.min_green must be > 0")

    if c["sumo"]["max_green"] < c["sumo"]["min_green"]:
        errors.append("sumo.max_green must be >= sumo.min_green")

    if not (0 < c["sumo"]["density_multiplier"] <= 10.0):
        errors.append(
            f"sumo.density_multiplier must be in (0, 10], "
            f"got {c['sumo']['density_multiplier']}"
        )

    # --- Reward weights ---
    rw = c["environment"].get("reward_weights", {})
    for wkey in ("queue_weight", "waiting_time_weight", "emergency_weight"):
        if wkey in rw and rw[wkey] < 0:
            errors.append(f"environment.reward_weights.{wkey} must be >= 0")

    # --- DQN-specific ---
    dqn = c["training"].get("dqn", {})
    if dqn:
        if "epsilon_start" in dqn and "epsilon_end" in dqn:
            if dqn["epsilon_start"] < dqn["epsilon_end"]:
                errors.append("dqn.epsilon_start must be >= dqn.epsilon_end")
        if "epsilon_decay" in dqn:
            if not (0 < dqn["epsilon_decay"] <= 1.0):
                errors.append("dqn.epsilon_decay must be in (0, 1]")
        if "replay_buffer_size" in dqn and dqn["replay_buffer_size"] <= 0:
            errors.append("dqn.replay_buffer_size must be > 0")
        if "hidden_dims" in dqn:
            if not isinstance(dqn["hidden_dims"], list) or len(dqn["hidden_dims"]) == 0:
                errors.append("dqn.hidden_dims must be a non-empty list")
        if "soft_update_tau" in dqn:
            if not (0.0 <= dqn["soft_update_tau"] <= 1.0):
                errors.append("dqn.soft_update_tau must be in [0, 1]")
        if "lr_scheduler" in dqn:
            if dqn["lr_scheduler"] not in _VALID_LR_SCHEDULERS:
                errors.append(
                    f"dqn.lr_scheduler must be one of {_VALID_LR_SCHEDULERS}, "
                    f"got '{dqn['lr_scheduler']}'"
                )
        if "grad_clip" in dqn and dqn["grad_clip"] < 0:
            errors.append("dqn.grad_clip must be >= 0")

    # --- PPO-specific ---
    ppo = c["training"].get("ppo", {})
    if ppo:
        if "clip_epsilon" in ppo and not (0 < ppo["clip_epsilon"] < 1.0):
            errors.append("ppo.clip_epsilon must be in (0, 1)")
        if "gae_lambda" in ppo and not (0 < ppo["gae_lambda"] <= 1.0):
            errors.append("ppo.gae_lambda must be in (0, 1]")
        if "ppo_epochs" in ppo and ppo["ppo_epochs"] <= 0:
            errors.append("ppo.ppo_epochs must be > 0")

    # --- Early stopping ---
    es = c["training"].get("early_stopping", {})
    if es:
        if "patience" in es and es["patience"] <= 0:
            errors.append("early_stopping.patience must be > 0")
        if "min_delta" in es and es["min_delta"] < 0:
            errors.append("early_stopping.min_delta must be >= 0")

    # --- Curriculum ---
    curr = c.get("curriculum", {})
    if curr.get("enabled", False):
        if curr.get("start_density", 0) <= 0:
            errors.append("curriculum.start_density must be > 0")
        if curr.get("end_density", 0) < curr.get("start_density", 0):
            errors.append("curriculum.end_density must be >= start_density")
        if curr.get("episodes_per_level", 0) <= 0:
            errors.append("curriculum.episodes_per_level must be > 0")

    if errors:
        raise ConfigError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    # --- Warnings (non-fatal) ---
    if c["training"]["total_episodes"] < 50:
        warnings.append(
            "training.total_episodes < 50 — model may not converge"
        )
    if c["training"]["batch_size"] < 32:
        warnings.append(
            "training.batch_size < 32 — gradient estimates may be noisy"
        )
    if dqn and dqn.get("replay_buffer_size", 0) < 10000:
        warnings.append(
            "dqn.replay_buffer_size < 10000 — may limit experience diversity"
        )

    for w in warnings:
        logger.warning("Config warning: %s", w)

    logger.info("Configuration validated successfully (%d warnings)", len(warnings))
    return warnings
