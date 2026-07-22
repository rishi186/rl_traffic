"""Agent implementations for traffic signal control.

Imports are deferred to avoid hard dependency on torch/tensorflow at import time.
Use ``from src.agents.dqn_agent import DQNAgent`` directly.
"""

from src.agents.replay_buffer import ReplayBuffer

__all__ = ["ReplayBuffer"]


def __getattr__(name: str):
    if name == "DQNAgent":
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent
    if name == "PPOAgent":
        from src.agents.ppo_agent import PPOAgent
        return PPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
