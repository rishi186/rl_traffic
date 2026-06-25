"""SUMO environment for multi-agent traffic signal control."""

__all__ = ["MultiAgentSumoEnv"]


def __getattr__(name: str):
    if name == "MultiAgentSumoEnv":
        from src.environment.sumo_env import MultiAgentSumoEnv
        return MultiAgentSumoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")