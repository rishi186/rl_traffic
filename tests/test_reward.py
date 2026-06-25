"""Tests for reward computation logic.

Since reward computation depends on TraCI metrics, we replicate the
_calculate_rewards logic in a self-contained stub to avoid importing
gymnasium/traci at test time.
"""

import pytest
import numpy as np


def _calculate_rewards(env, previous, current):
    """Standalone reward computation matching MultiAgentSumoEnv._calculate_rewards."""
    rewards = {}
    for ts_id in env.ts_ids:
        if env.reward_type == "diff-waiting-time":
            reward = previous[ts_id]["waiting_time"] - current[ts_id]["waiting_time"]
        elif env.reward_type == "queue":
            reward = -current[ts_id]["queue"]
        elif env.reward_type == "pressure":
            reward = -current[ts_id]["pressure"]
        elif env.reward_type == "custom-shaped":
            queue_penalty = -env.queue_weight * current[ts_id]["queue"]
            wait_penalty = -env.waiting_time_weight * current[ts_id]["waiting_time"]
            emerg_penalty = -env.emergency_weight * current[ts_id]["emergency_waiting"]
            reward = queue_penalty + wait_penalty + emerg_penalty
        else:
            reward = -current[ts_id]["waiting_time"]
        rewards[ts_id] = reward
    return rewards


def _make_env_stub(reward_type, queue_w=0.4, wait_w=0.4, emerg_w=0.2):
    """Create a minimal object mimicking MultiAgentSumoEnv for reward testing."""

    class EnvStub:
        def __init__(self):
            self.reward_type = reward_type
            self.queue_weight = queue_w
            self.waiting_time_weight = wait_w
            self.emergency_weight = emerg_w
            self.ts_ids = ["tl_0", "tl_1"]

    return EnvStub()


class TestRewardComputation:
    """Test all reward types."""

    def _metrics(self, wait=10.0, queue=5, pressure=3.0, emerg=0.0, veh_count=8, density=0.1):
        return {
            "waiting_time": wait,
            "queue": queue,
            "pressure": pressure,
            "emergency_waiting": emerg,
            "vehicle_count": veh_count,
            "density": density,
        }

    def test_diff_waiting_time(self):
        env = _make_env_stub("diff-waiting-time")
        prev = {"tl_0": self._metrics(wait=20.0), "tl_1": self._metrics(wait=15.0)}
        curr = {"tl_0": self._metrics(wait=10.0), "tl_1": self._metrics(wait=18.0)}
        rewards = _calculate_rewards(env, prev, curr)
        assert rewards["tl_0"] == pytest.approx(10.0)  # 20 - 10
        assert rewards["tl_1"] == pytest.approx(-3.0)  # 15 - 18

    def test_queue_reward(self):
        env = _make_env_stub("queue")
        prev = {"tl_0": self._metrics(), "tl_1": self._metrics()}
        curr = {"tl_0": self._metrics(queue=7), "tl_1": self._metrics(queue=3)}
        rewards = _calculate_rewards(env, prev, curr)
        assert rewards["tl_0"] == -7
        assert rewards["tl_1"] == -3

    def test_pressure_reward(self):
        env = _make_env_stub("pressure")
        prev = {"tl_0": self._metrics(), "tl_1": self._metrics()}
        curr = {"tl_0": self._metrics(pressure=5.0), "tl_1": self._metrics(pressure=2.0)}
        rewards = _calculate_rewards(env, prev, curr)
        assert rewards["tl_0"] == pytest.approx(-5.0)
        assert rewards["tl_1"] == pytest.approx(-2.0)

    def test_custom_shaped_reward(self):
        env = _make_env_stub("custom-shaped", queue_w=0.5, wait_w=0.3, emerg_w=0.2)
        prev = {"tl_0": self._metrics(), "tl_1": self._metrics()}
        curr = {
            "tl_0": self._metrics(queue=10, wait=20.0, emerg=5.0),
            "tl_1": self._metrics(queue=4, wait=8.0, emerg=0.0),
        }
        rewards = _calculate_rewards(env, prev, curr)
        # tl_0: -0.5*10 + -0.3*20 + -0.2*5 = -5 -6 -1 = -12
        assert rewards["tl_0"] == pytest.approx(-12.0)
        # tl_1: -0.5*4 + -0.3*8 + -0.2*0 = -2 -2.4 -0 = -4.4
        assert rewards["tl_1"] == pytest.approx(-4.4)

    def test_custom_shaped_no_emergency(self):
        env = _make_env_stub("custom-shaped", queue_w=0.4, wait_w=0.4, emerg_w=0.2)
        prev = {"tl_0": self._metrics(), "tl_1": self._metrics()}
        curr = {
            "tl_0": self._metrics(queue=5, wait=10.0, emerg=0.0),
            "tl_1": self._metrics(queue=5, wait=10.0, emerg=0.0),
        }
        rewards = _calculate_rewards(env, prev, curr)
        expected = -0.4 * 5 + -0.4 * 10.0 + -0.2 * 0.0
        assert rewards["tl_0"] == pytest.approx(expected)

    def test_default_reward_type(self):
        env = _make_env_stub("unknown-type")
        prev = {"tl_0": self._metrics(), "tl_1": self._metrics()}
        curr = {"tl_0": self._metrics(wait=12.0), "tl_1": self._metrics(wait=7.0)}
        rewards = _calculate_rewards(env, prev, curr)
        assert rewards["tl_0"] == pytest.approx(-12.0)
        assert rewards["tl_1"] == pytest.approx(-7.0)
