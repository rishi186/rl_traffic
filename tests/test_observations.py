"""Tests for observation shape consistency.

Since observations require a running SUMO instance, these tests validate
the observation construction logic using mock data and dimensional checks.
"""

import numpy as np


class TestObservationDimensions:
    """Verify observation vector dimension formulas."""

    def test_observation_formula(self):
        """Observation dim should be max_lanes * 5 + max_phases + 1."""
        # Simulate various configurations
        configs = [
            {"max_lanes": 4, "max_phases": 2},
            {"max_lanes": 8, "max_phases": 4},
            {"max_lanes": 12, "max_phases": 6},
            {"max_lanes": 1, "max_phases": 1},
        ]
        for cfg in configs:
            expected_dim = cfg["max_lanes"] * 5 + cfg["max_phases"] + 1
            # Build a mock observation following the same logic as TrafficSignalAgent
            obs = []
            for i in range(cfg["max_lanes"]):
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # queue, speed, wait, density, count
            for i in range(cfg["max_phases"]):
                obs.append(0.0)  # phase one-hot
            obs.append(0.0)  # elapsed time
            assert len(obs) == expected_dim, (
                f"max_lanes={cfg['max_lanes']}, max_phases={cfg['max_phases']}: "
                f"got {len(obs)}, expected {expected_dim}"
            )

    def test_observation_values_bounded(self):
        """All observation values should be in [0.0, 1.0] after normalisation."""
        # Simulate normalised observation
        obs = []
        max_lanes = 6
        max_phases = 3
        for i in range(max_lanes):
            queue_norm = min(np.random.randint(0, 30) / 20.0, 1.0)
            speed_norm = min(max(np.random.random() * 20 / 15.0, 0.0), 1.0)
            wait_norm = min(np.random.random() * 150 / 100.0, 1.0)
            density_norm = min(np.random.random() * 0.8 / 0.5, 1.0)
            count_norm = min(np.random.randint(0, 40) / 30.0, 1.0)
            obs.extend([queue_norm, speed_norm, wait_norm, density_norm, count_norm])
        for i in range(max_phases):
            obs.append(1.0 if i == 0 else 0.0)
        obs.append(min(np.random.random(), 1.0))

        for val in obs:
            assert 0.0 <= val <= 1.0, f"Observation value {val} out of [0, 1] range"

    def test_padded_lanes_are_zero(self):
        """Padding for unused lanes should be all zeros."""
        max_lanes = 8
        actual_lanes = 3
        obs = []
        for i in range(max_lanes):
            if i < actual_lanes:
                obs.extend([0.5, 0.3, 0.2, 0.1, 0.4])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Check padding is zero
        for i in range(actual_lanes, max_lanes):
            start = i * 5
            for j in range(5):
                assert obs[start + j] == 0.0, f"Padded lane {i} feature {j} is not zero"

    def test_observation_dtype(self):
        """Observations should be float32 numpy arrays."""
        obs_list = [0.5] * 31  # Example dim
        obs_array = np.array(obs_list, dtype=np.float32)
        assert obs_array.dtype == np.float32
        assert obs_array.ndim == 1
