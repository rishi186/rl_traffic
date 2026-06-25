"""Multi-agent SUMO environment for RL-based traffic signal control.

Provides a Gymnasium-compatible interface where each traffic light acts as
an independent agent.  Supports multiple reward types including a custom
shaped reward penalising queue length, cumulative waiting time, and
emergency-vehicle delay.

Features:
    - Per-lane vehicle density & count in the observation vector
    - Emergency vehicle detection via ``traci.vehicle.getVehicleClass``
    - Runtime density scaling through a ``density_multiplier`` config knob
"""

import os
import sys
import random
import numpy as np
import traci
from typing import Dict, Tuple, List, Optional, Any
import gymnasium as gym
from gymnasium import spaces

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Environment
# ======================================================================

class MultiAgentSumoEnv(gym.Env):
    """Multi-agent SUMO environment for traffic signal control.

    Each traffic light is an independent agent that chooses to **keep** or
    **switch** its current green phase every ``delta_time`` seconds.

    Args:
        config: Full experiment configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.sumo_cfg: str = config["sumo"]["cfg_file"]
        self.use_gui: bool = config["sumo"]["use_gui"]
        self.delta_time: int = config["sumo"]["delta_time"]
        self.yellow_time: int = config["sumo"]["yellow_time"]
        self.min_green: int = config["sumo"]["min_green"]
        self.max_green: int = config["sumo"]["max_green"]
        self.max_steps: int = config["training"]["max_steps_per_episode"]
        self.reward_type: str = config["environment"]["reward_type"]
        self.density_multiplier: float = config["sumo"].get("density_multiplier", 1.0)

        # Reward weights for custom-shaped reward
        rw = config["environment"].get("reward_weights", {})
        self.queue_weight: float = rw.get("queue_weight", 0.4)
        self.waiting_time_weight: float = rw.get("waiting_time_weight", 0.4)
        self.emergency_weight: float = rw.get("emergency_weight", 0.2)
        self.throughput_bonus: float = rw.get("throughput_bonus", 0.0)
        self.switch_penalty: float = rw.get("switch_penalty", 0.0)
        self.congestion_penalty: float = rw.get("congestion_penalty", 0.0)
        self.congestion_threshold: float = rw.get("congestion_threshold", 0.8)

        # Multi-agent parameter sharing
        self.num_agents: int = 0
        self.agent_id_map: Dict[str, int] = {}

        self.sumo_running: bool = False
        self.ts_ids: List[str] = []
        self.agents: Dict[str, "TrafficSignalAgent"] = {}
        self.current_step: int = 0
        self._prev_actions: Dict[str, int] = {}
        self._prev_throughput: int = 0

        self.observation_space: Optional[spaces.Box] = None
        self.action_space: Optional[spaces.Discrete] = None

        self.max_lanes: int = 0
        self.max_phases: int = 0

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------

    def start_sumo(self) -> None:
        """Start the SUMO simulation process via TraCI.

        Raises:
            RuntimeError: If SUMO fails to start after retries.
        """
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                traci.start(sumo_cmd)
                self.sumo_running = True
                logger.info("SUMO started (attempt %d) | cfg=%s", attempt, self.sumo_cfg)
                return
            except Exception as exc:
                logger.warning("SUMO start attempt %d failed: %s", attempt, exc)
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to start SUMO after {max_retries} attempts"
                    ) from exc

    def _get_traffic_lights(self) -> None:
        """Discover traffic lights and build per-signal agent wrappers."""
        self.ts_ids = list(traci.trafficlight.getIDList())
        logger.info("Found %d traffic lights", len(self.ts_ids))

        temp_agents: List["TrafficSignalAgent"] = []
        for ts_id in self.ts_ids:
            agent = TrafficSignalAgent(
                ts_id,
                self.delta_time,
                self.yellow_time,
                self.min_green,
                self.max_green,
            )
            temp_agents.append(agent)
            self.max_lanes = max(self.max_lanes, len(agent.incoming_lanes))
            self.max_phases = max(self.max_phases, len(agent.green_phases))

        logger.info(
            "Max lanes=%d | Max green phases=%d", self.max_lanes, self.max_phases
        )

        for agent in temp_agents:
            agent.max_lanes = self.max_lanes
            agent.max_phases = self.max_phases
            self.agents[agent.id] = agent

        # Build agent ID map for parameter sharing
        self.num_agents = len(self.ts_ids)
        self.agent_id_map = {ts_id: i for i, ts_id in enumerate(self.ts_ids)}

    def _apply_density_scaling(self) -> None:
        """Scale traffic density at runtime via TraCI.

        For multipliers < 1.0, randomly removes a fraction of queued
        vehicles.  For multipliers > 1.0, duplicates a fraction of
        departing vehicles with slight offset.
        """
        if abs(self.density_multiplier - 1.0) < 1e-3:
            return

        vehicle_ids = list(traci.vehicle.getIDList())
        if not vehicle_ids:
            return

        if self.density_multiplier < 1.0:
            # Remove vehicles to reduce density
            remove_frac = 1.0 - self.density_multiplier
            to_remove = random.sample(
                vehicle_ids, k=int(len(vehicle_ids) * remove_frac)
            )
            for vid in to_remove:
                try:
                    traci.vehicle.remove(vid)
                except traci.exceptions.TraCIException:
                    pass
            logger.debug(
                "Density scaling: removed %d / %d vehicles (multiplier=%.2f)",
                len(to_remove), len(vehicle_ids), self.density_multiplier,
            )
        else:
            # Clone vehicles to increase density
            clone_frac = self.density_multiplier - 1.0
            to_clone = random.sample(
                vehicle_ids,
                k=min(int(len(vehicle_ids) * clone_frac), len(vehicle_ids)),
            )
            cloned = 0
            for vid in to_clone:
                try:
                    route = traci.vehicle.getRouteID(vid)
                    new_id = f"{vid}_clone_{cloned}"
                    traci.vehicle.add(new_id, route)
                    cloned += 1
                except traci.exceptions.TraCIException:
                    pass
            logger.debug(
                "Density scaling: cloned %d vehicles (multiplier=%.2f)",
                cloned, self.density_multiplier,
            )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        """Reset the environment and return initial observations.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (observations dict, infos dict).
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if self.sumo_running:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self.sumo_running = False

        self.start_sumo()
        self._get_traffic_lights()
        self.current_step = 0

        # Run a few steps so vehicles populate the network for density scaling
        for _ in range(5):
            traci.simulationStep()
        self._apply_density_scaling()

        if self.observation_space is None:
            sample_obs = self._get_observations()
            obs_dim = len(sample_obs[self.ts_ids[0]])
            logger.info("Observation dimension (fixed): %d", obs_dim)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(2)

        self._prev_actions = {ts_id: 0 for ts_id in self.ts_ids}
        self._prev_throughput = traci.simulation.getArrivedNumber()

        observations = self._get_observations()
        infos: Dict[str, dict] = {ts_id: {} for ts_id in self.ts_ids}
        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """Execute one environment step.

        Args:
            actions: Mapping from traffic-light ID to action (0=keep, 1=switch).

        Returns:
            (observations, rewards, dones, truncateds, infos)
        """
        previous_metrics = self._get_metrics()

        for ts_id, action in actions.items():
            self.agents[ts_id].set_action(action)

        for _ in range(self.delta_time):
            traci.simulationStep()

        self.current_step += 1

        observations = self._get_observations()
        current_metrics = self._get_metrics()
        rewards = self._calculate_rewards(previous_metrics, current_metrics, actions)

        sim_done = traci.simulation.getMinExpectedNumber() <= 0
        done = (self.current_step >= self.max_steps) or sim_done
        dones: Dict[str, bool] = {ts_id: done for ts_id in self.ts_ids}
        dones["__all__"] = done

        truncateds: Dict[str, bool] = {ts_id: False for ts_id in self.ts_ids}
        truncateds["__all__"] = False

        # Compute throughput delta for info
        current_throughput = traci.simulation.getArrivedNumber()
        throughput_delta = current_throughput - self._prev_throughput
        self._prev_throughput = current_throughput

        infos: Dict[str, dict] = {
            ts_id: {
                "waiting_time": current_metrics[ts_id]["waiting_time"],
                "queue": current_metrics[ts_id]["queue"],
                "emergency_waiting": current_metrics[ts_id]["emergency_waiting"],
                "vehicle_count": current_metrics[ts_id]["vehicle_count"],
                "density": current_metrics[ts_id]["density"],
                "pressure": current_metrics[ts_id]["pressure"],
                "throughput_delta": throughput_delta,
            }
            for ts_id in self.ts_ids
        }

        self._prev_actions = dict(actions)

        return observations, rewards, dones, truncateds, infos

    # ------------------------------------------------------------------
    # Observations & metrics
    # ------------------------------------------------------------------

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Collect observation vectors for every traffic-light agent.

        Returns:
            Dict mapping traffic-light ID to its observation array.
        """
        observations: Dict[str, np.ndarray] = {}
        for ts_id in self.ts_ids:
            obs = self.agents[ts_id].get_observation()
            observations[ts_id] = np.array(obs, dtype=np.float32)
        return observations

    def _get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect current metrics for all traffic lights.

        Returns:
            Nested dict of per-signal metrics.
        """
        metrics: Dict[str, Dict[str, float]] = {}
        for ts_id in self.ts_ids:
            agent = self.agents[ts_id]
            metrics[ts_id] = {
                "waiting_time": agent.get_waiting_time(),
                "queue": agent.get_queue_length(),
                "pressure": agent.get_pressure(),
                "emergency_waiting": agent.get_emergency_waiting_time(),
                "vehicle_count": agent.get_vehicle_count(),
                "density": agent.get_density(),
            }
        return metrics

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _calculate_rewards(
        self,
        previous: Dict[str, Dict[str, float]],
        current: Dict[str, Dict[str, float]],
        actions: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute per-agent rewards.

        Supports reward types:
            - ``diff-waiting-time``: decrease in total waiting time
            - ``queue``: negative current queue length
            - ``pressure``: negative pressure
            - ``custom-shaped``: weighted combination of queue, waiting time,
              emergency-vehicle delay, throughput bonus, switch penalty,
              and congestion penalty
            - default: negative waiting time

        Args:
            previous: Metrics *before* the step.
            current: Metrics *after* the step.
            actions: Actions taken this step (0=keep, 1=switch).

        Returns:
            Dict mapping traffic-light ID to scalar reward.
        """
        rewards: Dict[str, float] = {}

        for ts_id in self.ts_ids:
            if self.reward_type == "diff-waiting-time":
                reward = previous[ts_id]["waiting_time"] - current[ts_id]["waiting_time"]

            elif self.reward_type == "queue":
                reward = -current[ts_id]["queue"]

            elif self.reward_type == "pressure":
                reward = -current[ts_id]["pressure"]

            elif self.reward_type == "custom-shaped":
                queue_penalty = -self.queue_weight * current[ts_id]["queue"]
                wait_penalty = -self.waiting_time_weight * current[ts_id]["waiting_time"]
                emerg_penalty = -self.emergency_weight * current[ts_id]["emergency_waiting"]
                reward = queue_penalty + wait_penalty + emerg_penalty

                # Throughput bonus: reward vehicles that cleared the intersection
                if self.throughput_bonus > 0:
                    throughput_delta = max(0, current[ts_id]["vehicle_count"] - previous[ts_id]["vehicle_count"])
                    # Actually, throughput is about vehicles leaving, not arriving
                    # Use the difference in pressure as a proxy
                    pressure_delta = previous[ts_id]["pressure"] - current[ts_id]["pressure"]
                    reward += self.throughput_bonus * max(0, pressure_delta)

                # Switch penalty: discourage excessive phase switching
                if self.switch_penalty > 0 and actions.get(ts_id, 0) == 1:
                    reward -= self.switch_penalty

                # Congestion penalty: extra penalty when density is very high
                if self.congestion_penalty > 0:
                    density = current[ts_id]["density"]
                    if density > self.congestion_threshold:
                        reward -= self.congestion_penalty * (density - self.congestion_threshold)

            else:
                reward = -current[ts_id]["waiting_time"]

            rewards[ts_id] = reward

        return rewards

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MultiAgentSumoEnv":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the SUMO simulation cleanly."""
        if self.sumo_running:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self.sumo_running = False
            logger.info("SUMO simulation closed")


# ======================================================================
# Per-signal agent wrapper
# ======================================================================

class TrafficSignalAgent:
    """Wrapper around a single SUMO traffic light for observation/action logic.

    Args:
        ts_id: SUMO traffic-light ID.
        delta_time: Simulation seconds per RL step.
        yellow_time: Duration of yellow phase in seconds.
        min_green: Minimum green phase duration before switching is allowed.
        max_green: Maximum green phase duration (used for normalisation).
    """

    def __init__(
        self,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
    ) -> None:
        self.id = ts_id
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green

        self.green_phase: int = 0
        self.is_yellow: bool = False
        self.time_since_last_phase_change: int = 0
        self.next_action_time: int = 0

        self.max_lanes: int = 0
        self.max_phases: int = 0

        # Discover controlled lanes
        self.lanes: List[str] = list(set(traci.trafficlight.getControlledLanes(self.id)))

        self.incoming_lanes: List[str] = []
        for lane in self.lanes:
            edge = traci.lane.getEdgeID(lane)
            if not edge.startswith(":"):
                self.incoming_lanes.append(lane)

        # Cache lane lengths for density computation
        self._lane_lengths: Dict[str, float] = {}
        for lane in self.incoming_lanes:
            try:
                self._lane_lengths[lane] = max(traci.lane.getLength(lane), 1.0)
            except traci.exceptions.TraCIException:
                self._lane_lengths[lane] = 100.0

        # Discover green phases
        logic = traci.trafficlight.getAllProgramLogics(self.id)[0]
        self.phases = logic.phases
        self.green_phases: List[int] = []
        for i, phase in enumerate(self.phases):
            if "y" not in phase.state.lower() and ("G" in phase.state or "g" in phase.state):
                self.green_phases.append(i)

        if len(self.green_phases) == 0:
            self.green_phases = list(range(len(self.phases)))

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def set_action(self, action: int) -> None:
        """Apply an action to this traffic light.

        Args:
            action: 0 = keep current phase, 1 = switch to next green phase.
        """
        if action == 1 and self.time_since_last_phase_change >= self.min_green:
            self._set_yellow_phase()
        else:
            self.time_since_last_phase_change += self.delta_time

    def _set_yellow_phase(self) -> None:
        """Transition through a yellow phase before the next green phase."""
        current_phase = traci.trafficlight.getPhase(self.id)

        try:
            current_idx = self.green_phases.index(current_phase)
            next_idx = (current_idx + 1) % len(self.green_phases)
            next_green = self.green_phases[next_idx]
        except ValueError:
            next_green = self.green_phases[0]

        if current_phase + 1 < len(self.phases):
            traci.trafficlight.setPhase(self.id, current_phase + 1)

        self.next_action_time = self.yellow_time
        self.green_phase = next_green
        self.is_yellow = True
        self.time_since_last_phase_change = 0

    # ------------------------------------------------------------------
    # Observation  (enhanced feature engineering)
    # ------------------------------------------------------------------

    def get_observation(self, agent_id_onehot: Optional[List[float]] = None) -> List[float]:
        """Build a fixed-size observation vector for this traffic light.

        The observation includes per-lane features (padded to ``max_lanes``):
            1. Queue length (normalised)
            2. Average speed (normalised)
            3. Waiting time (normalised)
            4. Vehicle density — vehicles / lane length (normalised)
            5. Vehicle count (normalised)

        Plus global features:
            - One-hot encoded current phase (size ``max_phases``)
            - Normalised time since last phase change (size 1)
            - Optional: Agent ID one-hot embedding (for parameter sharing)

        Total dimension: ``max_lanes * 5 + max_phases + 1 + num_agents``

        Args:
            agent_id_onehot: Optional one-hot encoded agent ID vector.

        Returns:
            List of floats representing the observation.
        """
        obs: List[float] = []

        # Normalisation ceilings
        max_queue = 20.0
        max_speed = 15.0
        max_waiting = 100.0
        max_density = 0.5  # vehicles per metre
        max_count = 30.0

        for i in range(self.max_lanes):
            if i < len(self.incoming_lanes):
                lane = self.incoming_lanes[i]

                queue = traci.lane.getLastStepHaltingNumber(lane)
                obs.append(min(queue / max_queue, 1.0))

                speed = traci.lane.getLastStepMeanSpeed(lane)
                obs.append(min(max(speed / max_speed, 0.0), 1.0))

                waiting = traci.lane.getWaitingTime(lane)
                obs.append(min(waiting / max_waiting, 1.0))

                # Vehicle density (vehicles / lane_length)
                veh_count = traci.lane.getLastStepVehicleNumber(lane)
                lane_len = self._lane_lengths.get(lane, 100.0)
                density = veh_count / lane_len
                obs.append(min(density / max_density, 1.0))

                # Raw vehicle count
                obs.append(min(veh_count / max_count, 1.0))
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Phase one-hot
        current_phase = traci.trafficlight.getPhase(self.id)
        for i in range(self.max_phases):
            if i < len(self.green_phases):
                obs.append(1.0 if current_phase == self.green_phases[i] else 0.0)
            else:
                obs.append(0.0)

        # Elapsed time feature
        obs.append(min(self.time_since_last_phase_change / self.max_green, 1.0))

        # Agent ID one-hot embedding (for parameter sharing)
        if agent_id_onehot is not None:
            obs.extend(agent_id_onehot)

        return obs

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def get_queue_length(self) -> int:
        """Total number of halting vehicles across incoming lanes.

        Returns:
            Integer queue length.
        """
        return sum(
            traci.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes
        )

    def get_waiting_time(self) -> float:
        """Total cumulative waiting time (seconds) across incoming lanes.

        Returns:
            Aggregate waiting time.
        """
        return sum(traci.lane.getWaitingTime(lane) for lane in self.incoming_lanes)

    def get_pressure(self) -> float:
        """Traffic pressure (incoming vehicle count) for this intersection.

        Returns:
            Float pressure value.
        """
        incoming = sum(
            traci.lane.getLastStepVehicleNumber(lane) for lane in self.incoming_lanes
        )
        return float(incoming)

    def get_vehicle_count(self) -> int:
        """Total vehicle count across incoming lanes.

        Returns:
            Integer vehicle count.
        """
        return sum(
            traci.lane.getLastStepVehicleNumber(lane) for lane in self.incoming_lanes
        )

    def get_density(self) -> float:
        """Average vehicle density (vehicles/metre) across incoming lanes.

        Returns:
            Mean density value.
        """
        if not self.incoming_lanes:
            return 0.0
        total_density = 0.0
        for lane in self.incoming_lanes:
            count = traci.lane.getLastStepVehicleNumber(lane)
            length = self._lane_lengths.get(lane, 100.0)
            total_density += count / length
        return total_density / len(self.incoming_lanes)

    def get_emergency_waiting_time(self) -> float:
        """Cumulative waiting time of emergency vehicles on incoming lanes.

        Scans each incoming lane for vehicles whose ``vClass`` is
        ``"emergency"`` and sums their individual waiting times.

        Returns:
            Total emergency-vehicle waiting time (seconds).
        """
        emergency_wait = 0.0
        for lane in self.incoming_lanes:
            try:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                for vid in vehicle_ids:
                    try:
                        vclass = traci.vehicle.getVehicleClass(vid)
                        if vclass == "emergency":
                            emergency_wait += traci.vehicle.getWaitingTime(vid)
                    except traci.exceptions.TraCIException:
                        continue
            except traci.exceptions.TraCIException:
                continue
        return emergency_wait