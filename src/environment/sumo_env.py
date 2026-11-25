"""Multi-agent SUMO environment with explicit seconds and lawful cyclic control.

Legacy lane features/reward options are retained for compatible networks.
Evaluation uses network accounting and TripInfo, never these reward proxies.
"""
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import traci
from traci import constants as tc
import gymnasium as gym
from gymnasium import spaces

from src.environment.timing import EpisodeTiming
from src.environment.episode_metrics import planned_demand, tripinfo_summary


class MultiAgentSumoEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        s = config['sumo']
        self.timing = EpisodeTiming.from_config(config)
        self.sumo_cfg = s['cfg_file']
        self.use_gui = s.get('use_gui', False)
        self.delta_time = self.timing.decision_s
        self.yellow_time = s['yellow_time']
        self.min_green, self.max_green = s['min_green'], s['max_green']
        self.control_mode = s.get('control_mode', 'rl')
        if self.control_mode not in ('rl', 'fixed', 'actuated'):
            raise ValueError('Unknown control mode')
        if s.get('density_multiplier', 1) != 1:
            raise ValueError('Generate complete demand files; runtime density deletion/cloning is disabled')
        signal_times = (self.min_green, self.max_green, self.yellow_time, s.get('all_red_time', 1))
        if any(not isinstance(x, int) or x < 1 for x in signal_times) or self.max_green < self.min_green:
            raise ValueError('Invalid signal timing')
        self.reward_type = config['environment']['reward_type']
        rw = config['environment'].get('reward_weights', {})
        for key, default in [('queue_weight', .4), ('waiting_time_weight', .4), ('emergency_weight', .2),
                             ('throughput_bonus', 0), ('switch_penalty', 0), ('congestion_penalty', 0),
                             ('congestion_threshold', .8)]:
            setattr(self, key, rw.get('incoming_count_decrease_bonus', rw.get(key, default)) if key == 'throughput_bonus' else rw.get(key, default))
        self.sumo_running = False
        self.ts_ids, self.agents = [], {}
        self.observation_space = self.action_space = None
        self.max_lanes = self.max_phases = self.current_step = 0
        self._temp = tempfile.TemporaryDirectory(prefix='rl-traffic-')
        self.tripinfo_path = str(Path(self._temp.name) / 'tripinfo.xml')
        self.seed = config['experiment']['seed']

    def start_sumo(self):
        cmd = ['sumo-gui' if self.use_gui else 'sumo', '-c', str(self.sumo_cfg),
               '--seed', str(self.seed), '--begin', str(self.timing.begin_s), '--step-length', '1',
               '--waiting-time-memory', '10000', '--time-to-teleport', '-1',
               '--no-step-log', 'true', '--duration-log.disable', 'true',
               '--tripinfo-output', self.tripinfo_path, '--tripinfo-output.write-unfinished', 'true',
               '--tripinfo-output.write-undeparted', 'true']
        traci.start(cmd)
        self.sumo_running = True

    def _get_traffic_lights(self):
        self.ts_ids = sorted(traci.trafficlight.getIDList())
        if not self.ts_ids:
            raise ValueError('No traffic lights in network')
        self.agents = {tid: TrafficSignalAgent(tid, self.delta_time, self.yellow_time,
                       self.min_green, self.max_green) for tid in self.ts_ids}
        for agent in self.agents.values():
            agent.configure_program(self.control_mode, self.config['sumo'].get('all_red_time', 1))
        self.max_lanes = max(len(a.incoming_lanes) for a in self.agents.values())
        self.max_phases = max(len(a.green_phases) for a in self.agents.values())
        for agent in self.agents.values():
            agent.max_lanes, agent.max_phases = self.max_lanes, self.max_phases
        self.num_agents = len(self.ts_ids)
        self.agent_id_map = {tid: i for i, tid in enumerate(self.ts_ids)}
        self.controlled_lanes = sorted({lane for a in self.agents.values() for lane in a.incoming_lanes})
        lane_variables = [tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.LAST_STEP_MEAN_SPEED,
                          tc.VAR_WAITING_TIME, tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_ID_LIST]
        for lane in self.controlled_lanes:
            traci.lane.subscribe(lane, lane_variables)
        for tid in self.ts_ids:
            traci.trafficlight.subscribe(tid, [tc.TL_RED_YELLOW_GREEN_STATE])
        self.observation_schema = {
            'version': 2, 'signals': self.ts_ids,
            'lanes_by_signal': {tid: a.incoming_lanes for tid, a in self.agents.items()},
            'green_states_by_signal': {tid: [a.phases[i].state for i in a.green_phases] for tid, a in self.agents.items()},
            'lane_features': ['queue/20', 'speed/15', 'lane_wait_snapshot/100', 'density/0.5', 'count/30'],
            'normalization': {'queue': 20, 'speed': 15, 'lane_wait_snapshot': 100,
                              'density': .5, 'count': 30, 'green_elapsed': self.max_green},
            'padding': 'zeros to max_lanes; green phase onehot to max_phases',
            'elapsed': 'green_elapsed_seconds/max_green, clipped at 1',
            'max_lanes': self.max_lanes, 'max_phases': self.max_phases,
            'action_schema': '0 hold, 1 request next cyclic green; safety constraints may override',
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = self.config['experiment']['seed'] if seed is None else int(seed)
        self.close()
        self.arrived_total = self.departed_total = self.arrived_window = 0
        self.stopped_vehicle_seconds = self.controlled_queue_seconds = 0
        self.elapsed_seconds = self.current_step = 0
        self.timeline, self.action_log = [], []
        self._finished = False
        self.demand_ids = planned_demand(self.sumo_cfg, self.timing.end_s, self.timing.begin_s)
        try:
            self.start_sumo()
            self._get_traffic_lights()
            for _ in range(self.timing.warmup_s):
                self._tick(measure=False)
            obs = self._get_observations()
            self.observation_space = spaces.Box(0, 1, shape=obs[self.ts_ids[0]].shape, dtype=np.float32)
            self.action_space = spaces.Discrete(2)
            return obs, {tid: {} for tid in self.ts_ids}
        except BaseException:
            self.close()
            raise

    def _tick(self, measure):
        if self.control_mode == 'rl':
            for a in self.agents.values():
                a.before_tick()
        traci.simulationStep()
        self.arrived_total += traci.simulation.getArrivedNumber()
        self.departed_total += traci.simulation.getDepartedNumber()
        for vid in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(vid, [tc.VAR_SPEED, tc.VAR_VEHICLECLASS, tc.VAR_WAITING_TIME])
        if self.control_mode == 'rl':
            for a in self.agents.values():
                a.after_tick()
        if measure:
            self.arrived_window += traci.simulation.getArrivedNumber()
            stopped = sum(x.get(tc.VAR_SPEED, 1) <= .1
                          for x in traci.vehicle.getAllSubscriptionResults().values())
            lanes = traci.lane.getAllSubscriptionResults()
            halting = tc.LAST_STEP_VEHICLE_HALTING_NUMBER
            queue = sum(lanes[lane][halting] for lane in self.controlled_lanes)
            self.stopped_vehicle_seconds += stopped
            self.controlled_queue_seconds += queue
            self.elapsed_seconds += 1
            if self.config['sumo'].get('record_timeline', False):
                signals = traci.trafficlight.getAllSubscriptionResults()
                self.timeline.append({
                    'simulated_second': traci.simulation.getTime(),
                    'elapsed_s': self.elapsed_seconds, 'network_stopped': stopped,
                    'controlled_lane_queue': queue, 'arrived_cumulative': self.arrived_total,
                    'signals': {tid: {
                        'state': signals[tid][tc.TL_RED_YELLOW_GREEN_STATE],
                        'queue': sum(lanes[lane][halting] for lane in a.incoming_lanes)}
                        for tid, a in self.agents.items()}})

    def step(self, actions):
        if self._finished or not self.sumo_running:
            raise RuntimeError('reset required before step')
        previous = self._get_metrics()
        start_arrived = self.arrived_total
        for tid, a in self.agents.items():
            a.applied_action = 0
            if self.control_mode == 'rl':
                a.set_action(actions.get(tid, 0))
        remaining = min(self.delta_time, self.timing.horizon_s - self.elapsed_seconds)
        for _ in range(remaining):
            self._tick(measure=True)
            if self.config['sumo'].get('terminate_when_empty', False) and traci.simulation.getMinExpectedNumber() == 0:
                break
        self.current_step += 1
        current = self._get_metrics()
        applied = {tid: a.applied_action for tid, a in self.agents.items()}
        rewards = self._calculate_rewards(previous, current, applied)
        natural = traci.simulation.getMinExpectedNumber() == 0
        at_limit = self.elapsed_seconds >= self.timing.horizon_s
        terminal = natural and (at_limit or self.config['sumo'].get('terminate_when_empty', False))
        truncated = at_limit and not terminal
        self._finished = terminal or truncated
        dones = {tid: terminal for tid in self.ts_ids + ['__all__']}
        truncs = {tid: truncated for tid in self.ts_ids + ['__all__']}
        infos = {tid: {
            **current[tid], 'requested_action': actions.get(tid, 0),
            'applied_action': applied[tid], 'simulated_second': traci.simulation.getTime()}
            for tid in self.ts_ids}
        infos['__network__'] = {'arrivals_this_decision': self.arrived_total - start_arrived,
                                'arrivals_total': self.arrived_total, 'elapsed_s': self.elapsed_seconds}
        self.action_log.append({'elapsed_s': self.elapsed_seconds, 'requested': dict(actions), 'applied': applied})
        return self._get_observations(), rewards, dones, truncs, infos

    def _get_observations(self):
        return {tid: np.asarray(a.get_observation(), dtype=np.float32) for tid, a in self.agents.items()}

    def _get_metrics(self):
        return {tid: {'waiting_time': a.get_waiting_time(), 'queue': a.get_queue_length(),
                      'pressure': a.get_pressure(), 'emergency_waiting': a.get_emergency_waiting_time(),
                      'vehicle_count': a.get_vehicle_count(), 'density': a.get_density()}
                for tid, a in self.agents.items()}

    def episode_summary(self):
        if self.sumo_running:
            raise RuntimeError('Close SUMO to flush TripInfo before reading summary')
        return {**tripinfo_summary(self.tripinfo_path, self.demand_ids),
                'seed': self.seed, 'simulated_seconds': self.elapsed_seconds,
                'warmup_seconds': self.timing.warmup_s,
                'window_begin_s': self.timing.window_begin_s,
                'window_end_s': self.timing.window_begin_s + self.elapsed_seconds,
                'arrivals_network_total': self.arrived_total,
                'arrivals_network_window': self.arrived_window,
                'departures_network_total': self.departed_total,
                'network_stopped_vehicle_seconds': self.stopped_vehicle_seconds,
                'controlled_lane_queue_mean': self.controlled_queue_seconds / max(1, self.elapsed_seconds)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.sumo_running:
            try:
                traci.close()
            finally:
                self.sumo_running = False

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

                # Legacy throughput_bonus is an incoming-count decrease proxy, not throughput.
                if self.throughput_bonus > 0:
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
        self.lanes: List[str] = sorted(set(traci.trafficlight.getControlledLanes(self.id)))

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

        if not self.green_phases:
            raise ValueError(f"No legal green phases: {self.id}")
        self.phase_index = self.green_phases[0]
        self.remaining = 0
        self.applied_action = 0

    def configure_program(self, mode, all_red_time):
        """Preserve network phase order, yellow >= configured minimum, insert all-red.

        Branching programs are rejected instead of guessing a legal transition.
        All policies get the same clearance plan; fixed keeps original green
        durations clipped to min/max, actuated uses SUMO's gap controller.
        """
        phases = []
        for i, phase in enumerate(self.phases):
            if phase.next and tuple(phase.next) != ((i + 1) % len(self.phases),):
                raise ValueError(f"Branching signal program unsupported: {self.id}")
        for index in self.green_phases:
            phase = self.phases[index]
            duration = max(self.min_green, min(self.max_green, phase.duration))
            phases.append(traci.trafficlight.Phase(duration, phase.state,
                          minDur=self.min_green, maxDur=self.max_green))
            # Every released green gets yellow before all-red, including
            # permissive greens that legacy programs carried through clearance.
            yellow = ''.join('y' if c in 'Gg' else 'r' for c in phase.state)
            following = self.phases[(index + 1) % len(self.phases)]
            yellow_s = max(self.yellow_time, following.duration if 'y' in following.state.lower() else 0)
            phases.append(traci.trafficlight.Phase(yellow_s, yellow))
            if all_red_time:
                phases.append(traci.trafficlight.Phase(all_red_time, 'r' * len(phase.state)))
        self.phases = phases
        self.green_phases = [i for i, p in enumerate(phases) if 'y' not in p.state.lower() and any(c in 'gG' for c in p.state)]
        for i in self.green_phases:
            following = phases[(i + 1) % len(phases)]
            # Green-to-green sequences can add permissions without conflict,
            # but are deliberately unsupported by this binary cyclic controller.
            if 'y' not in following.state.lower():
                raise ValueError(f"Missing yellow after green: {self.id} phase {i}")
        logic_type = 3 if mode == 'actuated' else 0
        logic = traci.trafficlight.Logic('benchmark', logic_type, self.green_phases[0], phases)
        traci.trafficlight.setProgramLogic(self.id, logic)
        traci.trafficlight.setProgram(self.id, 'benchmark')
        self.phase_index = self.green_phases[0]
        traci.trafficlight.setPhase(self.id, self.phase_index)
        if mode == 'rl':
            traci.trafficlight.setPhaseDuration(self.id, 1e6)

    def set_action(self, action):
        if action not in (0, 1):
            raise ValueError('Expected binary hold/switch request')
        if action and not self.is_yellow and self.time_since_last_phase_change >= self.min_green:
            self._advance_phase()

    def _advance_phase(self):
        leaving_green = not self.is_yellow
        self.phase_index = (self.phase_index + 1) % len(self.phases)
        self.is_yellow = self.phase_index not in self.green_phases
        self.remaining = int(np.ceil(self.phases[self.phase_index].duration)) if self.is_yellow else 0
        if leaving_green:
            self.applied_action = 1
        if not self.is_yellow:
            self.time_since_last_phase_change = 0
        traci.trafficlight.setPhase(self.id, self.phase_index)
        traci.trafficlight.setPhaseDuration(self.id, 1e6)

    def before_tick(self):
        if self.is_yellow and self.remaining == 0:
            self._advance_phase()
        elif not self.is_yellow and self.time_since_last_phase_change >= self.max_green:
            self._advance_phase()

    def after_tick(self):
        if self.is_yellow:
            self.remaining -= 1
        else:
            self.time_since_last_phase_change += 1

    # ------------------------------------------------------------------
    # Observation  (enhanced feature engineering)
    # ------------------------------------------------------------------

    def get_observation(self, agent_id_onehot: Optional[List[float]] = None) -> List[float]:
        """Build a fixed-size observation vector for this traffic light.

        The observation includes per-lane features (padded to ``max_lanes``):
            1. Queue length (normalised)
            2. Average speed (normalised)
            3. Waiting time (normalised)
            4. Vehicle density â€” vehicles / lane length (normalised)
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

                snapshot = traci.lane.getSubscriptionResults(lane)
                queue = snapshot[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                obs.append(min(queue / max_queue, 1.0))

                speed = snapshot[tc.LAST_STEP_MEAN_SPEED]
                obs.append(min(max(speed / max_speed, 0.0), 1.0))

                waiting = snapshot[tc.VAR_WAITING_TIME]
                obs.append(min(waiting / max_waiting, 1.0))

                # Vehicle density (vehicles / lane_length)
                veh_count = snapshot[tc.LAST_STEP_VEHICLE_NUMBER]
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

    def _lane_value(self, lane, variable):
        return traci.lane.getSubscriptionResults(lane)[variable]

    def get_queue_length(self) -> int:
        """Total number of halting vehicles across incoming lanes.

        Returns:
            Integer queue length.
        """
        return sum(
            self._lane_value(lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER) for lane in self.incoming_lanes
        )

    def get_waiting_time(self) -> float:
        """Controlled incoming-lane waiting snapshot sum (vehicle-seconds); not trip waiting.

        Returns:
            Aggregate waiting time.
        """
        return sum(self._lane_value(lane, tc.VAR_WAITING_TIME) for lane in self.incoming_lanes)

    def get_pressure(self) -> float:
        """Traffic pressure (incoming vehicle count) for this intersection.

        Returns:
            Float pressure value.
        """
        return float(self.get_vehicle_count())

    def get_vehicle_count(self) -> int:
        """Total vehicle count across incoming lanes.

        Returns:
            Integer vehicle count.
        """
        return sum(
            self._lane_value(lane, tc.LAST_STEP_VEHICLE_NUMBER) for lane in self.incoming_lanes
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
            count = self._lane_value(lane, tc.LAST_STEP_VEHICLE_NUMBER)
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
        vehicles = traci.vehicle.getAllSubscriptionResults()
        for lane in self.incoming_lanes:
            for vid in self._lane_value(lane, tc.LAST_STEP_VEHICLE_ID_LIST):
                vehicle = vehicles.get(vid, {})
                if vehicle.get(tc.VAR_VEHICLECLASS) == "emergency":
                    emergency_wait += vehicle[tc.VAR_WAITING_TIME]
        return emergency_wait
