import copy
import shutil
from pathlib import Path

import pytest
import yaml
import traci

from src.environment.sumo_env import MultiAgentSumoEnv
from src.environment.timing import EpisodeTiming


def tiny_config(horizon=60, mode='rl'):
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((root / 'config.yaml').read_text())
    c['sumo'].update(cfg_file=str(root / 'tests/fixtures/tiny/tiny.sumocfg'),
                     num_seconds=horizon, warmup_seconds=0, record_timeline=True, control_mode=mode)
    return c


def test_timing_uses_seconds_not_training_steps():
    c = tiny_config(13)
    c['training']['max_steps_per_episode'] = 1000
    assert EpisodeTiming.from_config(c).end_s == 13


@pytest.mark.integration
@pytest.mark.parametrize('mode', ['fixed', 'actuated', 'rl'])
def test_same_window_arrivals_and_tripinfo(mode):
    if not shutil.which('sumo'):
        pytest.skip('SUMO missing')
    c = tiny_config(103, mode)
    with MultiAgentSumoEnv(c) as env:
        obs, _ = env.reset(seed=9)
        arrivals = 0
        for _ in range(21):
            obs, _, done, trunc, infos = env.step({t: 1 for t in env.ts_ids})
            arrivals += infos['__network__']['arrivals_this_decision']
        assert done['__all__'] or trunc['__all__']
        assert env.elapsed_seconds == 103
        assert arrivals == env.arrived_total
        assert list(obs) == sorted(obs)
        assert env.agents['C'].incoming_lanes == sorted(env.agents['C'].incoming_lanes)
        env.close()
        summary = env.episode_summary()
        assert summary['scheduled_demand'] == 4
        assert summary['completed'] + summary['unfinished'] + summary['uninserted'] == 4
        assert summary['completed'] == env.arrived_total == 4
        assert summary['trip_duration_completed_mean_s'] > 0
        assert not traci.isLoaded()


@pytest.mark.integration
def test_hold_min_max_yellow_red_and_determinism():
    if not shutil.which('sumo'):
        pytest.skip('SUMO missing')
    c = tiny_config(63)
    c['sumo'].update(min_green=5, max_green=10, delta_time=1, all_red_time=2)
    runs = []
    for _ in range(2):
        with MultiAgentSumoEnv(copy.deepcopy(c)) as env:
            env.reset(seed=8)
            for t in range(63):
                env.step({'C': 1 if t < 2 else 0})
            assert env.action_log[0]['applied']['C'] == 0  # minimum green enforced
            assert any(x['applied']['C'] for x in env.action_log)  # max forces switch despite holds
            states = [x['signals']['C']['state'] for x in env.timeline]
            assert len(set(states[:10])) == 1  # no automatic SUMO green expiry
            assert 'y' in states[10].lower()
            yellow = next(i for i, s in enumerate(states) if 'y' in s.lower())
            red = next(i for i in range(yellow, len(states)) if set(states[i]) == {'r'})
            assert red - yellow >= 3
            assert states[red] == states[red + 1]
            runs.append(copy.deepcopy(env.timeline))
    assert runs[0] == runs[1]


def test_runtime_density_rejected():
    c = tiny_config()
    c['sumo']['density_multiplier'] = .5
    with pytest.raises(ValueError, match='complete demand'):
        MultiAgentSumoEnv(c)


def test_fractional_or_zero_clearance_rejected():
    for all_red in (0, .5):
        c = tiny_config()
        c['sumo']['all_red_time'] = all_red
        with pytest.raises(ValueError, match='signal timing'):
            MultiAgentSumoEnv(c)


@pytest.mark.integration
def test_clearance_advance_is_not_an_additional_applied_switch():
    if not shutil.which('sumo'):
        pytest.skip('SUMO missing')
    c = tiny_config(12)
    c['sumo']['delta_time'] = 1
    with MultiAgentSumoEnv(c) as env:
        env.reset(seed=7)
        for _ in range(5):
            env.step({'C': 0})
        _, _, _, _, info = env.step({'C': 1})
        assert info['C']['applied_action'] == 1
        for _ in range(3):
            _, _, _, _, info = env.step({'C': 0})
            assert info['C']['applied_action'] == 0


@pytest.mark.integration
def test_subscription_telemetry_equals_direct_queries(tmp_path):
    if not shutil.which('sumo'):
        pytest.skip('SUMO missing')
    c = tiny_config(55)
    fixture = Path(c['sumo']['cfg_file']).parent
    for name in ('tiny.net.xml', 'tiny.sumocfg'):
        shutil.copy2(fixture / name, tmp_path / name)
    routes = (fixture / 'routes.rou.xml').read_text()
    (tmp_path / 'routes.rou.xml').write_text(routes.replace('id="car" sigma=', 'id="car" vClass="emergency" sigma='))
    c['sumo'].update(cfg_file=str(tmp_path / 'tiny.sumocfg'), delta_time=1)
    observed_emergency_wait = False
    with MultiAgentSumoEnv(c) as env:
        env.reset(seed=7)
        for _ in range(55):
            observations, _, _, _, infos = env.step({'C': 0})
            tick = env.timeline[-1]
            lanes = env.agents['C'].incoming_lanes
            waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            emergency_waiting = sum(
                traci.vehicle.getWaitingTime(vehicle) for lane in lanes
                for vehicle in traci.lane.getLastStepVehicleIDs(lane)
                if traci.vehicle.getVehicleClass(vehicle) == 'emergency'
            )
            observed_emergency_wait |= emergency_waiting > 0
            assert infos['C']['waiting_time'] == waiting
            assert infos['C']['emergency_waiting'] == emergency_waiting
            assert infos['C']['vehicle_count'] == sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
            for index, lane in enumerate(lanes):
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
                assert observations['C'][index * 5 + 1] == pytest.approx(min(max(mean_speed / 15, 0), 1))
            assert tick['controlled_lane_queue'] == sum(traci.lane.getLastStepHaltingNumber(lane) for lane in env.controlled_lanes)
            assert tick['network_stopped'] == sum(traci.vehicle.getSpeed(vid) <= .1 for vid in traci.vehicle.getIDList())
            assert tick['signals']['C']['state'] == traci.trafficlight.getRedYellowGreenState('C')
            assert tick['signals']['C']['queue'] == env.agents['C'].get_queue_length()
    assert observed_emergency_wait


@pytest.mark.integration
def test_real_time_limit_truncates_and_excludes_warmup():
    if not shutil.which('sumo'):
        pytest.skip('SUMO missing')
    c = tiny_config(7)
    c['sumo']['warmup_seconds'] = 2
    with MultiAgentSumoEnv(c) as env:
        env.reset(seed=7)
        env.step({'C': 0})
        _, _, terminated, truncated, _ = env.step({'C': 0})
        assert truncated['__all__'] and not terminated['__all__']
        assert [t['simulated_second'] for t in env.timeline] == [3, 4, 5, 6, 7, 8, 9]
        assert env.stopped_vehicle_seconds == sum(t['network_stopped'] for t in env.timeline)
        env.close()
        summary = env.episode_summary()
        assert summary['unfinished'] == 4
