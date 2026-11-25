"""Seeded DQN/PPO training, validation selection and episode-boundary exact resume.

--warm-start loads weights only. --resume requires a local exact-resume bundle,
including replay, optimizer, scheduler, counters, RNG and validation state.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import signal
import shutil
import sys
import uuid

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.sumo_env import MultiAgentSumoEnv
from src.evaluation import create_agent, run_episode
from src.utils.config_validator import validate_config
from src.utils.provenance import digest, load_checked, provenance, save_checked, seed_everything, snapshot_source, scenario_files

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def _create_agent(algorithm, obs_dim, action_dim, config, device):
    """Retain the existing factory entry point for downstream callers."""
    return create_agent(algorithm, obs_dim, config)


def collect_episode(agent, env, algorithm, seed):
    """Keep behavior data attached to the observation/action that produced it."""
    observations, _ = env.reset(seed=seed)
    finished = False
    reward_sum = 0.0
    updates = []
    while not finished:
        behavior = {tid: agent.select_action(obs) for tid, obs in observations.items()}
        actions = {tid: b[0] for tid, b in behavior.items()}
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        for tid in env.ts_ids:
            if algorithm == 'ppo':
                action, log_prob, value = behavior[tid]
                agent.store_transition(tid, observations[tid], action, rewards[tid], log_prob,
                                       value, terminated[tid], next_obs[tid], truncated[tid])
            else:
                # One-step replay bootstraps time limits, never natural terminals.
                agent.store_transition(observations[tid], actions[tid], rewards[tid],
                                       next_obs[tid], terminated[tid])
            reward_sum += rewards[tid]
        if algorithm == 'dqn':
            info = agent.update()
            if info:
                updates.append(info)
        observations = next_obs
        finished = terminated['__all__'] or truncated['__all__']
    if algorithm == 'ppo':
        updates.append(agent.update())
    env.close()
    return {**env.episode_summary(), 'reward': reward_sum / len(env.ts_ids),
            'updates': updates, 'reward_label': 'controlled_lane_shaped_proxy_return'}


def resume_config_digest(config):
    c = copy.deepcopy(config)
    c['experiment'].pop('log_dir', None)
    c['training'].pop('total_episodes', None)
    return digest(c)


def train(config_path, resume=None, *, warm_start=None, allow_unverified=False, run_dir=None,
          algorithm=None, episodes=None, seed=None, horizon=None, scenario=None,
          validation_scenarios=None, validation_seeds=None):
    config = yaml.safe_load(Path(config_path).read_text())
    if algorithm:
        config['training']['algorithm'] = algorithm
    if episodes is not None:
        config['training']['total_episodes'] = episodes
    if seed is not None:
        config['experiment']['seed'] = seed
    if horizon is not None:
        config['sumo']['num_seconds'] = horizon
    if scenario:
        config['sumo']['cfg_file'] = scenario
    v = config.setdefault('validation', {})
    v['scenarios'] = validation_scenarios or v.get('scenarios', [config['sumo']['cfg_file']])
    v['seeds'] = validation_seeds or v.get('seeds', [7001])
    validate_config(config)
    if resume and warm_start:
        raise ValueError('Choose exact resume OR weights-only warm start')
    if config.get('curriculum', {}).get('enabled'):
        raise ValueError('Runtime density curriculum unsupported; use explicit scenario manifests')
    if config['training'].get('dqn', {}).get('n_step', 1) != 1:
        raise ValueError('Benchmark training requires n_step=1; legacy multi-agent n-step collector is unsafe')
    if config['training'].get('dqn', {}).get('num_agents', 0):
        raise ValueError('Benchmark shares weights without an agent-ID embedding; num_agents must be 0')
    algorithm = config['training']['algorithm']
    seed_everything(config['experiment']['seed'])
    name = f"{algorithm}-{config['experiment']['seed']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    out = Path(run_dir or Path(config['experiment']['log_dir']) / name).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    (out / 'config_snapshot.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    prov = provenance(config)
    prov['validation_scenario_files'] = {path: scenario_files(path) for path in v['scenarios']}
    (out / 'provenance.json').write_text(json.dumps(prov, indent=2))
    snapshot_source(out, prov)
    start, best, metrics = 0, float('inf'), []
    state = None
    if resume:
        # Only load trusted local bundles; torch pickle can execute code.
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['config_digest'] != resume_config_digest(config):
            raise ValueError('Exact resume configuration mismatch')
        if state['source_digest'] != prov['source_digest'] or state['packages'] != prov['packages']:
            raise ValueError('Exact resume source/dependency mismatch; use --warm-start')
        if state['sumo_version'] != prov['sumo'] or state['scenario_files'] != prov['scenario_files']:
            raise ValueError('Exact resume simulator/scenario mismatch')
        if state['validation_scenario_files'] != prov['validation_scenario_files']:
            raise ValueError('Exact resume validation scenario mismatch')
        if state['python'] != prov['python']:
            raise ValueError('Exact resume Python mismatch')
    with MultiAgentSumoEnv(config) as env:
        obs, _ = env.reset(seed=config['experiment']['seed'])
        schema = copy.deepcopy(env.observation_schema)
        env.close()
        agent = create_agent(algorithm, len(next(iter(obs.values()))), config)
        agent.observation_schema = schema
        if warm_start:
            loaded = load_checked(agent, warm_start, schema, allow_unverified, weights_only=True)
            (out / 'warm_start.json').write_text(json.dumps(loaded, indent=2))
        if state:
            model = Path(resume).resolve().parent / state['checkpoint']
            load_checked(agent, model, schema, weights_only=False)
            if algorithm == 'dqn':
                agent.replay_buffer = state['replay_buffer']
            start, best, metrics = state['next_episode'], state['best_validation'], state['metrics']
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if torch.cuda.is_available() and state.get('cuda_rng'):
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            for suffix in ('', '.json'):
                source = Path(resume).resolve().parent / ('models/best_model.pth' + suffix)
                shutil.copy2(source, out / ('models/best_model.pth' + suffix))
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(out / 'tensorboard')) as writer:
            for episode in range(start, config['training']['total_episodes']):
                row = collect_episode(agent, env, algorithm, config['experiment']['seed'] + episode)
                row['episode'] = episode + 1
                validations = []
                for cfg_file in v['scenarios']:
                    vc = copy.deepcopy(config)
                    vc['sumo']['cfg_file'] = cfg_file
                    for validation_seed in v['seeds']:
                        result, _, _, _ = run_episode(vc, algorithm, validation_seed, agent=agent)
                        validations.append({'scenario': cfg_file, **result})
                score = float(np.mean([r['network_stopped_vehicle_seconds'] for r in validations]))
                row['validation'] = validations
                row['validation_stopped_vehicle_seconds'] = score
                metrics.append(row)
                metadata = {'algorithm': algorithm, 'episode': episode + 1, 'training_seed': config['experiment']['seed'],
                            'validation_metric': 'network_stopped_vehicle_seconds', 'validation_score': score,
                            'config': config, 'source_digest': prov['source_digest']}
                checkpoint = out / 'models' / f'checkpoint_ep{episode + 1}.pth'
                save_checked(agent, checkpoint, schema, metadata)
                if score < best:
                    best = score
                    save_checked(agent, out / 'models/best_model.pth', schema, metadata)
                save_checked(agent, out / 'models/final_model.pth', schema, metadata)
                bundle = {'next_episode': episode + 1, 'best_validation': best, 'metrics': metrics,
                          'checkpoint': str(checkpoint.relative_to(out)), 'config_digest': resume_config_digest(config),
                          'source_digest': prov['source_digest'], 'packages': prov['packages'],
                          'sumo_version': prov['sumo'], 'scenario_files': prov['scenario_files'], 'python': prov['python'],
                          'validation_scenario_files': prov['validation_scenario_files'],
                          'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                          'torch_rng': torch.get_rng_state(),
                          'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                          'replay_buffer': agent.replay_buffer if algorithm == 'dqn' else None}
                torch.save(bundle, out / 'resume.pt')
                (out / 'training_metrics.json').write_text(json.dumps(metrics, indent=2))
                writer.add_scalar('Train/ControlledLaneProxyReturn', row['reward'], episode + 1)
                writer.add_scalar('Validation/NetworkStoppedVehicleSeconds', score, episode + 1)
                print(f"{algorithm} episode={episode + 1} reward_proxy={row['reward']:.3f} validation_stopped_vehicle_seconds={score:.1f}", flush=True)
                if _shutdown_requested:
                    break
    print(f'RUN_DIR={out}', flush=True)
    return str(out)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _signal_handler)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--resume', help='Trusted local resume.pt bundle, exact episode-boundary resume')
    p.add_argument('--warm-start', help='Weights-only checkpoint restart')
    p.add_argument('--allow-unverified', action='store_true')
    p.add_argument('--run-dir')
    p.add_argument('--algorithm', choices=['dqn', 'ppo'])
    p.add_argument('--episodes', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--horizon', type=int)
    p.add_argument('--scenario')
    p.add_argument('--validation-scenarios', nargs='+')
    p.add_argument('--validation-seeds', nargs='+', type=int)
    args = p.parse_args()
    train(args.config, args.resume, warm_start=args.warm_start, allow_unverified=args.allow_unverified,
          run_dir=args.run_dir, algorithm=args.algorithm, episodes=args.episodes, seed=args.seed,
          horizon=args.horizon, scenario=args.scenario,
          validation_scenarios=args.validation_scenarios, validation_seeds=args.validation_seeds)
