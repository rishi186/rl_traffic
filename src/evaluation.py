"""Shared runner for normal evaluation entry points and paired benchmarking."""
import copy
import json
from pathlib import Path
import time

from src.environment.sumo_env import MultiAgentSumoEnv
from src.utils.provenance import digest, load_checked, scenario_files


def create_agent(algorithm, obs_dim, config):
    if algorithm == 'dqn':
        from src.agents.dqn_agent import DQNAgent
        return DQNAgent(obs_dim, 2, config)
    if algorithm == 'ppo':
        import torch
        from src.agents.ppo_agent import PPOAgent
        device = config['experiment'].get('device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        if device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'
        return PPOAgent(obs_dim, 2, config, device)
    raise ValueError('Expected dqn or ppo')


def run_episode(config, policy, seed, model_path=None, agent=None, allow_unverified=False, record=False):
    cfg = copy.deepcopy(config)
    cfg['sumo'].update(control_mode=policy if policy in ('fixed', 'actuated') else 'rl', record_timeline=record)
    started = time.perf_counter()
    with MultiAgentSumoEnv(cfg) as env:
        obs, _ = env.reset(seed=seed)
        checkpoint = None
        if agent is not None and getattr(agent, 'observation_schema', env.observation_schema) != env.observation_schema:
            raise ValueError('Validation observation schema differs from training')
        if policy in ('dqn', 'ppo') and agent is None:
            metadata_path = Path(str(model_path) + '.json')
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            if metadata.get('algorithm', policy) != policy:
                raise ValueError('Checkpoint algorithm mismatch')
            agent = create_agent(policy, len(next(iter(obs.values()))), metadata.get('config', cfg))
            checkpoint = load_checked(agent, model_path, env.observation_schema, allow_unverified)
        done = False
        while not done:
            actions = ({tid: agent.select_action(o, deterministic=True)[0] for tid, o in obs.items()}
                       if agent is not None else {})
            obs, _, terms, truncs, _ = env.step(actions)
            done = terms['__all__'] or truncs['__all__']
        env.close()
        result = env.episode_summary()
        common = {'scenario_hashes': sorted(scenario_files(cfg['sumo']['cfg_file']).values()),
                  'seed': seed, 'timing': vars(env.timing),
                  'min_green': env.min_green, 'max_green': env.max_green,
                  'yellow_time': env.yellow_time, 'all_red_time': cfg['sumo'].get('all_red_time', 1),
                  'terminate_when_empty': cfg['sumo'].get('terminate_when_empty', False),
                  'simulated_seconds': result['simulated_seconds'], 'schema': env.observation_schema}
        result.update(policy=policy, match_id=digest(common), runtime_seconds=time.perf_counter() - started)
        replay = {'policy': policy, 'match_id': result['match_id'], 'seed': seed,
                  'observation_schema': env.observation_schema, 'metrics': result,
                  'timeline': env.timeline, 'actions': env.action_log}
        return result, replay, Path(env.tripinfo_path).read_text(), checkpoint


def evaluate_policy(config, policy, seeds, output_dir, model=None, allow_unverified=False):
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Evaluation requires a nonempty unique seed list')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        row, replay, tripinfo, _ = run_episode(config, policy, seed, model, allow_unverified=allow_unverified, record=True)
        rows.append(row)
        (out / f'{policy}-{seed}.tripinfo.xml').write_text(tripinfo)
        (out / f'{policy}-{seed}.replay.json').write_text(json.dumps(replay))
    summary = {'schema_version': 2, 'policy': policy, 'episodes': rows}
    (out / ('baseline_results.json' if policy == 'fixed' else 'eval_results.json')).write_text(json.dumps(summary, indent=2))
    return summary
