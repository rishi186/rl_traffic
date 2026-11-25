import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.train import collect_episode, train
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.provenance import load_checked, save_checked, seed_everything
from tests.test_episode_semantics import tiny_config


def test_gae_terminal_truncation_and_trace_boundaries():
    c = tiny_config()
    c['training']['gamma'] = .9
    c['training']['ppo']['gae_lambda'] = 1
    agent = PPOAgent(2, 2, c, 'cpu')
    adv, ret = agent.compute_gae(
        [1, 2, 3], [10, 20, 30], [False, False, True], 999,
        [False, True, False], [20, 40, 999])
    # t=1 bootstraps 40, but never includes the next episode's residual.
    assert adv == pytest.approx([25.2, 18, -27])
    assert ret == pytest.approx([35.2, 38, 3])


def test_ppo_one_transition_finite_and_true_next_state():
    seed_everything(12)
    c = tiny_config()
    c['training']['ppo']['ppo_epochs'] = 2
    a = PPOAgent(2, 2, c, 'cpu')
    obs, nxt = np.array([0., 1.], np.float32), np.array([1., 0.], np.float32)
    action, logp, value = a.select_action(obs)
    a.store_transition('x', obs, action, 2., logp, value, False, nxt, True)
    obs[:] = 5  # storage owns its data
    assert a.storage['x'][0]['obs'][0] == 0
    assert np.array_equal(a.storage['x'][0]['next_obs'], nxt)
    before = copy.deepcopy(a.policy.state_dict())
    assert all(np.isfinite(v) for v in a.update().values())
    assert any(not torch.equal(before[k], v) for k, v in a.policy.state_dict().items())


class TwoStepEnv:
    ts_ids = ['a', 'b']

    def reset(self, seed):
        self.i = 0
        return {t: np.array([0., j], np.float32) for j, t in enumerate(self.ts_ids)}, {}

    def step(self, actions):
        self.i += 1
        obs = {t: np.array([self.i, j], np.float32) for j, t in enumerate(self.ts_ids)}
        term = {t: False for t in self.ts_ids + ['__all__']}
        trunc = {t: self.i == 2 for t in self.ts_ids + ['__all__']}
        return obs, {t: float(self.i) for t in self.ts_ids}, term, trunc, {}

    def close(self):
        pass

    def episode_summary(self):
        return {}


def test_training_loop_behavior_alignment():
    class RecordingPPO(PPOAgent):
        def select_action(self, obs, deterministic=False):
            return int(obs[0]) % 2, float(obs[0] + 10), float(obs[0] + 20)

        def update(self):
            self.recorded = copy.deepcopy(self.storage)
            return {}
    a = RecordingPPO(2, 2, tiny_config(), 'cpu')
    collect_episode(a, TwoStepEnv(), 'ppo', 2)
    for trajectory in a.recorded.values():
        for t, transition in enumerate(trajectory):
            assert transition['log_prob'] == t + 10
            assert transition['value'] == t + 20
            assert transition['obs'][0] == t
            assert transition['next_obs'][0] == t + 1
            assert transition['action'] == t % 2
        assert trajectory[-1]['truncated'] and not trajectory[-1]['done']


def test_dqn_replay_keeps_true_next_and_terminal_mask():
    c = tiny_config()
    c['training']['batch_size'] = 64
    agent = DQNAgent(2, 2, c)
    for seed in (1, 2):
        collect_episode(agent, TwoStepEnv(), 'dqn', seed)
    assert len(agent.replay_buffer) == 8
    for state, action, reward, nxt, terminal in agent.replay_buffer.buffer:
        assert nxt[0] == state[0] + 1
        assert nxt[1] == state[1]  # no cross-signal or cross-episode transition
        assert terminal is False  # bootstrap artificial time limit
    agent.store_transition(np.zeros(2), 0, 1, np.ones(2), True)
    assert agent.replay_buffer.buffer[-1][-1] is True


@pytest.mark.parametrize('algorithm', ['dqn', 'ppo'])
def test_checked_checkpoint_inference_and_schema(tmp_path, algorithm):
    c = tiny_config()
    c['training']['algorithm'] = algorithm
    agent = DQNAgent(2, 2, c) if algorithm == 'dqn' else PPOAgent(2, 2, c, 'cpu')
    schema = {'version': 2, 'lanes': ['a', 'b']}
    path = tmp_path / 'model.pth'
    obs = np.array([.2, .9], np.float32)
    expected = agent.select_action(obs, deterministic=True)
    save_checked(agent, path, schema, {})
    load_checked(agent, path, schema)
    assert agent.select_action(obs, deterministic=True) == expected
    with pytest.raises(ValueError, match='Incompatible'):
        load_checked(agent, path, {'version': 2, 'lanes': ['b', 'a']})
    Path(str(path) + '.json').unlink()
    with pytest.raises(ValueError, match='unverifiable'):
        load_checked(agent, path, schema)
    assert not load_checked(agent, path, schema, allow_unverified=True)['schema_verified']


@pytest.mark.integration
@pytest.mark.parametrize('algorithm', ['dqn', 'ppo'])
def test_exact_episode_resume_matches_uninterrupted(tmp_path, algorithm):
    import shutil
    if not shutil.which('sumo'):
        pytest.skip('SUMO missing')
    c = tiny_config(12)
    c['training']['algorithm'] = algorithm
    c['training']['batch_size'] = 2
    c['training']['ppo']['ppo_epochs'] = 1
    c['training']['dqn']['hidden_dims'] = [8]
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump(c))
    full = train(cfg, episodes=2, run_dir=tmp_path / 'full')
    first = train(cfg, episodes=1, run_dir=tmp_path / 'first')
    resumed = train(cfg, str(Path(first) / 'resume.pt'), episodes=2, run_dir=tmp_path / 'resumed')
    a = torch.load(Path(full) / 'models/final_model.pth', weights_only=True)
    b = torch.load(Path(resumed) / 'models/final_model.pth', weights_only=True)
    key = 'q_network_state_dict' if algorithm == 'dqn' else 'policy_state_dict'
    assert all(torch.equal(a[key][k], b[key][k]) for k in a[key])
