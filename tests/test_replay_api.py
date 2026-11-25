import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from web.server import app


def test_default_replay_uses_tirupati(monkeypatch):
    monkeypatch.delenv('RL_TRAFFIC_BENCHMARK_DIR', raising=False)
    response = TestClient(app).get('/api/benchmark/replay')
    assert response.status_code == 200
    replay = response.json()
    assert replay['label'].startswith('Tirupati demo')
    assert [panel['policy'] for panel in replay['panels']] == ['fixed', 'dqn']
    assert all(len(panel['timeline']) == 600 for panel in replay['panels'])


def test_missing_replay_is_not_fabricated(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_TRAFFIC_BENCHMARK_DIR', str(tmp_path))
    response = TestClient(app).get('/api/benchmark/replay')
    assert response.status_code == 404
    assert 'metrics' not in response.json()


def test_saved_replay_is_matched_and_tamper_detected(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'docs/benchmark/pilot'
    if not source.exists():
        import pytest
        pytest.skip('Pilot artifact not generated yet')
    for name in ('replay.json', 'raw.csv', 'manifest.json'):
        shutil.copy2(source / name, tmp_path / name)
    monkeypatch.setenv('RL_TRAFFIC_BENCHMARK_DIR', str(tmp_path))
    client = TestClient(app)
    response = client.get('/api/benchmark/replay')
    assert response.status_code == 200
    data = response.json()
    left, right = data['panels']
    assert left['match_id'] == right['match_id'] == data['match_id']
    assert [t['elapsed_s'] for t in left['timeline']] == [t['elapsed_s'] for t in right['timeline']]
    data['panels'][0]['timeline'][0]['network_stopped'] += 100
    (tmp_path / 'replay.json').write_text(json.dumps(data))
    assert client.get('/api/benchmark/replay').status_code == 409
    (tmp_path / 'raw.csv').write_text('corrupted')
    assert client.get('/api/benchmark/replay').status_code == 409
