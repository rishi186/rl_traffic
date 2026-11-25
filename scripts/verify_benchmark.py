"""Cross-check benchmark CSV against independent TripInfo and per-tick records."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark import paired_summary, read_rows
from src.environment.episode_metrics import planned_demand, tripinfo_summary
from src.utils.provenance import sha256


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = read_rows(directory / 'raw.csv')
    assert sha256(directory / 'raw.csv') == manifest['raw_csv_sha256']
    assert len(rows) == len(manifest['runs'])
    for entry, row in zip(manifest['runs'], rows):
        assert entry['match_id'] == row['match_id']
        replay = json.loads((directory / entry['replay']).read_text())
        ticks = replay['timeline']
        assert len(ticks) == int(row['simulated_seconds'])
        assert [t['elapsed_s'] for t in ticks] == list(range(1, len(ticks) + 1))
        assert sum(t['network_stopped'] for t in ticks) == row['network_stopped_vehicle_seconds']
        assert abs(sum(t['controlled_lane_queue'] for t in ticks) / len(ticks) - row['controlled_lane_queue_mean']) < 1e-9
        cfg = next(p for p in entry['scenario_files'] if p.endswith('.sumocfg'))
        for path, expected in entry['scenario_files'].items():
            assert sha256(path) == expected
        ids = planned_demand(cfg, float(row['window_end_s']), manifest['config']['sumo'].get('begin_time', 0))
        trip = tripinfo_summary(directory / entry['tripinfo'], ids)
        for key, value in trip.items():
            stored = row.get(key)
            if value is None:
                assert stored in (None, '')
            else:
                assert abs(float(stored) - value) < 1e-9, (entry['id'], key)
        assert int(row['completed']) + int(row['unfinished']) + int(row['uninserted']) == len(ids)
        if entry['checkpoint']:
            assert entry['checkpoint']['schema_verified']
    derived = paired_summary(rows)
    summary = json.loads((directory / 'summary.json').read_text())
    assert derived['groups'] == summary['groups']
    assert all(not g['missing_matches'] for g in derived['groups'])
    groups = {(r['scenario'], r['seed'], r['match_id']) for r in rows}
    for scenario, seed, match in groups:
        assert {r['policy'] for r in rows if (r['scenario'], r['seed'], r['match_id']) == (scenario, seed, match)} == {'fixed', 'actuated', 'dqn', 'ppo'}
    print(f'Validated {len(rows)} real runs against TripInfo, timeline, schema and matched summary')
    print('VERDICT: PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', required=True)
    args = p.parse_args()
    verify(args.results_dir)
