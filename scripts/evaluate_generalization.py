"""Paired benchmark: fixed, SUMO actuated, DQN and PPO on explicit demand manifests."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark import generate_artifacts
from src.evaluation import run_episode
from src.utils.provenance import provenance, scenario_files, sha256, snapshot_source, seed_everything


def benchmark(config_path, demand_manifest, dqn_models, ppo_models, seeds, output_dir,
              horizon=None, scenarios=None, label='Pilot benchmark'):
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('Evaluation seeds must be nonempty and unique')
    config = yaml.safe_load(Path(config_path).read_text())
    seed_everything(config['experiment']['seed'])
    if horizon:
        config['sumo']['num_seconds'] = horizon
    config['sumo']['terminate_when_empty'] = False
    dm = Path(demand_manifest).resolve()
    demand = json.loads(dm.read_text())
    selected = scenarios or ['light', 'normal', 'heavy']
    variants = [v for v in demand['scenarios'] if v['name'] in selected]
    if {v['name'] for v in variants} != set(selected):
        raise ValueError('Missing requested demand scenario')
    policies = [('fixed', None, ''), ('actuated', None, '')]
    for policy, paths in [('dqn', dqn_models), ('ppo', ppo_models)]:
        seen = set()
        for path in paths:
            meta = json.loads(Path(str(path) + '.json').read_text())
            training_seed = meta['training_seed']
            if training_seed in seen:
                raise ValueError('Duplicate training seed for policy')
            seen.add(training_seed)
            policies.append((policy, str(Path(path).resolve()), training_seed))
    if not dqn_models or not ppo_models:
        raise ValueError('Provide validation-selected checkpoints for both DQN and PPO')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 2, 'label': label, 'config': config, 'evaluation_seeds': seeds,
                'demand_manifest': demand, 'demand_manifest_sha256': sha256(dm),
                'models': {str(p): sha256(p) for p in dqn_models + ppo_models},
                'provenance': provenance(config), 'runs': [],
                'limitations': [
                    'Pilot or finite training budget; convergence is not established.',
                    'Intervals resample matched evaluation seeds within each training seed; they do not estimate training-seed population uncertainty.',
                    'Completed-only TripInfo waiting/travel times exclude censored unfinished and uninserted trips; inspect completion before comparing them.',
                    'Controlled-lane queue is an incoming-lane snapshot metric, not a whole-network queue or a trip mean.',
                    'Shaped rewards use incoming-count and lane-wait proxies, not movement pressure or measured intersection throughput.',
                    'Derived common plans yellow all released greens and insert all-red. Fixed is this bounded derived plan, not the unmodified source program.',
                    'Simulation demand and signal plans have no field calibration.',
                ]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    snapshot_source(out, manifest['provenance'])
    writer = None
    with (out / 'raw.csv').open('w', newline='') as f:
        for variant in variants:
            c = copy.deepcopy(config)
            c['sumo']['cfg_file'] = str(dm.parent / variant['cfg_file'])
            import xml.etree.ElementTree as ET
            cfg_xml = ET.parse(c['sumo']['cfg_file']).getroot()
            route = dm.parent / cfg_xml.find('.//route-files').get('value')
            net = dm.parent / cfg_xml.find('.//net-file').get('value')
            if sha256(route) != variant['route_sha256'] or sha256(net) != demand['network_sha256']:
                raise ValueError('Demand manifest hash mismatch')
            for seed in seeds:
                for policy, model, training_seed in policies:
                    row, replay, tripinfo, checkpoint = run_episode(c, policy, seed, model, record=True)
                    row.update(scenario=variant['name'], training_seed=training_seed)
                    tag = f"{variant['name']}-{seed}-{policy}-{training_seed or 'rule'}"
                    (out / f'{tag}.tripinfo.xml').write_text(tripinfo)
                    (out / f'{tag}.replay.json').write_text(json.dumps(replay))
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    manifest['runs'].append({
                        'id': tag, 'match_id': row['match_id'], 'metrics': row,
                        'tripinfo': f'{tag}.tripinfo.xml', 'replay': f'{tag}.replay.json',
                        'checkpoint': checkpoint, 'scenario_files': scenario_files(c['sumo']['cfg_file'])})
                    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
                    print(f"{tag}: {row['runtime_seconds']:.2f}s wall, stopped={row['network_stopped_vehicle_seconds']}, completed={row['completed']}/{row['scheduled_demand']}", flush=True)
    manifest['raw_csv_sha256'] = sha256(out / 'raw.csv')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = generate_artifacts(out)
    # One actual matched replay, compacted to the busiest recorded signal.
    base_runs = [r for r in manifest['runs'] if r['metrics']['policy'] == 'fixed']
    if base_runs:
        first = next((r for r in base_runs if r['metrics']['scenario'] == 'normal'), base_runs[0])
        peer = next(r for r in manifest['runs'] if r['match_id'] == first['match_id'] and r['metrics']['policy'] == 'dqn')
        left = json.loads((out / first['replay']).read_text())
        right = json.loads((out / peer['replay']).read_text())
        signal_scores = {tid: sum(t['signals'][tid]['queue'] for p in (left, right) for t in p['timeline'])
                         for tid in left['observation_schema']['signals']}
        signal = max(signal_scores, key=signal_scores.get)
        for panel in (left, right):
            for tick in panel['timeline']:
                tick['signal'] = tick.pop('signals')[signal]
        pair = {'schema_version': 2, 'label': label, 'scenario': first['metrics']['scenario'],
                'models': manifest['models'],
                'seed': first['metrics']['seed'], 'match_id': first['match_id'],
                'signal_id': signal, 'provenance': manifest['provenance'],
                'manifest_sha256': sha256(out / 'manifest.json'), 'raw_csv_sha256': manifest['raw_csv_sha256'],
                'limitations': manifest['limitations'], 'panels': [left, right]}
        (out / 'replay.json').write_text(json.dumps(pair, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--manifest', required=True, help='Explicit demand manifest from generate_density_configs.py')
    p.add_argument('--dqn-model', nargs='+', required=True)
    p.add_argument('--ppo-model', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[101, 102, 103, 104, 105])
    p.add_argument('--scenarios', nargs='+', default=['light', 'normal', 'heavy'])
    p.add_argument('--horizon', type=int)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--label', default='Pilot benchmark')
    args = p.parse_args()
    benchmark(args.config, args.manifest, args.dqn_model, args.ppo_model, args.seeds,
              args.output_dir, args.horizon, args.scenarios, args.label)
