"""Matched comparisons and derived artifacts; CSV is the numerical source of truth."""
import csv
import json
from pathlib import Path

import numpy as np

PRIMARY = 'network_stopped_vehicle_seconds'
METRICS = [PRIMARY, 'controlled_lane_queue_mean', 'completion_fraction', 'completed', 'unfinished', 'uninserted',
           'trip_wait_completed_mean_s', 'trip_wait_completed_p95_s', 'trip_wait_completed_max_s',
           'trip_duration_completed_mean_s', 'unfinished_wait_lower_bound_mean_s', 'runtime_seconds']


def read_rows(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in METRICS:
            r[k] = float(r[k]) if r.get(k) not in (None, '', 'None') else None
        r['seed'] = int(r['seed'])
    return rows


def paired_summary(rows):
    groups = sorted({(r['scenario'], r['policy'], str(r.get('training_seed', ''))) for r in rows})
    baselines = {}
    for r in rows:
        if r['policy'] == 'fixed':
            key = (r['scenario'], r['match_id'])
            if key in baselines:
                raise ValueError('Duplicate fixed-time match')
            baselines[key] = r
    result = []
    for scenario, policy, training_seed in groups:
        group = [r for r in rows if (r['scenario'], r['policy'], str(r.get('training_seed', ''))) == (scenario, policy, training_seed)]
        keys = [r['match_id'] for r in group]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate policy match')
        missing = [r['seed'] for r in group if (scenario, r['match_id']) not in baselines]
        baseline_seeds = {r['seed'] for r in rows if r['scenario'] == scenario and r['policy'] == 'fixed'}
        missing += sorted(baseline_seeds - {r['seed'] for r in group})
        pairs = [(r, baselines[(scenario, r['match_id'])]) for r in group if (scenario, r['match_id']) in baselines]
        diff = np.array([r[PRIMARY] - b[PRIMARY] for r, b in pairs], dtype=float)
        baseline_mean = float(np.mean([b[PRIMARY] for _, b in pairs])) if pairs else None
        ci = None
        if len(diff) > 1:
            draws = np.random.default_rng(42).choice(diff, (10000, len(diff)), replace=True).mean(axis=1)
            ci = np.percentile(draws, [2.5, 97.5]).tolist()
        means = {k: float(np.mean([r[k] for r in group if r.get(k) is not None]))
                 if any(r.get(k) is not None for r in group) else None for k in METRICS}
        result.append({'scenario': scenario, 'policy': policy, 'training_seed': training_seed,
                       'runs': len(group), 'paired_n': len(pairs), 'missing_matches': sorted(set(missing)),
                       'means': means, 'paired_difference_stopped_vehicle_seconds': float(diff.mean()) if len(diff) else None,
                       'paired_bootstrap_95_ci': ci,
                       'improvement_pct': (-float(diff.mean()) / baseline_mean * 100
                                           if len(diff) and baseline_mean and not missing else None)})
    return {'schema_version': 2, 'primary_metric': PRIMARY,
            'uncertainty': '95% percentile bootstrap of paired mean differences, 10000 resamples, seed 42; evaluation seeds only',
            'groups': result}


def generate_artifacts(results_dir):
    """Regenerate summary, plot and report from exactly the same raw CSV."""
    out = Path(results_dir)
    rows = read_rows(out / 'raw.csv')
    summary = paired_summary(rows)
    manifest = json.loads((out / 'manifest.json').read_text())
    summary['limitations'] = manifest['limitations']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [g for g in summary['groups'] if g['policy'] != 'fixed']
    fig, ax = plt.subplots(figsize=(11, max(4, len(groups) * .4)))
    for i, g in enumerate(groups):
        val = g['paired_difference_stopped_vehicle_seconds']
        if val is None:
            continue
        ci = g['paired_bootstrap_95_ci']
        ax.plot(val, i, 'o', color='#176b87')
        if ci:
            ax.hlines(i, ci[0], ci[1], color='#176b87')
    ax.set_yticks(range(len(groups)), [f"{g['scenario']} / {g['policy']} / train {g['training_seed'] or 'rule'}" for g in groups])
    ax.axvline(0, color='grey', linewidth=1)
    ax.set_xlabel('Policy minus matched fixed-time stopped vehicle-seconds (negative is better)')
    ax.set_title(manifest['label'] + ' — paired differences and 95% bootstrap intervals')
    fig.tight_layout()
    fig.savefig(out / 'paired-comparison.png', dpi=140)
    plt.close(fig)
    all_groups = summary['groups']
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, g in enumerate(all_groups):
        fractions = [r['completion_fraction'] for r in rows
                     if (r['scenario'], r['policy'], str(r.get('training_seed', '')))
                     == (g['scenario'], g['policy'], g['training_seed']) and r['completion_fraction'] is not None]
        ax.scatter([i] * len(fractions), fractions, alpha=.7, color='#176b87')
    ax.set_xticks(range(len(all_groups)), [f"{g['scenario']}\n{g['policy']} {g['training_seed']}" for g in all_groups], rotation=35, ha='right')
    ax.set_ylabel('Completed / scheduled demand (one dot per evaluation seed)')
    ax.set_ylim(0, 1)
    ax.set_title('Completion context for the fixed evaluation window')
    fig.tight_layout()
    fig.savefig(out / 'completion.png', dpi=140)
    plt.close(fig)
    lines = ['# ' + manifest['label'], '', 'Generated from `raw.csv`; all outcomes retained.', '',
             'Primary: network stopped vehicle-seconds in the fixed measured window. TripInfo statistics are completed-only; unfinished waits are censored lower bounds.', '',
             '| Demand | Policy | Train seed | N pairs | Stopped veh-s | Completed fraction | Unfinished | Uninserted | Difference vs fixed | 95% paired CI | Improvement |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']

    def fmt(v):
        return 'N/A' if v is None else f'{v:.2f}'
    for g in summary['groups']:
        m = g['means']
        ci = g['paired_bootstrap_95_ci']
        ci_text = 'N/A' if ci is None else f'[{ci[0]:.2f}, {ci[1]:.2f}]'
        lines.append('| ' + ' | '.join([g['scenario'], g['policy'], g['training_seed'] or 'rule', str(g['paired_n']),
                     fmt(m[PRIMARY]), fmt(m['completion_fraction']), fmt(m['unfinished']), fmt(m['uninserted']),
                     fmt(g['paired_difference_stopped_vehicle_seconds']), ci_text,
                     fmt(g['improvement_pct']) + ('%' if g['improvement_pct'] is not None else '')]) + ' |')
    lines += ['', summary['uncertainty'], '', '![Paired differences](paired-comparison.png)',
              '', '![Completion by seed](completion.png)', '', '## Trip and queue context', '',
              'Each cell averages run-level statistics, including p95/max; these are not pooled trip quantiles. Completed trips exclude censored demand.', '',
              '| Demand | Policy | Train seed | Controlled queue mean (vehicles) | Completed wait mean (s) | Mean of run p95 waits (s) | Mean of run max waits (s) | Completed duration mean (s) | Unfinished wait lower bound mean (s) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in all_groups:
        m = g['means']
        keys = ('controlled_lane_queue_mean', 'trip_wait_completed_mean_s', 'trip_wait_completed_p95_s',
                'trip_wait_completed_max_s', 'trip_duration_completed_mean_s', 'unfinished_wait_lower_bound_mean_s')
        values = [g['scenario'], g['policy'], g['training_seed'] or 'rule'] + [fmt(m[k]) for k in keys]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Limits', '']
    lines += ['- ' + x for x in manifest['limitations']]
    lines += ['', '## Provenance', '', 'Full configuration, seeds, file hashes, model hashes and actual tool versions are in `manifest.json`. Individual TripInfo and replay files are indexed there.', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    from src.utils.provenance import sha256
    derived = {'raw_csv_sha256': sha256(out / 'raw.csv'), 'generator_sha256': sha256(__file__),
               'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__}
    (out / 'derived-artifacts.json').write_text(json.dumps(derived, indent=2))
    return summary
