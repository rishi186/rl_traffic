import json
import xml.etree.ElementTree as ET

import pytest

from scripts.generate_density_configs import _scale_route_file
from src.benchmark import paired_summary, PRIMARY


def row(policy, seed, match, value):
    return {'policy': policy, 'scenario': 'light', 'training_seed': '',
            'seed': seed, 'match_id': match, PRIMARY: value}


def test_missing_match_never_yields_percentage():
    rows = [row('fixed', 1, 'a', 10), row('fixed', 2, 'b', 20), row('dqn', 1, 'a', 8)]
    group = next(g for g in paired_summary(rows)['groups'] if g['policy'] == 'dqn')
    assert group['improvement_pct'] is None
    assert group['missing_matches'] == [2]
    rows.append(row('dqn', 2, 'b', 24))
    group = next(g for g in paired_summary(rows)['groups'] if g['policy'] == 'dqn')
    assert group['improvement_pct'] == pytest.approx(-100 / 15)
    assert group['paired_difference_stopped_vehicle_seconds'] == 1
    assert group['paired_bootstrap_95_ci'] is not None
    rows[-1]['match_id'] = 'different-horizon'
    assert next(g for g in paired_summary(rows)['groups'] if g['policy'] == 'dqn')['improvement_pct'] is None


def test_zero_baseline_and_duplicate_match():
    rows = [row('fixed', 1, 'a', 0), row('dqn', 1, 'a', 1)]
    assert all(g['improvement_pct'] is None for g in paired_summary(rows)['groups'])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_summary(rows + [rows[0]])


def test_density_files_complete_deterministic_and_sorted(tmp_path):
    source = tmp_path / 'source.xml'
    source.write_text('<routes><vType id="car"/><route id="r" edges="a b"/>'
                      '<vehicle id="a" type="car" route="r" depart="0"/>'
                      '<vehicle id="b" type="car" depart="1"><route edges="a b"/></vehicle></routes>')
    dest = tmp_path / 'dest.xml'
    assert _scale_route_file(source, dest, 2.5) == 5
    original = dest.read_bytes()
    assert _scale_route_file(source, dest, 2.5) == 5
    assert original == dest.read_bytes()
    root = ET.parse(dest).getroot()
    assert root.find('vType').get('id') == 'car'
    assert root.find('route').get('id') == 'r'
    vs = root.findall('vehicle')
    assert len({v.get('id') for v in vs}) == 5
    assert [float(v.get('depart')) for v in vs] == sorted(float(v.get('depart')) for v in vs)
    assert all(v.get('route') == 'r' or v.find('route') is not None for v in vs)
    source.write_text('<routes><flow id="f" number="5"/></routes>')
    with pytest.raises(ValueError, match='explicit'):
        _scale_route_file(source, dest, 1)
    # Test data are not benchmark measurements.
    json.dumps(paired_summary([row('fixed', 1, 'a', 0)]), allow_nan=False)
