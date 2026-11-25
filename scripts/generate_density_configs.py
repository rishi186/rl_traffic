"""Create explicit, seeded complete demand files. No runtime vehicle removal.

Preserves vTypes, named routes and inline routes. Rejects flows/distributions
instead of silently producing incomplete demand. Rush-hour changes departure
shape while preserving the selected base vehicles.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.provenance import sha256


def _scale_route_file(source_path, dest_path, multiplier, seed=42, demand_seconds=None, rush=False):
    root = ET.parse(source_path).getroot()
    if multiplier <= 0:
        raise ValueError('Multiplier must be positive')
    if any(x.tag in ('flow', 'person', 'personFlow', 'routeDistribution', 'vTypeDistribution', 'trip') for x in root):
        raise ValueError('Expand flows/distributions/trips into explicit routed vehicles first')
    vehicles = [x for x in root if x.tag == 'vehicle']
    if any(not x.get('route') and x.find('route') is None for x in vehicles):
        raise ValueError('Vehicle without a complete route')
    if demand_seconds is not None:
        vehicles = [x for x in vehicles if float(x.get('depart')) < demand_seconds]
    rng = random.Random(seed)
    count = round(len(vehicles) * multiplier)
    if count and not vehicles:
        raise ValueError('Empty base demand')
    if count <= len(vehicles):
        selected = rng.sample(vehicles, count) if multiplier != 1 else vehicles
    else:
        selected = vehicles + rng.choices(vehicles, k=count - len(vehicles))
    out = ET.Element('routes')
    for item in root:
        if item.tag != 'vehicle':
            out.append(copy.deepcopy(item))
    horizon = demand_seconds or max(float(x.get('depart')) for x in vehicles) + 1
    demand = []
    for i, item in enumerate(selected):
        x = copy.deepcopy(item)
        x.set('id', f'v{i:06d}')
        depart = float(item.get('depart'))
        if i >= len(vehicles):
            depart = min(horizon - .001, depart + rng.uniform(.01, .5))
        if rush:
            u = depart / horizon
            if u < .1:
                depart = horizon * u / .3
            elif u < .9:
                depart = horizon / 3 + horizon / 3 * (u - .1) / .8
            else:
                depart = 2 * horizon / 3 + horizon / 3 * (u - .9) / .1
        x.set('depart', f'{depart:.3f}')
        demand.append(x)
    out.extend(sorted(demand, key=lambda x: (float(x.get('depart')), x.get('id'))))
    ET.indent(out)
    ET.ElementTree(out).write(dest_path, encoding='utf-8', xml_declaration=True)
    return len(demand)


def _create_sumocfg(net_file, route_file, output_path):
    root = ET.Element('configuration')
    inp = ET.SubElement(root, 'input')
    ET.SubElement(inp, 'net-file', value=str(net_file))
    ET.SubElement(inp, 'route-files', value=str(route_file))
    ET.indent(root)
    ET.ElementTree(root).write(output_path, encoding='utf-8', xml_declaration=True)


def generate(config_path, output_dir=None, seed=42, demand_seconds=None):
    c = yaml.safe_load(Path(config_path).read_text())
    source = Path(c['sumo']['cfg_file']).resolve()
    cfg = ET.parse(source).getroot()
    net = (source.parent / cfg.find('.//net-file').get('value')).resolve()
    routes = cfg.find('.//route-files').get('value').split(',')
    if len(routes) != 1 or cfg.find('.//additional-files') is not None:
        raise ValueError('Merge multiple route/additional files explicitly before demand generation')
    route = source.parent / routes[0]
    out = Path(output_dir or source.parent / 'density_variants').resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Copy network so the scenario bundle is portable and hashes are stable.
    import shutil
    network_name = 'network.net.xml.gz' if net.suffix == '.gz' else 'network.net.xml'
    shutil.copy2(net, out / network_name)
    scenarios = []
    for name, mult, rush in [('light', .5, False), ('normal', 1., False), ('heavy', 1.5, False), ('rush_hour', 1., True)]:
        count = _scale_route_file(route, out / f'{name}.rou.xml', mult, seed, demand_seconds, rush)
        _create_sumocfg(network_name, f'{name}.rou.xml', out / f'{name}.sumocfg')
        scenarios.append({'name': name, 'multiplier': mult, 'held_out': rush,
                          'cfg_file': f'{name}.sumocfg', 'vehicles': count,
                          'route_sha256': sha256(out / f'{name}.rou.xml')})
    manifest = {'schema_version': 2, 'generation_seed': seed, 'demand_seconds': demand_seconds,
                'source_route_sha256': sha256(route), 'network_sha256': sha256(net), 'scenarios': scenarios}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return str(out / 'manifest.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--output-dir')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--demand-seconds', type=int)
    args = p.parse_args()
    generate(args.config, args.output_dir, args.seed, args.demand_seconds)
