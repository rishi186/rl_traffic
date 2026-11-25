"""Portable provenance and strict schema-aware checkpoint loading."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def scenario_files(cfg_file):
    cfg = Path(cfg_file).resolve()
    root = ET.parse(cfg).getroot()
    paths = [cfg]
    for tag in ('net-file', 'route-files', 'additional-files'):
        item = root.find('.//' + tag)
        if item is not None:
            paths += [(cfg.parent / name.strip()).resolve() for name in item.get('value').split(',')]
    return {str(p): sha256(p) for p in paths}


def provenance(config):
    def command(args):
        return subprocess.check_output(args, text=True, encoding='utf-8', errors='replace').strip()
    packages = {}
    for name in ('torch', 'numpy', 'traci', 'gymnasium', 'PyYAML', 'pandas', 'matplotlib', 'fastapi'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    root = Path(__file__).resolve().parents[2]
    source = {str(p.relative_to(root)).replace('\\', '/'): sha256(p)
              for folder in ('src', 'scripts', 'web') for p in (root / folder).rglob('*')
              if p.is_file() and p.suffix in ('.py', '.jsx', '.js')
              and not {'node_modules', 'dist', '__pycache__'}.intersection(p.parts)}
    # Container images and source archives may omit Git metadata.
    try:
        revision = command(['git', '-C', str(root), 'rev-parse', 'HEAD'])
        status = command(['git', '-C', str(root), 'status', '--short'])
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'sumo': command(['sumo', '--version']).splitlines()[0],
            'git_revision': revision, 'git_status': status,
            'source_hashes': source, 'source_digest': digest(source),
            'scenario_files': scenario_files(config['sumo']['cfg_file'])}


def load_checked(agent, path, schema, allow_unverified=False, weights_only=True):
    """Legacy weights are usable only with explicit acknowledgement, never silently."""
    metadata_path = Path(str(path) + '.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get('checkpoint_sha256') and metadata['checkpoint_sha256'] != sha256(path):
        raise ValueError('Checkpoint hash does not match its metadata')
    stored = metadata.get('observation_schema')
    if stored is None and not allow_unverified:
        raise ValueError('Checkpoint observation schema is unverifiable; use --allow-unverified for a weights-only restart')
    if stored is not None and stored != schema:
        raise ValueError('Incompatible checkpoint observation schema')
    agent.load(str(path), weights_only=weights_only)
    return {'schema_verified': stored is not None, 'sha256': sha256(path), 'metadata': metadata}


def save_checked(agent, path, schema, metadata):
    agent.save(str(path))
    Path(str(path) + '.json').write_text(json.dumps({
        **metadata, 'observation_schema': schema, 'checkpoint_sha256': sha256(path)}, indent=2))


def snapshot_source(out, prov):
    """Retain exact dirty source alongside runs, so a Git SHA is never misleading."""
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(Path(out) / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in prov['source_hashes'].items():
            if sha256(root / name) != expected:
                raise ValueError('Source changed during provenance snapshot')
            archive.write(root / name, name)
        for name in ('requirements.txt', 'config.yaml', 'web/frontend/package.json', 'web/frontend/package-lock.json'):
            archive.write(root / name, name)
