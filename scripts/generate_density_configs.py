"""Generate traffic density variant route files and SUMO configs.

Parses the base ``.rou.xml`` file and scales vehicle departure times
by a set of multipliers to produce 12+ density variants.  Each variant
gets its own ``.rou.xml`` and ``.sumocfg`` in ``maps/density_variants/``.

Usage:
    python scripts/generate_density_configs.py --config config.yaml
"""

import os
import sys
import copy
import random
import yaml
import argparse
import xml.etree.ElementTree as ET

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

DEFAULT_MULTIPLIERS = [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]


def _scale_route_file(
    source_path: str, dest_path: str, multiplier: float
) -> int:
    """Scale departure times in a route file by a multiplier.

    For multiplier < 1.0, some vehicles are removed (subsampled).
    For multiplier > 1.0, vehicles are cloned with offset departure times.
    For multiplier == 1.0, the file is copied unchanged.

    Args:
        source_path: Path to the base route XML.
        dest_path: Output path for the scaled variant.
        multiplier: Traffic density scaling factor.

    Returns:
        Number of vehicles in the output file.
    """
    tree = ET.parse(source_path)
    root = tree.getroot()

    vehicles = root.findall("vehicle")
    original_count = len(vehicles)

    if abs(multiplier - 1.0) < 1e-3:
        tree.write(dest_path, encoding="UTF-8", xml_declaration=True)
        return original_count

    if multiplier < 1.0:
        # Remove vehicles to reduce density
        random.seed(42)
        remove_count = int(original_count * (1.0 - multiplier))
        to_remove = random.sample(vehicles, k=min(remove_count, len(vehicles)))
        for v in to_remove:
            root.remove(v)
        tree.write(dest_path, encoding="UTF-8", xml_declaration=True)
        return original_count - len(to_remove)

    else:
        # Clone vehicles to increase density — add copies with offset depart times
        random.seed(42)
        clone_count = int(original_count * (multiplier - 1.0))
        to_clone = random.choices(vehicles, k=clone_count)
        for i, v in enumerate(to_clone):
            clone = copy.deepcopy(v)
            old_id = clone.get("id", f"veh_{i}")
            clone.set("id", f"{old_id}_d{multiplier:.2f}_{i}")
            # Offset depart time slightly
            try:
                old_depart = float(clone.get("depart", "0"))
                clone.set("depart", f"{old_depart + random.uniform(0.5, 3.0):.2f}")
            except (ValueError, TypeError):
                pass
            root.append(clone)

        # Sort by depart time for SUMO compatibility
        all_vehicles = root.findall("vehicle")
        all_vehicles.sort(key=lambda v: float(v.get("depart", "0")))
        # Rebuild tree
        for v in list(root):
            if v.tag == "vehicle":
                root.remove(v)
        for v in all_vehicles:
            root.append(v)

        tree.write(dest_path, encoding="UTF-8", xml_declaration=True)
        return len(all_vehicles)


def _create_sumocfg(net_file: str, route_file: str, output_path: str) -> None:
    """Write a SUMO configuration file.

    Args:
        net_file: Relative path to the network file.
        route_file: Relative path to the route file.
        output_path: Where to write the .sumocfg.
    """
    content = (
        '<configuration>\n'
        '    <input>\n'
        f'        <net-file value="{net_file}"/>\n'
        f'        <route-files value="{route_file}"/>\n'
        '    </input>\n'
        '</configuration>\n'
    )
    with open(output_path, "w") as f:
        f.write(content)


def generate(config_path: str) -> None:
    """Generate all density variant configs.

    Args:
        config_path: Path to the main YAML config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    setup_logger(log_dir=config["experiment"]["log_dir"], log_level="INFO")

    multipliers = config.get("density_variants", {}).get("multipliers", DEFAULT_MULTIPLIERS)

    # Determine base paths from sumo config
    sumo_cfg_path = config["sumo"]["cfg_file"]
    sumo_tree = ET.parse(sumo_cfg_path)
    sumo_root = sumo_tree.getroot()

    maps_dir = os.path.dirname(sumo_cfg_path)
    net_file = sumo_root.find(".//net-file").get("value")
    route_file = sumo_root.find(".//route-files").get("value")

    base_route_path = os.path.join(maps_dir, route_file)
    output_dir = os.path.join(maps_dir, "density_variants")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Generating %d density variants", len(multipliers))
    logger.info("Base route: %s", base_route_path)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    for mult in multipliers:
        tag = f"{mult:.2f}x"
        variant_route = os.path.join(output_dir, f"routes_{tag}.rou.xml")
        variant_cfg = os.path.join(output_dir, f"sumo_{tag}.sumocfg")

        veh_count = _scale_route_file(base_route_path, variant_route, mult)

        # sumocfg paths are relative to the variant directory
        _create_sumocfg(
            net_file=f"../{net_file}",
            route_file=f"routes_{tag}.rou.xml",
            output_path=variant_cfg,
        )

        logger.info("  %s — %d vehicles | cfg=%s", tag, veh_count, variant_cfg)

    logger.info("Done. Generated %d variants in %s", len(multipliers), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate traffic density variant route files"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    generate(args.config)
