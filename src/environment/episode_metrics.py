"""Network accounting and TripInfo summaries; no per-intersection throughput sums."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def planned_demand(cfg_file, end_seconds, begin_seconds=0):
    """Count explicit scheduled demand in [begin, end); fail on ambiguous flows."""
    cfg = Path(cfg_file).resolve()
    root = ET.parse(cfg).getroot()
    result = []
    for name in root.find('.//route-files').get('value').split(','):
        route = ET.parse(cfg.parent / name.strip()).getroot()
        if route.findall('flow') or route.findall('personFlow') or route.findall('person'):
            raise ValueError('Benchmark requires explicit vehicle/trip demand, not flows/persons')
        for item in list(route):
            if item.tag in ('vehicle', 'trip'):
                depart = float(item.get('depart'))
                if begin_seconds <= depart < end_seconds:
                    result.append(item.get('id'))
    if len(result) != len(set(result)):
        raise ValueError('Duplicate demand IDs')
    return set(result)


def tripinfo_summary(path, demand_ids):
    """Completed-only means/tails, plus separate censored lower bounds.

    Trip metrics cover the whole trip (including warm-up); window stopped
    vehicle-seconds are calculated independently, never from lane wait sums.
    """
    trips = [x for x in ET.parse(path).getroot().findall('tripinfo') if x.get('id') in demand_ids]
    completed = [x for x in trips if float(x.get('arrival')) >= 0 and x.get('vaporized', '') == '']
    inserted = [x for x in trips if float(x.get('depart')) >= 0]
    waits = [float(x.get('waitingTime')) for x in completed]
    durations = [float(x.get('duration')) for x in completed]
    censored = [float(x.get('waitingTime')) for x in inserted if float(x.get('arrival')) < 0]

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    return {
        'scheduled_demand': len(demand_ids), 'completed': len(completed),
        'unfinished': len(inserted) - len(completed),
        'uninserted': len(demand_ids) - len(inserted),
        'completion_fraction': len(completed) / len(demand_ids) if demand_ids else None,
        'trip_wait_completed_mean_s': mean(waits),
        'trip_wait_completed_p95_s': float(np.percentile(waits, 95)) if waits else None,
        'trip_wait_completed_max_s': max(waits) if waits else None,
        'trip_duration_completed_mean_s': mean(durations),
        'unfinished_wait_lower_bound_mean_s': mean(censored),
    }
