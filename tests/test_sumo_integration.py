"""CPU-friendly real simulator checks, never substituted with mocks."""
import shutil
from pathlib import Path

import pytest
import traci


@pytest.mark.integration
def test_real_sumo_start_step_close():
    if not shutil.which("sumo"):
        pytest.skip("SUMO executable not on PATH")
    cfg = Path(__file__).parent / "fixtures/tiny/tiny.sumocfg"
    traci.start(["sumo", "-c", str(cfg), "--seed", "7", "--no-step-log", "true"])
    try:
        assert traci.trafficlight.getIDList() == ("C",)
        for _ in range(10):
            traci.simulationStep()
        assert traci.simulation.getTime() == 10
        assert traci.simulation.getDepartedNumber() >= 0
    finally:
        traci.close()
    assert not traci.isLoaded()
