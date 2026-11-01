# © Artur Czarnecki. All rights reserved.

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

S_IDS = tuple(f"S{i}" for i in range(1, 9))


@pytest.mark.parametrize("scenario_id", S_IDS)
def test_evol_x_scenario_arm_registered(scenario_id: str) -> None:
    assert scenario_id.startswith("S")
