# © Artur Czarnecki. All rights reserved.

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_no_parallel_suite_registry_in_change_safety() -> None:
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory

    assert callable(build_closed_world_inventory)
