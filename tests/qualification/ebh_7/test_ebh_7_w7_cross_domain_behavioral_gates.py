# © Artur Czarnecki. All rights reserved.

"""EBH-7 W7 cross-domain — behavioral proof execution."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._cross_domain_scenarios import (
    cross_domain_pending_scenario_ids,
    cross_domain_qualified_scenarios,
)
from tests.qualification.ebh_7._proof_execution import execute_proof

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


@pytest.mark.parametrize(
    "scenario",
    [pytest.param(s, id=s.scenario_id) for s in cross_domain_qualified_scenarios()],
)
def test_ebh7_w7_cross_domain_executes_registered_proof(scenario) -> None:
    assert scenario.proof_module is not None
    assert scenario.proof_function is not None
    if scenario.proof_needs_tmp_path:
        pytest.skip("proof requires tmp_path")
    execute_proof(scenario.proof_module, scenario.proof_function)


def test_ebh7_w7_all_mandatory_scenarios_qualified() -> None:
    pending = cross_domain_pending_scenario_ids()
    assert pending == [], f"OPEN cross-domain scenarios: {pending}"
