# © Artur Czarnecki. All rights reserved.

"""EBH-7 W6 tenant E2E — behavioral / adversarial proof execution."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._proof_execution import execute_proof
from tests.qualification.ebh_7._semantic_proof_binding_audit import tenant_e2e_semantically_qualified_ids
from tests.qualification.ebh_7._tenant_e2e_scenarios import (
    EBH7_TENANT_E2E_SCENARIOS,
    tenant_e2e_pending_scenario_ids,
    tenant_e2e_qualified_scenarios,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


@pytest.mark.parametrize(
    "scenario",
    [pytest.param(s, id=s.scenario_id) for s in tenant_e2e_qualified_scenarios()],
)
def test_ebh7_w6_tenant_e2e_executes_registered_proof(scenario) -> None:
    assert scenario.proof_module is not None
    assert scenario.proof_function is not None
    if scenario.scenario_id == "EBH7-TEN-E2E-11":
        pytest.skip("governed_execution proof runs in tenant_platform_e2e module")
    if scenario.proof_needs_tmp_path:
        pytest.skip("proof requires tmp_path — run in dedicated fixture test")
    execute_proof(scenario.proof_module, scenario.proof_function)


def test_ebh7_w6_tenant_e2e_scenario_07_durable_cross_tenant(tmp_path) -> None:
    scenario = EBH7_TENANT_E2E_SCENARIOS[6]
    assert scenario.scenario_id == "EBH7-TEN-E2E-07"
    execute_proof(scenario.proof_module, scenario.proof_function, tmp_path=tmp_path)


def test_ebh7_w6_tenant_e2e_semantic_open_register() -> None:
    open_ids = tenant_e2e_pending_scenario_ids()
    exact = tenant_e2e_semantically_qualified_ids()
    assert len(open_ids) + len(exact) == 26
    assert len(open_ids) == 9
