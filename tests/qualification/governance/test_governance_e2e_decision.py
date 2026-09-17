# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — Decision material and Governance interaction scenarios."""

from __future__ import annotations

import pytest

from testing_support.mp4r7_enterprise_integration.composition import (
    open_mp4r7_enterprise_integration_composition,
)
from testing_support.mp4r7_enterprise_integration.contracts import Mp4R7QualificationDisposition
from testing_support.mp4r7_enterprise_integration.scenario import (
    Mp4R7EnterpriseIntegrationScenarioExecutor,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_scenario_m_decision_accepted_governance_deny_blocks_operation() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_governance_deny_after_human_approve()
    assert result.disposition is Mp4R7QualificationDisposition.QUALIFIED
    assert result.protected_operation_completed is False


@pytest.mark.asyncio
async def test_scenario_l_stale_proposal_fail_closed() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_stale_proposal()
    assert result.disposition is Mp4R7QualificationDisposition.FAIL_CLOSED


@pytest.mark.asyncio
async def test_scenario_z_cross_tenant_fail_closed() -> None:
    executor = Mp4R7EnterpriseIntegrationScenarioExecutor(
        open_mp4r7_enterprise_integration_composition(),
    )
    result = await executor.run_cross_tenant()
    assert result.disposition is Mp4R7QualificationDisposition.FAIL_CLOSED
