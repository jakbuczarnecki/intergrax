# © Artur Czarnecki. All rights reserved.

"""EBH-7 Wave A2 — adversarial semantic integrity for promoted TEN-03/05 bindings."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_a2_ten_e2e_03_gov_exec_projection_is_fail_closed() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_gov_exec_cross_tenant_projection_denied,
    )

    assert_gov_exec_cross_tenant_projection_denied()


def test_a2_ten_e2e_04_invoker_and_tool_runtime_fail_closed() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_exec_tool_cross_tenant_invocation_denied,
    )

    assert_exec_tool_cross_tenant_invocation_denied()


@pytest.mark.asyncio
async def test_a2_ten_e2e_04_tool_runtime_chain_fail_closed() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_exec_tool_cross_tenant_via_tool_runtime,
    )

    await assert_exec_tool_cross_tenant_via_tool_runtime()


def test_a2_ten_e2e_05_credential_scope_rejects_cross_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_config_provider_cross_tenant_rejected,
    )

    assert_config_provider_cross_tenant_rejected()
