# © Artur Czarnecki. All rights reserved.

"""EBH-7 Batch D — integrated TEN D1 qualification gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
    assert_config_provider_cross_tenant_rejected,
    assert_exec_tool_cross_tenant_invocation_denied,
    assert_exec_tool_cross_tenant_via_tool_runtime,
    assert_gov_exec_cross_tenant_projection_denied,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_batch_d_ten03_gov_exec_cross_tenant() -> None:
    assert_gov_exec_cross_tenant_projection_denied()


def test_ebh7_batch_d_ten04_exec_tool_invoker() -> None:
    assert_exec_tool_cross_tenant_invocation_denied()


@pytest.mark.asyncio
async def test_ebh7_batch_d_ten04_exec_tool_runtime() -> None:
    await assert_exec_tool_cross_tenant_via_tool_runtime()


def test_ebh7_batch_d_ten05_config_provider() -> None:
    assert_config_provider_cross_tenant_rejected()
