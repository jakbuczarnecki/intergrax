# © Artur Czarnecki. All rights reserved.

"""EBH-7 W1 integrated tenant boundary proofs (RC-03/04/05/13)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
    assert_config_provider_cross_tenant_rejected,
    assert_exec_tool_cross_tenant_invocation_denied,
    assert_gov_exec_cross_tenant_projection_denied,
    assert_plugin_host_compat_cross_tenant_activation_denied,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_integrated_ten_e2e_03_gov_exec_cross_tenant_denied() -> None:
    assert_gov_exec_cross_tenant_projection_denied()


def test_ebh7_integrated_ten_e2e_04_exec_tool_cross_tenant_denied() -> None:
    assert_exec_tool_cross_tenant_invocation_denied()


def test_ebh7_integrated_ten_e2e_05_config_provider_cross_tenant() -> None:
    assert_config_provider_cross_tenant_rejected()


def test_ebh7_integrated_ten_e2e_13_plugin_host_cross_tenant_denied() -> None:
    assert_plugin_host_compat_cross_tenant_activation_denied()
