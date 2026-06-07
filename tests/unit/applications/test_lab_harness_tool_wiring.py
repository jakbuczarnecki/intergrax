# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.registry.profile import IntegrationProfile
from lab_application.host.tool_wiring import wire_lab_tools

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_harness_lab_tools_include_runtime_bound_catalog_tools() -> None:
    wiring = wire_lab_tools(
        integration_profile=IntegrationProfile.harness_lab(),
        harness=True,
    )
    registry = wiring.registry
    for tool_id in (
        "workspace.write_file",
        "workspace.read_file",
        "memory.read",
        "issues.create_issue",
    ):
        assert registry.has(tool_id), tool_id


def test_harness_lab_mcp_export_matches_registry_size() -> None:
    from intergrax.tools.exporters.mcp import to_mcp_tools

    wiring = wire_lab_tools(
        integration_profile=IntegrationProfile.harness_lab(),
        harness=True,
    )
    exported = to_mcp_tools(wiring.registry)
    assert len(exported) == len(wiring.registry.tool_ids())
