# © Artur Czarnecki. All rights reserved.

"""APP-ADOPTION-1: LKW host consumes canonical Tier-3 platform plugin evidence."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_CONTEXT,
    PLATFORM_PLUGIN_DOMAIN_MEMORY,
)
from intergrax.core.plugins.discovery import EP_CONTEXT, EP_MEMORY_STORES
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.wiring import build_local_workspace_host_composition

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.no_ci
def test_lkw_host_composition_exposes_platform_plugin_evidence() -> None:
    composition = build_local_workspace_host_composition(LocalWorkspaceBackendSettings.from_env())

    evidence = composition.platform_plugin_evidence
    memory_report = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_MEMORY)
    assert memory_report is not None
    assert memory_report.group == EP_MEMORY_STORES
    context_report = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_CONTEXT)
    assert context_report is not None
    assert context_report.group == EP_CONTEXT
    assert composition.registry.list_agent_ids()
