# © Artur Czarnecki. All rights reserved.

"""Tests for LKW message_bus tool wiring guardrails (LKW.4B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from intergrax.tools.providers.message_bus.bundle import MESSAGE_BUS_TOOL_IDS
from intergrax.tools.providers.message_bus.service import MESSAGE_BUS_ENQUEUE_TOOL_ID
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.tool_wiring import wire_local_workspace_tools

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeMessageBus(TaskQueue):
    def enqueue(self, request: TaskRequest) -> TaskHandle:
        return TaskHandle(task_id="task-1", provider="fake", tenant_id=request.tenant_id)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        return TaskStatus.PENDING

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        return None


def test_default_profile_does_not_expose_message_bus_tools() -> None:
    settings = LocalWorkspaceBackendSettings(enable_rag=False, enable_rag_ingest=False)
    wiring = wire_local_workspace_tools(settings=settings, integration_profile=IntegrationProfile())

    enabled = set(wiring.profile.enabled)
    assert not any(tool_id in enabled for tool_id in MESSAGE_BUS_TOOL_IDS)
    for tool_id in MESSAGE_BUS_TOOL_IDS:
        assert not wiring.registry.has(tool_id)
    assert wiring.wiring_context.message_bus is None


def test_extra_enabled_tool_ids_cannot_expose_message_bus_without_integration() -> None:
    settings = LocalWorkspaceBackendSettings(
        extra_enabled_tool_ids=(MESSAGE_BUS_ENQUEUE_TOOL_ID,),
        enable_rag=False,
        enable_rag_ingest=False,
    )
    wiring = wire_local_workspace_tools(settings=settings, integration_profile=IntegrationProfile())

    assert MESSAGE_BUS_ENQUEUE_TOOL_ID not in wiring.profile.enabled
    assert wiring.registry.has(MESSAGE_BUS_ENQUEUE_TOOL_ID) is False


def test_message_bus_integration_exposes_all_message_bus_tools() -> None:
    settings = LocalWorkspaceBackendSettings(enable_rag=False, enable_rag_ingest=False)
    bus = _FakeMessageBus()
    profile = IntegrationProfile(message_bus=IntegrationBinding.from_instance(bus))
    wiring = wire_local_workspace_tools(settings=settings, integration_profile=profile)

    enabled = wiring.profile.enabled
    for tool_id in MESSAGE_BUS_TOOL_IDS:
        assert tool_id in enabled
        assert wiring.registry.has(tool_id)
    assert wiring.wiring_context.message_bus is bus
    assert len([tool_id for tool_id in enabled if tool_id in MESSAGE_BUS_TOOL_IDS]) == len(
        MESSAGE_BUS_TOOL_IDS
    )


def test_base_tools_remain_enabled_with_message_bus_guardrails() -> None:
    settings = LocalWorkspaceBackendSettings(enable_rag=False, enable_rag_ingest=False)
    wiring = wire_local_workspace_tools(settings=settings, integration_profile=IntegrationProfile())

    enabled = set(wiring.profile.enabled)
    assert "workspace.search" in enabled
    assert "workspace.read_file" in enabled or "rag.list_collections" in enabled


def test_tool_wiring_receives_web_url_staging_read_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))

    settings = LocalWorkspaceBackendSettings.from_env()
    wiring = wire_local_workspace_tools(settings=settings, integration_profile=IntegrationProfile())

    assert settings.web_url_staging_dir in wiring.wiring_context.read_allowlist_roots
    assert settings.managed_upload_staging_dir in wiring.wiring_context.read_allowlist_roots
