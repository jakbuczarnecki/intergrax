# © Artur Czarnecki. All rights reserved.

"""Tests for integration-driven Tier-1 tool profile extension (M-P6-WIRE)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.integration_tool_profile import (
    extend_tool_profile_for_integration,
    integration_category_configured,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.security_scanner.trivy.bundle import create_trivy_security_scanner
from intergrax.integrations.registry.presets import harness_security_stack, harness_sandbox_stack
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.security.service import SECURITY_SCAN_TOOL_ID
from intergrax.tools.providers.notify.service import NOTIFY_SEND_BATCH_TOOL_ID, NOTIFY_SCHEDULE_TOOL_ID
from intergrax.tools.providers.storage.service import STORAGE_EXISTS_TOOL_ID
from intergrax.tools.providers.records.service import RECORDS_COUNT_TOOL_ID
from intergrax.tools.providers.message_bus.service import MESSAGE_BUS_PURGE_COMPLETED_TOOL_ID
from intergrax.tools.providers.workflow.service import (
    WORKFLOW_CANCEL_RUN_TOOL_ID,
    WORKFLOW_LIST_RUNS_TOOL_ID,
    WORKFLOW_TRIGGER_TOOL_ID,
)
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeScanner:
    def scan_image(self, image_ref: str):
        return {"target": image_ref, "status": "completed", "findings": []}

    def scan_repo(self, repo_path: str):
        return {"target": repo_path, "status": "completed", "findings": []}

    def health(self) -> bool:
        return True


def test_integration_category_configured_for_instance_binding() -> None:
    profile = IntegrationProfile(security_scanner=create_trivy_security_scanner(client=_FakeScanner()))
    assert integration_category_configured(profile, IntegrationCategory.SECURITY_SCANNER) is True
    assert profile.slug_for_category(IntegrationCategory.SECURITY_SCANNER) is None


def test_extend_tool_profile_adds_security_and_workflow_tools() -> None:
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations(override=True)
    profile = harness_security_stack()
    tool_profile = extend_tool_profile_for_integration(ToolProfile(enabled=["rag.retrieve"]), profile)
    assert SECURITY_SCAN_TOOL_ID in tool_profile.enabled


def test_extend_tool_profile_adds_sandbox_exec_for_host() -> None:
    from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations(override=True)

    class _FakeHost:
        def create_session(self) -> dict[str, str]:
            return {"session_id": "s1", "status": "running"}

        def exec(self, session_id: str, command: str) -> dict[str, str]:
            return {"exit_code": 0, "stdout": command, "stderr": ""}

        def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> dict[str, str]:
            return {"artifact_id": remote_name, "uri": "hosted://x"}

        def health(self) -> bool:
            return True

    integration = IntegrationProfile(sandbox_host=create_e2b_sandbox_host(client=_FakeHost()))
    tool_profile = extend_tool_profile_for_integration(ToolProfile(enabled=[]), integration)
    assert "sandbox.exec" in tool_profile.enabled


def test_harness_sandbox_stack_preset_enables_sandbox_tool_id() -> None:
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations(override=True)
    profile = harness_sandbox_stack()
    tool_profile = extend_tool_profile_for_integration(ToolProfile(enabled=[]), profile)
    assert "sandbox.exec" in tool_profile.enabled


def test_workflow_tools_enabled_for_orchestrator_slug() -> None:
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations(override=True)
    profile = IntegrationProfile(workflow_orchestrator="prefect")
    tool_profile = extend_tool_profile_for_integration(ToolProfile(enabled=[]), profile)
    assert WORKFLOW_TRIGGER_TOOL_ID in tool_profile.enabled
    assert WORKFLOW_LIST_RUNS_TOOL_ID in tool_profile.enabled
    assert WORKFLOW_CANCEL_RUN_TOOL_ID in tool_profile.enabled


def test_notify_batch_enabled_for_notification_channel() -> None:
    from intergrax.integrations.registry.bootstrap import register_default_integrations
    from intergrax.integrations.registry.catalog_manifests import LOG

    register_default_integrations(override=True)
    profile = IntegrationProfile(notification_channel=LOG)
    tool_profile = extend_tool_profile_for_integration(ToolProfile(enabled=[]), profile)
    assert NOTIFY_SEND_BATCH_TOOL_ID in tool_profile.enabled
    assert NOTIFY_SCHEDULE_TOOL_ID in tool_profile.enabled


def test_t10_integration_tools_enabled_for_matching_categories() -> None:
    profile = IntegrationProfile(
        object_storage=object(),
        document_store=object(),
        message_bus=object(),
    )
    tool_profile = extend_tool_profile_for_integration(ToolProfile(enabled=[]), profile)
    assert STORAGE_EXISTS_TOOL_ID in tool_profile.enabled
    assert RECORDS_COUNT_TOOL_ID in tool_profile.enabled
    assert MESSAGE_BUS_PURGE_COMPLETED_TOOL_ID in tool_profile.enabled
