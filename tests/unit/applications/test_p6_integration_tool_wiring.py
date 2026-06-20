# © Artur Czarnecki. All rights reserved.

"""Tests for P6 integration tool wiring, security/workflow tools, and sandbox bridge."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.applications._shared.harness_auth import is_harness_identity_token_valid
from intergrax.applications._shared.identity_wiring import resolve_identity_provider_backend
from intergrax.applications._shared.integration_tool_wiring import wire_integration_tool_context
from intergrax.applications._shared.sandbox_host_wiring import resolve_hosted_sandbox_session
from intergrax.integrations._shared.speech_integration_bridge import IntegrationSpeechAdapter
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.presets import harness_security_stack
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.sandbox.hosted_session import HostedSandboxSession
from intergrax.speech_adapters.contracts.io import SpeechSynthesizeInput
from intergrax.tools.providers.security.service import SECURITY_SCAN_TOOL_ID, security_scan
from intergrax.tools.providers.security.contracts import SecurityScanInput
from intergrax.tools.providers.speech.backends import SPEECH_BACKEND_EXTRA_KEY
from intergrax.tools.providers.workflow.service import WORKFLOW_TRIGGER_TOOL_ID, workflow_trigger
from intergrax.tools.providers.workflow.contracts import WorkflowTriggerInput
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, list_catalog_tool_ids
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    clear_catalog()
    clear_tool_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    yield
    clear_catalog()
    clear_tool_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()


class _FakeScanner:
    def scan_image(self, image_ref: str) -> dict[str, Any]:
        return {"target": image_ref, "status": "completed", "findings": []}

    def scan_repo(self, repo_path: str) -> dict[str, Any]:
        return {"target": repo_path, "status": "completed", "findings": []}

    def health(self) -> bool:
        return True


class _FakeSandboxHost:
    def create_session(self) -> dict[str, str]:
        return {"session_id": "remote-1", "status": "running"}

    def exec(self, session_id: str, command: str) -> dict[str, Any]:
        return {"exit_code": 0, "stdout": command, "stderr": ""}

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> dict[str, str]:
        return {"artifact_id": remote_name, "uri": f"hosted://{session_id}/{remote_name}"}

    def health(self) -> bool:
        return True


class _FakeIdentity:
    def verify_token(self, token: str) -> IdentityUser:
        if token == "valid-token":
            return IdentityUser(user_id="user-1", email="user@example.com")
        raise ValueError("invalid token")

    def userinfo(self, token: str) -> IdentityUser:
        return self.verify_token(token)

    def list_tenants(self, *, limit: int) -> list:
        return []

    def health(self) -> bool:
        return True


class _FakeSpeech:
    def synthesize(self, text: str, *, voice_id: str = "default") -> dict[str, Any]:
        return {"audio_uri": "speech://audio/1", "character_count": len(text)}

    def transcribe(self, audio_uri: str) -> dict[str, str]:
        return {"transcript": audio_uri, "duration_ms": 100}

    def health(self) -> bool:
        return True


class _FakeWorkflow:
    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str]) -> dict[str, str]:
        return {"run_id": workflow_id, "status": "pending", "url": "https://wf/run/1"}

    def poll_status(self, run_id: str) -> dict[str, str]:
        return {"run_id": run_id, "status": "success", "conclusion": "success"}

    def fetch_logs(self, run_id: str, *, tail_lines: int) -> str:
        return f"log:{run_id}"

    def health(self) -> bool:
        return True


def test_security_and_workflow_tools_registered() -> None:
    register_default_tools()
    tool_ids = set(list_catalog_tool_ids())
    assert SECURITY_SCAN_TOOL_ID in tool_ids
    assert WORKFLOW_TRIGGER_TOOL_ID in tool_ids


def test_security_scan_tool_uses_scanner_backend() -> None:
    from intergrax.integrations.providers.security_scanner.trivy.bundle import create_trivy_security_scanner

    ctx = ToolWiringContext(security_scanner=create_trivy_security_scanner(client=_FakeScanner()))
    result = security_scan(ctx, SecurityScanInput(target=".", scan_type="repo"))
    assert result.status == "completed"


def test_workflow_trigger_tool_uses_orchestrator_backend() -> None:
    from intergrax.integrations.providers.workflow_orchestrator.prefect.bundle import create_prefect_workflow_orchestrator

    ctx = ToolWiringContext(workflow_orchestrator=create_prefect_workflow_orchestrator(client=_FakeWorkflow()))
    result = workflow_trigger(ctx, WorkflowTriggerInput(workflow_id="eval-refresh"))
    assert result.run_id == "eval-refresh"


def test_hosted_sandbox_session_exec_echo() -> None:
    from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host

    backend = create_e2b_sandbox_host(client=_FakeSandboxHost())
    session = HostedSandboxSession.open(backend, tenant_id="t1", task_id="task1")
    result = session.execute("echo", {"message": "hello"})
    assert result.success is True
    assert result.output is not None


def test_resolve_hosted_sandbox_from_integration_instance() -> None:
    from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host

    profile = IntegrationProfile(
        sandbox_host=create_e2b_sandbox_host(client=_FakeSandboxHost()),
    )
    session = resolve_hosted_sandbox_session(profile, tenant_id="t1", task_id="task1")
    assert session is not None
    result = session.execute("echo", {"message": "hi"})
    assert result.success is True


def test_integration_speech_adapter_bridge() -> None:
    from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider

    backend = create_deepgram_speech_provider(client=_FakeSpeech())
    adapter = IntegrationSpeechAdapter(backend, provider_slug="deepgram")
    output = adapter.synthesize(SpeechSynthesizeInput(text="hello"))
    assert output.audio_uri
    assert adapter.provider_slug == "deepgram"


def test_wire_integration_tool_context_speech_extra() -> None:
    register_default_integrations()
    from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider

    profile = IntegrationProfile(
        speech_provider=create_deepgram_speech_provider(client=_FakeSpeech()),
    )
    ctx = wire_integration_tool_context(ToolWiringContext(), profile)
    adapter = ctx.extras.get(SPEECH_BACKEND_EXTRA_KEY)
    assert isinstance(adapter, IntegrationSpeechAdapter)
    assert adapter.provider_slug == "deepgram"


def test_wire_modality_extras_skips_speech_backend_when_integration_provider_set() -> None:
    from intergrax.applications._shared.modality_wiring import wire_modality_extras
    from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider

    backend = create_deepgram_speech_provider(client=_FakeSpeech())
    ctx = ToolWiringContext(speech_provider=backend)
    wire_modality_extras(ctx)
    assert SPEECH_BACKEND_EXTRA_KEY not in ctx.extras


def test_identity_provider_resolution_and_token_validation() -> None:
    register_default_integrations()
    from intergrax.integrations.providers.identity_provider.keycloak.bundle import create_keycloak_identity_provider

    profile = IntegrationProfile(
        identity_provider=create_keycloak_identity_provider(client=_FakeIdentity()),
    )
    backend = resolve_identity_provider_backend(profile)
    assert backend is not None
    assert is_harness_identity_token_valid(authorization="Bearer valid-token", identity_provider=backend) is True
    assert is_harness_identity_token_valid(authorization="Bearer bad", identity_provider=backend) is False


def test_harness_security_stack_preset_for_promote_gate() -> None:
    register_default_integrations()
    profile = harness_security_stack()
    assert profile.slug_for_category(IntegrationCategory.SECURITY_SCANNER.value) == "trivy"
    assert "semgrep" in profile.options
