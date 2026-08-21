# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from legal_application.serving.fastapi_router import DefaultLegalAgentService, LegalAgentServingConfig
from intergrax.agents.runtime_request_bridge import runtime_request_to_agent_run
from intergrax.applications._shared.harness_auth import (
    HarnessAuthState,
    resolve_harness_authenticated_principal,
)
from intergrax.applications._shared.identity_wiring import wire_application_identity
from intergrax.applications._shared.harness_principal import (
    HarnessAuthenticatedPrincipal,
    harness_principal_to_request_identity,
    reject_identity_assertion_conflicts,
)
from intergrax.applications.contracts.environment_profile import IdentityProfile
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.request_identity_spine import (
    api_key_service_request_identity,
    assert_untrusted_metadata_identity_compatible,
    identity_user_to_request_identity,
    request_identity_to_actor_identity,
)
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.fastapi_core.context import RequestContext
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.interactions.actor_resolution import resolve_actor_from_envelope
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from legal_application.serving.schemas import LegalChatRequestV1
from research_application.serving.fastapi_router import ResearchRunService
from research_application.serving.schemas import ResearchRunRequestV1

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _BridgeContract:
    id = "bridge-agent"


class _FakeIdentityProvider:
    def __init__(
        self,
        *,
        valid_token: str,
        user_id: str = "U1",
        tenant_id: str = "A",
    ) -> None:
        self._valid_token = valid_token
        self._user_id = user_id
        self._tenant_id = tenant_id

    def verify_token(self, token: str) -> IdentityUser:
        if token != self._valid_token:
            raise ValueError("invalid token")
        return IdentityUser(
            user_id=self._user_id,
            tenant_id=self._tenant_id,
        )

    def userinfo(self, token: str) -> IdentityUser:
        return self.verify_token(token)

    def list_tenants(self, *, limit: int = 50) -> tuple[()]:
        return ()


def _canonical_user_identity() -> RequestIdentity:
    return RequestIdentity(
        tenant_id="A",
        user_id="U1",
        principal_type=PrincipalType.USER,
        auth_subject="U1",
    )


def _roundtrip_task_and_run_ids() -> tuple[str, str]:
    return (
        "task_a3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "run_a3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )


def test_idt_r1_t1_runtime_request_canonical_identity_round_trip() -> None:
    canonical = RequestIdentity(
        tenant_id="A",
        user_id="U1",
        principal_type=PrincipalType.USER,
        auth_subject="S1",
    )
    task_id, run_id = _roundtrip_task_and_run_ids()
    request = RuntimeRequest(
        agent_id="a",
        user_id="U1",
        session_id="s",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="A",
        canonical_identity=canonical,
    )
    restored = RuntimeRequest.from_envelope(
        request.to_envelope(),
        task_id=task_id,
        run_id=run_id,
    )
    assert restored.canonical_identity == canonical


def test_idt_r1_t2_metadata_attack_rejected_after_envelope_round_trip() -> None:
    canonical = _canonical_user_identity()
    task_id, run_id = _roundtrip_task_and_run_ids()
    request = RuntimeRequest(
        agent_id="a",
        user_id="U1",
        session_id="s",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="A",
        canonical_identity=canonical,
        metadata={"tenant_id": "B", "user_id": "U2"},
    )
    restored = RuntimeRequest.from_envelope(
        request.to_envelope(),
        task_id=task_id,
        run_id=run_id,
    )
    assert restored.canonical_identity == canonical
    with pytest.raises(ValueError, match="metadata tenant_id conflicts"):
        runtime_request_to_agent_run(restored, contract=_BridgeContract())


def test_idt_r1_t3_service_principal_type_survives_envelope_round_trip() -> None:
    canonical = RequestIdentity(
        tenant_id="A",
        user_id="svc-ops",
        principal_type=PrincipalType.SERVICE,
        auth_subject="svc-ops",
    )
    task_id, run_id = _roundtrip_task_and_run_ids()
    request = RuntimeRequest(
        agent_id="a",
        user_id="svc-ops",
        session_id="s",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="A",
        canonical_identity=canonical,
        metadata={"principal_type": "user"},
    )
    restored = RuntimeRequest.from_envelope(
        request.to_envelope(),
        task_id=task_id,
        run_id=run_id,
    )
    assert restored.canonical_identity is not None
    assert restored.canonical_identity.principal_type is PrincipalType.SERVICE
    with pytest.raises(ValueError, match="metadata principal_type conflicts"):
        runtime_request_to_agent_run(restored, contract=_BridgeContract())


def test_idt_r1_t4_auth_subject_survives_envelope_round_trip() -> None:
    canonical = RequestIdentity(
        tenant_id="A",
        user_id="U1",
        principal_type=PrincipalType.USER,
        auth_subject="S1",
    )
    task_id, run_id = _roundtrip_task_and_run_ids()
    request = RuntimeRequest(
        agent_id="a",
        user_id="U1",
        session_id="s",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="A",
        canonical_identity=canonical,
    )
    restored = RuntimeRequest.from_envelope(
        request.to_envelope(),
        task_id=task_id,
        run_id=run_id,
    )
    assert restored.canonical_identity is not None
    assert restored.canonical_identity.auth_subject == "S1"


def test_idt_r1_t5_envelope_carries_no_raw_credential() -> None:
    canonical = _canonical_user_identity()
    task_id, run_id = _roundtrip_task_and_run_ids()
    request = RuntimeRequest(
        agent_id="a",
        user_id="U1",
        session_id="s",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="A",
        canonical_identity=canonical,
    )
    envelope = request.to_envelope()
    assert envelope.canonical_identity == canonical
    assert "canonical_identity" not in envelope.metadata
    identity_payload = envelope.canonical_identity.model_dump() if envelope.canonical_identity else {}
    assert set(identity_payload.keys()) == {
        "tenant_id",
        "user_id",
        "principal_type",
        "auth_subject",
    }


def test_idt_r1_t6_legacy_unauthenticated_envelope_round_trip() -> None:
    task_id, run_id = _roundtrip_task_and_run_ids()
    envelope = TaskEnvelope(tenant_id="legacy-t", user_id="legacy-u", message="go")
    restored = RuntimeRequest.from_envelope(envelope, task_id=task_id, run_id=run_id)
    assert restored.canonical_identity is None
    agent_run = runtime_request_to_agent_run(restored, contract=_BridgeContract())
    assert agent_run.identity.tenant_id == "legacy-t"
    assert agent_run.identity.user_id == "legacy-u"


def test_idt_r1_t7_unrelated_service_identity_not_borrowed_for_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-key")
    app = FastAPI()
    wire_application_identity(
        app,
        IdentityProfile(
            require_api_key=False,
            service_identities={
                "billing": "billing-service",
                "scheduler": "scheduler-service",
            },
        ),
    )
    assert app.state.harness_auth.api_key_principal_service_id == "harness-api-key"


def test_idt_r1_t8_explicit_harness_service_identity_used_for_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-key")
    app = FastAPI()
    wire_application_identity(
        app,
        IdentityProfile(
            require_api_key=False,
            service_identities={"harness": "platform-api"},
        ),
    )
    assert app.state.harness_auth.api_key_principal_service_id == "platform-api"


def test_idt_r1_task_envelope_canonical_identity_round_trip() -> None:
    canonical = _canonical_user_identity()
    envelope = TaskEnvelope(
        tenant_id="A",
        user_id="U1",
        message="go",
        agent_id="echo",
        canonical_identity=canonical,
    )
    task = Task.from_envelope(envelope)
    assert task.canonical_identity == canonical
    assert task.to_envelope().canonical_identity == canonical


def test_idt_a1_verified_user_cannot_change_tenant_by_body() -> None:
    canonical = _canonical_user_identity()
    with pytest.raises(HTTPException) as exc:
        reject_identity_assertion_conflicts(
            canonical=canonical,
            asserted_tenant_id="B",
            asserted_user_id="U1",
        )
    assert exc.value.status_code == 400
    assert "tenant_id" in exc.value.detail


def test_idt_a2_verified_user_cannot_change_user_id() -> None:
    canonical = _canonical_user_identity()
    with pytest.raises(HTTPException) as exc:
        reject_identity_assertion_conflicts(
            canonical=canonical,
            asserted_tenant_id="A",
            asserted_user_id="U2",
        )
    assert exc.value.status_code == 400
    assert "user_id" in exc.value.detail


def test_idt_a3_metadata_user_override_closed() -> None:
    canonical = _canonical_user_identity()
    runtime_request = RuntimeRequest(
        agent_id="a",
        user_id="U1",
        session_id="s",
        message="hello",
        task_id="task_a3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        run_id="run_a3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        tenant_id="A",
        canonical_identity=canonical,
        metadata={"user_id": "U2"},
    )
    with pytest.raises(ValueError, match="metadata user_id conflicts"):
        runtime_request_to_agent_run(runtime_request, contract=_BridgeContract())
    runtime_request.metadata["user_id"] = "U1"
    agent_run = runtime_request_to_agent_run(runtime_request, contract=_BridgeContract())
    assert agent_run.identity.user_id == "U1"


def test_idt_a4_principal_type_metadata_cannot_escalate() -> None:
    canonical = _canonical_user_identity()
    with pytest.raises(ValueError, match="metadata principal_type conflicts"):
        assert_untrusted_metadata_identity_compatible(
            canonical,
            {"principal_type": "service"},
        )


def test_idt_a5_auth_subject_metadata_cannot_redefine_principal() -> None:
    canonical = _canonical_user_identity()
    with pytest.raises(ValueError, match="metadata auth_subject conflicts"):
        assert_untrusted_metadata_identity_compatible(
            canonical,
            {"auth_subject": "S2"},
        )


def test_idt_a6_verified_identity_provider_principal_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(valid_token="token-a"),
        require_api_key=True,
    )

    @app.get("/principal")
    def principal_endpoint(
        resolved: HarnessAuthenticatedPrincipal | None = Depends(
            resolve_harness_authenticated_principal
        ),
    ) -> dict[str, str]:
        assert resolved is not None
        identity = harness_principal_to_request_identity(resolved)
        return {
            "tenant_id": identity.tenant_id,
            "user_id": identity.user_id or "",
            "auth_subject": identity.auth_subject or "",
        }

    client = TestClient(app)
    response = client.get("/principal", headers={"Authorization": "Bearer token-a"})
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"tenant_id": "A", "user_id": "U1", "auth_subject": "U1"}


def test_idt_a7_api_key_cannot_choose_arbitrary_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    monkeypatch.setenv("INTERGRAX_HARNESS_TENANT_ID", "host-tenant-a")
    identity = api_key_service_request_identity(
        tenant_id="host-tenant-a",
        service_id="harness-api-key",
    )
    assert identity.tenant_id == "host-tenant-a"
    assert identity.user_id == "harness-api-key"
    assert identity.principal_type is PrincipalType.SERVICE


@pytest.mark.asyncio
async def test_idt_a8_research_concrete_regression() -> None:
    runner = AsyncMock(spec=UnifiedTaskRunner)
    service = ResearchRunService(task_runner=runner)
    principal = HarnessAuthenticatedPrincipal(
        tenant_id="A",
        user_id="U1",
        principal_type=PrincipalType.USER,
        auth_subject="U1",
        auth_mode="identity_provider",
    )
    body = ResearchRunRequestV1(tenant_id="B", user_id="U2", message="summarize")
    with pytest.raises(HTTPException) as exc:
        await service.run_pipeline(body, authenticated_principal=principal)
    assert exc.value.status_code == 400
    runner.run_task.assert_not_called()


@pytest.mark.asyncio
async def test_idt_a9_matching_assertion_succeeds() -> None:
    runner = AsyncMock(spec=UnifiedTaskRunner)
    runner.run_task.return_value = MagicMock(
        task_id="task-1",
        run_id="run-1",
        state=MagicMock(value="completed"),
        answer="ok",
        metadata={},
    )
    service = ResearchRunService(task_runner=runner)
    principal = HarnessAuthenticatedPrincipal(
        tenant_id="A",
        user_id="U1",
        principal_type=PrincipalType.USER,
        auth_subject="U1",
        auth_mode="identity_provider",
    )
    body = ResearchRunRequestV1(tenant_id="A", user_id="U1", message="summarize")
    await service.run_pipeline(body, authenticated_principal=principal)
    runner.run_task.assert_awaited_once()
    task = runner.run_task.await_args.args[0]
    assert task.tenant_id == "A"
    assert task.user_id == "U1"


def test_idt_a10_actor_writer_reader_round_trip() -> None:
    envelope = TaskEnvelope(
        tenant_id="t1",
        user_id="u1",
        message="go",
    ).with_actor(actor_kind=ActorKind.SERVICE.value, actor_id="svc-ops")
    actor = resolve_actor_from_envelope(envelope)
    assert actor.kind is ActorKind.SERVICE
    assert actor.actor_id == "svc-ops"
    assert actor.tenant_id == "t1"


def test_idt_a11_request_identity_to_actor_projection() -> None:
    identity = _canonical_user_identity()
    actor = request_identity_to_actor_identity(identity)
    assert actor.kind is ActorKind.USER
    assert actor.actor_id == "U1"
    assert actor.tenant_id == "A"


def test_idt_a12_untrusted_metadata_does_not_create_authenticated_actor() -> None:
    identity = _canonical_user_identity()
    actor = request_identity_to_actor_identity(identity)
    assert actor.kind is ActorKind.USER
    assert actor.actor_id == "U1"
    assert actor.permission_scopes == ()


def test_idt_a_legal_positive_control_preserves_conflict_rejection() -> None:
    config = MagicMock(spec=LegalAgentServingConfig)
    config.identity_source = "context_only"
    service = DefaultLegalAgentService(config=config)
    body = LegalChatRequestV1(message="review", session_id="sess-1", tenant_id="B", user_id="U2")
    http_ctx = RequestContext(
        request_id="req-1",
        path="/v1/legal/chat",
        method="POST",
        tenant_id="A",
        user_id="U1",
        auth=None,
    )
    with pytest.raises(HTTPException) as tenant_exc:
        service._resolve_identity(body, http_ctx)
    assert tenant_exc.value.status_code == 400
    body_match = LegalChatRequestV1(message="review", session_id="sess-1", tenant_id="A", user_id="U1")
    tenant, user = service._resolve_identity(body_match, http_ctx)
    assert tenant == "A"
    assert user == "U1"


def test_idt_a_identity_user_mapping_is_deterministic() -> None:
    user = IdentityUser(user_id="U1", tenant_id="A")
    identity = identity_user_to_request_identity(user)
    assert identity.tenant_id == "A"
    assert identity.user_id == "U1"
    assert identity.principal_type is PrincipalType.USER
    assert identity.auth_subject == "U1"
