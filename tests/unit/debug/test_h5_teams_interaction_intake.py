# © Artur Czarnecki. All rights reserved.

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.debug.app import create_debug_app
from intergrax.runtime.interactions.verification.teams_signature import TeamsSignatureVerifier
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _EchoPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        answer = f"echo: {state.request.message}"
        state.raw_answer = answer
        state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer)
        return state.runtime_answer


class _EchoStubAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="echo",
            name="Echo",
            description="echo stub for Teams interaction intake",
            capabilities=["echo.basic"],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability == "echo.basic":
            return CapabilityMatchResult(
                matched=True,
                agent_id="echo",
                matched_capabilities=["echo.basic"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="echo"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = _EchoPipeline()
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


def _teams_activity_payload(*, text: str) -> dict:
    return {
        "type": "message",
        "id": "activity_teams_1",
        "timestamp": "2026-05-27T10:00:00.000Z",
        "serviceUrl": "https://smba.trafficmanager.net/teams/",
        "channelId": "msteams",
        "from": {"id": "29:user1", "name": "Jane Doe", "aadObjectId": "aad-user-1"},
        "conversation": {"id": "conv1", "tenantId": "tenant-abc"},
        "text": text,
        "entities": [
            {
                "type": "mention",
                "text": "<at>Intergrax</at>",
                "mentioned": {"id": "28:bot", "name": "Intergrax"},
            }
        ],
        "channelData": {"teamsTeamId": "team-xyz"},
    }


def _signed_teams_body(*, token: str, payload: dict) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload).encode("utf-8")
    digest = base64.b64encode(
        hmac.new(token.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": digest,
    }
    return body, headers


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_teams_activity_json():
    registry = AgentRegistry()
    registry.register(_EchoStubAgent())
    app = create_debug_app(registry=registry)
    payload = _teams_activity_payload(text="<at>Intergrax</at> echo.basic hello teams")
    with TestClient(app) as client:
        response = client.post(
            "/debug/interactions/intake",
            params={"tenant": "fallback", "execute": "false"},
            json=payload,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "echo.basic"
    assert body["message"] == "hello teams"
    assert body["interaction_channel"] == "teams"
    assert body["tenant_id"] == "tenant-abc"
    assert body["executed"] is False


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_teams_execute():
    registry = AgentRegistry()
    registry.register(_EchoStubAgent())
    app = create_debug_app(registry=registry)
    payload = _teams_activity_payload(text="<at>Intergrax</at> echo.basic run via teams")
    with TestClient(app) as client:
        response = client.post(
            "/debug/interactions/intake",
            params={"execute": "true"},
            json=payload,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is True
    assert body["state"] == TaskState.COMPLETED.value
    assert "run via teams" in (body["answer"] or "")


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_with_teams_verifier():
    registry = AgentRegistry()
    registry.register(_EchoStubAgent())
    token = "teams_test_security_token"
    verifier = TeamsSignatureVerifier(security_token=token, enabled=True)
    from intergrax.debug.interaction_service import DebugInteractionIntakeService
    from intergrax.runtime.nexus.nexus_loop import NexusLoop

    loop = NexusLoop(registry)
    service = DebugInteractionIntakeService(nexus_loop=loop, verifier=verifier)
    app = create_debug_app(registry=registry, interaction_service=service, nexus_loop=loop)

    payload = _teams_activity_payload(text="<at>Intergrax</at> echo.basic signed teams request")
    body, headers = _signed_teams_body(token=token, payload=payload)
    with TestClient(app) as client:
        ok = client.post(
            "/debug/interactions/intake",
            params={"execute": "true"},
            content=body,
            headers=headers,
        )
        bad = client.post(
            "/debug/interactions/intake",
            content=body,
            headers={"Content-Type": "application/json", "Authorization": "bad"},
        )
    assert ok.status_code == 200
    assert ok.json()["state"] == TaskState.COMPLETED.value
    assert bad.status_code == 401
