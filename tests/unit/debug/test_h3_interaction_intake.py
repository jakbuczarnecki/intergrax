# © Artur Czarnecki. All rights reserved.

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.debug.app import create_debug_app
from intergrax.runtime.interactions.http_intake import parse_inbound_http_body
from intergrax.runtime.interactions.verification.slack_signature import SlackSignatureVerifier
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
            description="echo stub for interaction intake",
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


def _signed_slack_body(*, secret: str, payload: dict) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload).encode("utf-8")
    timestamp = str(int(time.time()))
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    digest = hmac.new(
        secret.encode("utf-8"),
        basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": f"v0={digest}",
    }
    return body, headers


@pytest.mark.unit
@pytest.mark.gate
def test_parse_inbound_form_body():
    body = b"command=%2Fintergrax&text=echo.basic+hello&user_id=U1&team_id=T1"
    payload = parse_inbound_http_body(
        content_type="application/x-www-form-urlencoded",
        body=body,
    )
    assert payload["command"] == "/intergrax"
    assert payload["text"] == "echo.basic hello"
    assert payload["team_id"] == "T1"


@pytest.mark.unit
@pytest.mark.gate
def test_slack_signature_verifier_disabled_by_default():
    verifier = SlackSignatureVerifier(signing_secret="secret", enabled=False)
    verifier.verify(headers={}, body=b"{}")


@pytest.mark.unit
@pytest.mark.gate
def test_slack_signature_verifier_rejects_invalid_signature():
    verifier = SlackSignatureVerifier(signing_secret="secret", enabled=True)
    with pytest.raises(ValueError, match="invalid Slack signature"):
        verifier.verify(
            headers={
                "X-Slack-Request-Timestamp": str(int(time.time())),
                "X-Slack-Signature": "v0=deadbeef",
            },
            body=b'{"text":"x"}',
        )


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_json_only():
    registry = AgentRegistry()
    registry.register(_EchoStubAgent())
    app = create_debug_app(registry=registry)
    with TestClient(app) as client:
        response = client.post(
            "/debug/interactions/intake",
            params={"tenant": "t1", "execute": "false"},
            json={
                "command": "/intergrax",
                "text": "echo.basic hello lab",
                "user_id": "U1",
                "team_id": "T1",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "echo.basic"
    assert body["message"] == "hello lab"
    assert body["interaction_channel"] == "slack"
    assert body["executed"] is False


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_form_execute():
    registry = AgentRegistry()
    registry.register(_EchoStubAgent())
    app = create_debug_app(registry=registry)
    with TestClient(app) as client:
        response = client.post(
            "/debug/interactions/intake",
            params={"tenant": "fallback", "execute": "true"},
            data={
                "command": "/intergrax",
                "text": "echo.basic run via form",
                "user_id": "U9",
                "team_id": "T9",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is True
    assert body["state"] == TaskState.COMPLETED.value
    assert "run via form" in (body["answer"] or "")


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_not_configured():
    app = create_debug_app()
    with TestClient(app) as client:
        response = client.post(
            "/debug/interactions/intake",
            json={"message": "x", "capability": "echo.basic"},
        )
    assert response.status_code == 503


@pytest.mark.unit
@pytest.mark.gate
def test_debug_interaction_intake_with_verifier():
    registry = AgentRegistry()
    registry.register(_EchoStubAgent())
    secret = "test_signing_secret"
    verifier = SlackSignatureVerifier(signing_secret=secret, enabled=True)
    from intergrax.debug.interaction_service import DebugInteractionIntakeService
    from intergrax.runtime.nexus.nexus_loop import NexusLoop

    loop = NexusLoop(registry)
    service = DebugInteractionIntakeService(nexus_loop=loop, verifier=verifier)
    app = create_debug_app(registry=registry, interaction_service=service, nexus_loop=loop)

    payload = {
        "command": "/intergrax",
        "text": "echo.basic signed request",
        "user_id": "U1",
        "team_id": "T1",
    }
    body, headers = _signed_slack_body(secret=secret, payload=payload)
    with TestClient(app) as client:
        ok = client.post(
            "/debug/interactions/intake",
            params={"execute": "true", "tenant": "T1"},
            content=body,
            headers=headers,
        )
        bad = client.post(
            "/debug/interactions/intake",
            params={"tenant": "T1"},
            content=body,
            headers={"Content-Type": "application/json", "X-Slack-Signature": "v0=bad"},
        )
    assert ok.status_code == 200
    assert ok.json()["state"] == TaskState.COMPLETED.value
    assert bad.status_code == 401
