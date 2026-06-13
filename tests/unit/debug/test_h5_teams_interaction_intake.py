# © Artur Czarnecki. All rights reserved.

import base64
import hashlib
import hmac
import json

import pytest

pytestmark = pytest.mark.no_ci
from fastapi.testclient import TestClient

from intergrax.debug.app import create_debug_app
from intergrax.runtime.interactions.verification.teams_signature import TeamsSignatureVerifier
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent


def _echo_stub() -> UaepPipelineStubAgent:
    return UaepPipelineStubAgent(
        agent_id="echo",
        capability="echo.basic",
        prefix="echo",
        description="echo stub for Teams interaction intake",
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
    registry.register(_echo_stub())
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
    registry.register(_echo_stub())
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
    registry.register(_echo_stub())
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
