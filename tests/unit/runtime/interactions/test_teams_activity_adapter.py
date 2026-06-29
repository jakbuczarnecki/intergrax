# © Artur Czarnecki. All rights reserved.

import base64
import hashlib
import hmac
import json

import pytest

from intergrax.integrations.providers.notification_channel.teams.adapter import _TeamsInteractionAdapter
from intergrax.runtime.interactions.factory import create_interaction_adapter, resolve_interaction_settings
from intergrax.runtime.interactions.metadata_keys import INTERACTION_CHANNEL_KEY
from intergrax.runtime.interactions.parsers.teams_activity import (
    parse_teams_activity_text,
    strip_teams_mentions,
)
from intergrax.runtime.interactions.verification.factory import create_inbound_verifier, resolve_inbound_verifier_settings
from intergrax.runtime.interactions.verification.teams_signature import TeamsSignatureVerifier


def _teams_activity_payload(*, text: str) -> dict:
    return {
        "type": "message",
        "id": "activity_1",
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


@pytest.mark.unit
@pytest.mark.gate
def test_strip_teams_mentions():
    text = "<at>Intergrax</at> echo.basic hello teams"
    entities = [{"type": "mention", "text": "<at>Intergrax</at>"}]
    assert strip_teams_mentions(text, entities) == "echo.basic hello teams"


@pytest.mark.unit
@pytest.mark.gate
def test_parse_teams_activity_text():
    capability, message = parse_teams_activity_text(
        "<at>Intergrax</at> echo.basic hello teams",
        entities=[{"type": "mention", "text": "<at>Intergrax</at>"}],
    )
    assert capability == "echo.basic"
    assert message == "hello teams"


@pytest.mark.unit
@pytest.mark.gate
def test_teams_activity_adapter_to_task():
    adapter = _TeamsInteractionAdapter()
    payload = _teams_activity_payload(text="<at>Intergrax</at> echo.basic hello teams")
    assert adapter.can_handle(payload)
    task = adapter.to_task(payload, tenant_id="fallback")
    assert task.tenant_id == "tenant-abc"
    assert task.user_id == "aad-user-1"
    assert task.context.capability == "echo.basic"
    assert task.message == "hello teams"
    assert task.metadata[INTERACTION_CHANNEL_KEY] == "teams"
    assert task.metadata["interaction_service_url"] == payload["serviceUrl"]


@pytest.mark.unit
@pytest.mark.gate
def test_create_interaction_adapter_teams_surface():
    adapter = create_interaction_adapter(resolve_interaction_settings(surface="teams"))
    assert isinstance(adapter, _TeamsInteractionAdapter)


@pytest.mark.unit
@pytest.mark.gate
def test_teams_signature_verifier_disabled_by_default():
    verifier = TeamsSignatureVerifier(security_token="secret", enabled=False)
    verifier.verify(headers={}, body=b"{}")


@pytest.mark.unit
@pytest.mark.gate
def test_teams_signature_verifier_rejects_invalid_signature():
    verifier = TeamsSignatureVerifier(security_token="secret", enabled=True)
    with pytest.raises(ValueError, match="invalid Teams signature"):
        verifier.verify(headers={"Authorization": "bad"}, body=b'{"text":"x"}')


@pytest.mark.unit
@pytest.mark.gate
def test_teams_signature_verifier_accepts_valid_hmac():
    token = "teams_security_token"
    body = b'{"text":"echo.basic hello"}'
    digest = base64.b64encode(
        hmac.new(token.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")
    verifier = TeamsSignatureVerifier(security_token=token, enabled=True)
    verifier.verify(headers={"Authorization": digest}, body=body)


@pytest.mark.unit
@pytest.mark.gate
def test_create_inbound_verifier_teams_mode():
    verifier = create_inbound_verifier(
        resolve_inbound_verifier_settings(
            mode="teams",
            teams_security_token="secret",
            teams_verify_enabled=False,
        )
    )
    assert isinstance(verifier, TeamsSignatureVerifier)
    assert verifier.enabled is False
