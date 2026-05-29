# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.interactions.adapters.chained_adapter import ChainedInteractionAdapter
from intergrax.runtime.interactions.adapters.lab_json_adapter import LabJsonInteractionAdapter
from intergrax.integrations.providers.slack.adapter import SlackInteractionAdapter
from intergrax.runtime.interactions.adapters.teams_activity_adapter import TeamsActivityInteractionAdapter
from intergrax.runtime.interactions.factory import (
    InteractionSurface,
    create_interaction_adapter,
    intake_payload_to_task,
    resolve_interaction_settings,
)
from intergrax.runtime.interactions.metadata_keys import (
    INTERACTION_CHANNEL_KEY,
    INTERACTION_COMMAND_KEY,
    INTERACTION_RESPONSE_URL_KEY,
)
from intergrax.runtime.interactions.parsers.slash_command import parse_slash_command_text


@pytest.mark.unit
@pytest.mark.gate
def test_parse_slash_command_text():
    assert parse_slash_command_text("echo.basic hello world") == ("echo.basic", "hello world")
    assert parse_slash_command_text("/research.pipeline summarize") == (
        "research.pipeline",
        "summarize",
    )
    assert parse_slash_command_text("plain message") == (None, "plain message")
    assert parse_slash_command_text("") == (None, "")


@pytest.mark.unit
@pytest.mark.gate
def test_lab_json_adapter_to_task():
    adapter = LabJsonInteractionAdapter()
    payload = {
        "tenant_id": "t1",
        "user_id": "u1",
        "message": "run smoke test",
        "capability": "echo.basic",
        "metadata": {"source": "notebook"},
    }
    assert adapter.can_handle(payload)
    task = adapter.to_task(payload, tenant_id="fallback")
    assert task.tenant_id == "t1"
    assert task.user_id == "u1"
    assert task.message == "run smoke test"
    assert task.context.capability == "echo.basic"
    assert task.metadata[INTERACTION_CHANNEL_KEY] == "lab"


@pytest.mark.unit
@pytest.mark.gate
def test_slash_command_adapter_slack_payload_to_task():
    adapter = SlackInteractionAdapter()
    payload = {
        "command": "/intergrax",
        "text": "echo.basic hello from slack",
        "user_id": "U123",
        "team_id": "T456",
        "trigger_id": "trigger_1",
        "response_url": "https://hooks.slack.com/commands/1",
    }
    assert adapter.can_handle(payload)
    task = adapter.to_task(payload, tenant_id="default_tenant")
    assert task.tenant_id == "T456"
    assert task.user_id == "U123"
    assert task.context.capability == "echo.basic"
    assert task.message == "hello from slack"
    assert task.metadata[INTERACTION_CHANNEL_KEY] == "slack"
    assert task.metadata[INTERACTION_COMMAND_KEY] == "/intergrax"
    assert task.metadata[INTERACTION_RESPONSE_URL_KEY] == payload["response_url"]


@pytest.mark.unit
@pytest.mark.gate
def test_chained_adapter_prefers_slack():
    adapter = ChainedInteractionAdapter(
        [SlackInteractionAdapter(), LabJsonInteractionAdapter()]
    )
    task = adapter.to_task(
        {"command": "/x", "text": "cap.a do thing", "user_id": "u1", "team_id": "t1"},
        tenant_id="t1",
    )
    assert task.context.capability == "cap.a"
    assert task.message == "do thing"


@pytest.mark.unit
@pytest.mark.gate
def test_create_interaction_adapter_surfaces():
    assert isinstance(
        create_interaction_adapter(resolve_interaction_settings(surface="lab")),
        LabJsonInteractionAdapter,
    )
    assert isinstance(
        create_interaction_adapter(resolve_interaction_settings(surface="slash_command")),
        SlackInteractionAdapter,
    )
    assert isinstance(
        create_interaction_adapter(resolve_interaction_settings(surface="slack")),
        SlackInteractionAdapter,
    )
    assert isinstance(
        create_interaction_adapter(resolve_interaction_settings(surface="teams")),
        TeamsActivityInteractionAdapter,
    )
    auto = create_interaction_adapter(resolve_interaction_settings(surface="auto"))
    assert isinstance(auto, ChainedInteractionAdapter)


@pytest.mark.unit
@pytest.mark.gate
def test_intake_payload_to_task_auto():
    task = intake_payload_to_task(
        {
            "message": "direct lab intake",
            "capability": "echo.basic",
            "user_id": "u9",
            "tenant_id": "t9",
        },
        tenant_id="ignored_when_present",
    )
    assert task.tenant_id == "t9"
    assert task.context.capability == "echo.basic"
