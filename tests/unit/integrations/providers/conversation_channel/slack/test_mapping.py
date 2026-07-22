# © Artur Czarnecki. All rights reserved.

"""Slack inbound mapping unit tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from intergrax.integrations.contracts.conversation_channel import ConversationEventKind
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    deterministic_block_action_event_id,
    map_block_actions,
    map_events_api_message,
)

pytestmark = pytest.mark.unit


def _dm_payload(**overrides: object) -> dict:
    payload: dict = {
        "event_id": "EvMESSAGE1",
        "team_id": "TTEAM1",
        "event": {
            "type": "message",
            "channel_type": "im",
            "channel": "DCHANNEL1",
            "user": "UUSER1",
            "text": "hello workspace",
            "ts": "1710000000.000100",
            "client_msg_id": "client-msg-should-not-be-event-id",
        },
    }
    for key, value in overrides.items():
        if key == "event" and isinstance(value, dict):
            payload["event"] = {**payload["event"], **value}
        else:
            payload[key] = value
    return payload


def _action_payload(**overrides: object) -> dict:
    payload: dict = {
        "type": "block_actions",
        "team": {"id": "TTEAM1"},
        "user": {"id": "UUSER1", "username": "alice"},
        "channel": {"id": "DCHANNEL1"},
        "container": {"message_ts": "1710000000.000200", "thread_ts": "1710000000.000100"},
        "message": {"ts": "1710000000.000200", "thread_ts": "1710000000.000100"},
        "actions": [
            {
                "type": "static_select",
                "action_id": "choose_workspace",
                "action_ts": "1710000001.000001",
                "selected_option": {"value": "ws-1", "text": {"type": "plain_text", "text": "WS"}},
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_valid_top_level_dm() -> None:
    event = map_events_api_message(_dm_payload())
    assert event is not None
    assert event.kind is ConversationEventKind.MESSAGE
    assert event.address.thread_id == "1710000000.000100"
    assert event.text == "hello workspace"


def test_valid_threaded_dm() -> None:
    event = map_events_api_message(_dm_payload(event={"thread_ts": "1710000000.000050"}))
    assert event is not None
    assert event.address.thread_id == "1710000000.000050"


def test_payload_event_id_is_event_identity() -> None:
    event = map_events_api_message(_dm_payload())
    assert event is not None
    assert event.event_id == "EvMESSAGE1"


def test_client_msg_id_does_not_become_event_id() -> None:
    event = map_events_api_message(_dm_payload())
    assert event is not None
    assert event.event_id != "client-msg-should-not-be-event-id"
    assert event.metadata.get("client_msg_id") == "client-msg-should-not-be-event-id"


def test_missing_event_id_returns_none() -> None:
    payload = _dm_payload()
    del payload["event_id"]
    assert map_events_api_message(payload) is None


def test_blank_event_id_returns_none() -> None:
    assert map_events_api_message(_dm_payload(event_id="  ")) is None


def test_missing_team_id_rejected() -> None:
    payload = _dm_payload()
    del payload["team_id"]
    assert map_events_api_message(payload) is None


def test_non_im_message_ignored() -> None:
    assert map_events_api_message(_dm_payload(event={"channel_type": "channel"})) is None


def test_bot_authored_message_ignored() -> None:
    assert map_events_api_message(_dm_payload(event={"bot_id": "B123"})) is None


def test_subtype_ignored() -> None:
    assert map_events_api_message(_dm_payload(event={"subtype": "message_changed"})) is None


def test_blank_text_rejected() -> None:
    assert map_events_api_message(_dm_payload(event={"text": "  "})) is None


def test_missing_user_rejected() -> None:
    payload = _dm_payload()
    del payload["event"]["user"]
    assert map_events_api_message(payload) is None


def test_missing_channel_rejected() -> None:
    payload = _dm_payload()
    del payload["event"]["channel"]
    assert map_events_api_message(payload) is None


def test_missing_message_timestamp_rejected() -> None:
    payload = _dm_payload()
    del payload["event"]["ts"]
    assert map_events_api_message(payload) is None


def test_top_level_thread_id_equals_event_ts() -> None:
    event = map_events_api_message(_dm_payload())
    assert event is not None
    assert event.address.thread_id == event.address.thread_id
    assert event.address.thread_id == "1710000000.000100"


def test_existing_thread_preserves_thread_ts() -> None:
    event = map_events_api_message(_dm_payload(event={"thread_ts": "1700000000.000001"}))
    assert event is not None
    assert event.address.thread_id == "1700000000.000001"


def test_invalid_timestamp_leaves_occurred_at_none() -> None:
    event = map_events_api_message(_dm_payload(event={"ts": "not-a-ts"}))
    assert event is not None
    assert event.occurred_at is None
    assert event.event_id == "EvMESSAGE1"
    assert event.address.thread_id == "not-a-ts"


def test_valid_block_actions_static_select() -> None:
    event = map_block_actions(_action_payload())
    assert event is not None
    assert event.kind is ConversationEventKind.ACTION
    assert event.address.installation_id == "TTEAM1"
    assert event.address.conversation_id == "DCHANNEL1"
    assert event.address.thread_id == "1710000000.000100"
    assert event.actor.actor_id == "UUSER1"
    assert event.actor.display_name == "alice"
    assert event.action is not None
    assert event.action.action_id == "choose_workspace"
    assert event.action.selected_value == "ws-1"


def test_action_missing_team_rejected() -> None:
    payload = _action_payload()
    del payload["team"]
    assert map_block_actions(payload) is None


def test_action_missing_user_rejected() -> None:
    payload = _action_payload()
    del payload["user"]
    assert map_block_actions(payload) is None


def test_action_missing_channel_rejected() -> None:
    payload = _action_payload()
    del payload["channel"]
    assert map_block_actions(payload) is None


def test_action_missing_selected_option_rejected() -> None:
    payload = _action_payload()
    payload["actions"][0].pop("selected_option")
    assert map_block_actions(payload) is None


def test_unsupported_action_type_ignored() -> None:
    payload = _action_payload()
    payload["actions"][0]["type"] = "button"
    assert map_block_actions(payload) is None


def test_multi_action_payload_ignored() -> None:
    payload = _action_payload()
    payload["actions"] = payload["actions"] + deepcopy(payload["actions"])
    assert map_block_actions(payload) is None


def test_action_event_id_deterministic() -> None:
    first = map_block_actions(_action_payload())
    second = map_block_actions(_action_payload())
    assert first is not None and second is not None
    assert first.event_id == second.event_id
    assert first.event_id.startswith("slack:block_action:v1:")


def test_different_selected_value_changes_event_id() -> None:
    base = map_block_actions(_action_payload())
    changed_payload = _action_payload()
    changed_payload["actions"][0]["selected_option"]["value"] = "ws-2"
    changed = map_block_actions(changed_payload)
    assert base is not None and changed is not None
    assert base.event_id != changed.event_id


def test_different_message_ts_changes_event_id() -> None:
    base = map_block_actions(_action_payload())
    changed_payload = _action_payload()
    changed_payload["container"]["message_ts"] = "1710000999.000999"
    changed_payload["message"]["ts"] = "1710000999.000999"
    changed = map_block_actions(changed_payload)
    assert base is not None and changed is not None
    assert base.event_id != changed.event_id


def test_different_actor_changes_event_id() -> None:
    base = map_block_actions(_action_payload())
    changed_payload = _action_payload()
    changed_payload["user"] = {"id": "UOTHER"}
    changed = map_block_actions(changed_payload)
    assert base is not None and changed is not None
    assert base.event_id != changed.event_id


def test_envelope_id_not_part_of_action_event_id() -> None:
    event_id = deterministic_block_action_event_id(
        team_id="TTEAM1",
        user_id="UUSER1",
        channel_id="DCHANNEL1",
        action_ts="1",
        message_ts="2",
        action_id="choose_workspace",
        selected_value="ws-1",
    )
    assert "envelope" not in event_id
    assert event_id == deterministic_block_action_event_id(
        team_id="TTEAM1",
        user_id="UUSER1",
        channel_id="DCHANNEL1",
        action_ts="1",
        message_ts="2",
        action_id="choose_workspace",
        selected_value="ws-1",
    )
