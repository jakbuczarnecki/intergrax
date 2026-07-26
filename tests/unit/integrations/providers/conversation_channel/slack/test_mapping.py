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


def test_unsupported_subtype_ignored() -> None:
    for subtype in (
        "message_changed",
        "message_deleted",
        "bot_message",
        "thread_broadcast",
        "channel_join",
        "channel_leave",
    ):
        assert map_events_api_message(_dm_payload(event={"subtype": subtype})) is None


def test_blank_text_without_files_rejected() -> None:
    assert map_events_api_message(_dm_payload(event={"text": "  "})) is None


def _file_entry(**overrides: object) -> dict:
    entry: dict = {
        "id": "F111",
        "name": "contract.pdf",
        "mimetype": "application/pdf",
        "size": 1234,
        "url_private": "https://files.slack.com/files-pri/T-F/private",
        "url_private_download": "https://files.slack.com/files-pri/T-F/download",
        "permalink": "https://slack.com/files/U/F",
        "title": "should-not-map",
    }
    entry.update(overrides)
    return entry


def test_ordinary_dm_message_with_files_maps_attachments() -> None:
    event = map_events_api_message(
        _dm_payload(event={"files": [_file_entry()], "text": "please index"})
    )
    assert event is not None
    assert event.text == "please index"
    assert len(event.attachments) == 1
    ref = event.attachments[0]
    assert ref.attachment_id == "F111"
    assert ref.file_name == "contract.pdf"
    assert ref.content_type == "application/pdf"
    assert ref.size_bytes == 1234
    assert ref.metadata == {}
    assert "url_private" not in event.metadata
    assert "url_private_download" not in event.metadata


def test_file_share_subtype_with_files_maps_attachments() -> None:
    event = map_events_api_message(
        _dm_payload(
            event={
                "subtype": "file_share",
                "text": "",
                "files": [_file_entry(id="F222", name="notes.txt", mimetype="text/plain")],
            }
        )
    )
    assert event is not None
    assert event.text is None or event.text == ""
    assert len(event.attachments) == 1
    assert event.attachments[0].attachment_id == "F222"


def test_attachment_only_dm_maps() -> None:
    event = map_events_api_message(
        _dm_payload(event={"text": "  ", "files": [_file_entry()]})
    )
    assert event is not None
    assert event.text is None
    assert len(event.attachments) == 1


def test_text_and_files_maps_both() -> None:
    event = map_events_api_message(
        _dm_payload(event={"text": "caption", "files": [_file_entry()]})
    )
    assert event is not None
    assert event.text == "caption"
    assert len(event.attachments) == 1


def test_multiple_files_preserve_provider_order() -> None:
    event = map_events_api_message(
        _dm_payload(
            event={
                "files": [
                    _file_entry(id="F1", name="a.pdf"),
                    _file_entry(id="F2", name="b.pdf"),
                    _file_entry(id="F3", name="c.pdf"),
                ]
            }
        )
    )
    assert event is not None
    assert [a.attachment_id for a in event.attachments] == ["F1", "F2", "F3"]


def test_private_url_not_mapped_and_file_object_not_copied() -> None:
    event = map_events_api_message(_dm_payload(event={"files": [_file_entry()]}))
    assert event is not None
    serialized = str(event.model_dump())
    assert "files.slack.com" not in serialized
    assert "url_private" not in serialized
    assert "permalink" not in event.metadata
    assert event.attachments[0].metadata == {}


def test_malformed_files_value_rejected() -> None:
    assert map_events_api_message(_dm_payload(event={"files": "bad"})) is None


def test_missing_file_id_rejects_whole_event() -> None:
    assert (
        map_events_api_message(
            _dm_payload(event={"files": [{"name": "a.pdf", "mimetype": "application/pdf"}]})
        )
        is None
    )


def test_one_malformed_item_rejects_whole_event() -> None:
    assert (
        map_events_api_message(
            _dm_payload(
                event={
                    "files": [
                        _file_entry(id="F1"),
                        "not-a-mapping",
                    ]
                }
            )
        )
        is None
    )


def test_bot_file_message_ignored() -> None:
    assert (
        map_events_api_message(
            _dm_payload(event={"bot_id": "B1", "files": [_file_entry()], "text": ""})
        )
        is None
    )


def test_non_im_file_message_ignored() -> None:
    assert (
        map_events_api_message(
            _dm_payload(
                event={
                    "channel_type": "channel",
                    "files": [_file_entry()],
                    "text": "",
                }
            )
        )
        is None
    )


def test_file_shared_top_level_event_ignored() -> None:
    payload = {
        "event_id": "EvFILE",
        "team_id": "TTEAM1",
        "event": {
            "type": "file_shared",
            "file_id": "F111",
            "user_id": "UUSER1",
            "channel_id": "DCHANNEL1",
        },
    }
    assert map_events_api_message(payload) is None


def test_attachment_event_id_remains_top_level() -> None:
    event = map_events_api_message(_dm_payload(event={"files": [_file_entry()]}))
    assert event is not None
    assert event.event_id == "EvMESSAGE1"


def test_attachment_thread_ts_behavior_unchanged() -> None:
    event = map_events_api_message(
        _dm_payload(
            event={
                "thread_ts": "1700000000.000001",
                "files": [_file_entry()],
                "text": "",
            }
        )
    )
    assert event is not None
    assert event.address.thread_id == "1700000000.000001"


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
