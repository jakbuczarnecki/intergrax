# Transform messages and hosted_content tests from chat to channel.
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "tests/unit/integrations/providers/collaboration_suite"

TEAM = "team-abc-123"
OTHER_TEAM = "other-team-456"
CHANNEL = "channel-abc-123"
OTHER_CHANNEL = "other-channel-456"
ROOT_MSG = "root-msg-001"
REPLY_MSG = "reply-msg-002"
OTHER_ROOT = "other-root-msg"
MSG_ID = ROOT_MSG
ETAG = "etag-1"


def replace_all(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        text = text.replace(old, new)
    return text


MSG_PAIRS = [
    ("Teams Chat knowledge-read messages", "Teams Channel knowledge-read messages"),
    ("teams_chat_inventory", "teams_channel_inventory"),
    ("teams_chat_messages", "teams_channel_messages"),
    ("parse_msgraph_teams_chat", "parse_msgraph_teams_channel"),
    ("MsGraphTeamsChatMessagesReader", "MsGraphTeamsChannelMessagesReader"),
    ("MsGraphTeamsChatMessageSnapshotPage", "MsGraphTeamsChannelRootMessagePage"),
    ("MsGraphTeamsChatMessageWindow", "REMOVE_WINDOW"),
    ("validate_msgraph_teams_chat_message_snapshot_page", "validate_msgraph_teams_channel_root_message_page"),
    ("validate_msgraph_teams_chat_messages_continuation", "validate_msgraph_teams_channel_root_messages_continuation"),
    ("read_teams_chat_messages_page", "read_teams_channel_root_messages_page"),
    ("parse_msgraph_teams_chat_message", "parse_msgraph_teams_channel_message"),
    ("validate_msgraph_teams_chat_message", "validate_msgraph_teams_channel_message"),
    ("ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS", "ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS"),
    ("DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS", "DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS"),
    ("_MAILBOX", "_TEAM_ID"),
    ("_OTHER_MAILBOX", "_OTHER_TEAM_ID"),
    ("_CHAT_ID", "_CHANNEL_ID"),
    ("_OTHER_CHAT_ID", "_OTHER_CHANNEL_ID"),
    ("_QUOTED_MAILBOX", "_QUOTED_TEAM"),
    ("_QUOTED_OTHER_MAILBOX", "_QUOTED_OTHER_TEAM"),
    ("_QUOTED_CHAT", "_QUOTED_CHANNEL"),
    ("unexpected Microsoft Graph Teams chat messages response", "unexpected Microsoft Graph Teams channel messages response"),
    ("invalid Microsoft Graph Teams chat messages continuation", "invalid Microsoft Graph Teams channel messages continuation"),
    ("invalid Microsoft Graph Teams chat messages request", "invalid Microsoft Graph Teams channel messages request"),
    ("Microsoft Graph Teams Chat validation is not configured", "Microsoft Graph Teams Channel validation is not configured"),
    ("mailbox_user_id", "team_remote_id"),
    ("expected_mailbox_user_id", "expected_team_id"),
    ("expected_chat_id", "expected_channel_id"),
    ("chat_remote_id", "channel_remote_id"),
    ("_teams_chat_messages_reader", "_teams_channel_messages_reader"),
    ("_graph_base_url_for_teams_chat_validation", "_graph_base_url_for_teams_channel_validation"),
    ("format_msgraph_teams_chat_window_datetime", "REMOVE_FORMAT"),
    ("def _chat()", "def _channel()"),
    ("_chat()", "_channel()"),
    ("/users/", "/teams/"),
    ("/chats", "/channels"),
    ("_TEAM_ID = \"user@contoso.com\"", f"_TEAM_ID = \"{TEAM}\""),
    ("_OTHER_TEAM_ID = \"other@contoso.com\"", f"_OTHER_TEAM_ID = \"{OTHER_TEAM}\""),
    ("_CHANNEL_ID = \"19:chat-abc@thread.v2\"", f"_CHANNEL_ID = \"{CHANNEL}\""),
    ("_OTHER_CHANNEL_ID = \"19:other-chat@thread.v2\"", f"_OTHER_CHANNEL_ID = \"{OTHER_CHANNEL}\""),
    ("_MESSAGE_ID = \"msg-001\"", f"_MESSAGE_ID = \"{ROOT_MSG}\""),
    ("\"chatId\": _CHANNEL_ID", "\"channelIdentity\": {\"teamId\": _TEAM_ID, \"channelId\": _CHANNEL_ID}"),
    ("\"chatId\": _CHAT_ID", "\"channelIdentity\": {\"teamId\": _TEAM_ID, \"channelId\": _CHANNEL_ID}"),
    ("MsGraphTeamsChatBodyKind", "MsGraphTeamsChannelBodyKind"),
    ("MsGraphTeamsChatImportance", "MsGraphTeamsChannelImportance"),
    ("MsGraphTeamsChatMessageType", "MsGraphTeamsChannelMessageType"),
    ("MsGraphTeamsChatMessageState", "MsGraphTeamsChannelMessageState"),
    ("MsGraphTeamsChatMessage", "MsGraphTeamsChannelMessage"),
    ("read_teams_channel_root_messages_page", "read_teams_channel_root_messages_page"),
    ("def _window()", "def _REMOVE_WINDOW()"),
    ("_window()", "_REMOVE_WINDOW()"),
    ("_valid_snapshot_page", "_valid_root_page"),
    ("snapshot_page", "root_message_page"),
    ("SnapshotPage", "RootMessagePage"),
    ("snapshot", "root"),
    ("Snapshot", "Root"),
    ("_graph_base_url_for_teams_channel_validation", "_graph_base_url_for_teams_channel_validation"),
]


def fix_messages(text: str) -> str:
    text = replace_all(text, MSG_PAIRS)
    # remove window-related test functions
    for fn in [
        "test_format_window_datetime_utc_z",
        "test_window_rejects_naive_and_invalid_bounds",
        "test_snapshot_page_rejects_message_outside_window",
        "test_snapshot_page_accepts_message_inside_window",
        "test_snapshot_page_is_complete_when_no_continuation",
        "test_reader_snapshot_request_params",
        "def _REMOVE_WINDOW",
        "def _valid_root_page",
    ]:
        if fn.startswith("def "):
            text = re.sub(rf"{re.escape(fn)}\(.*?\n(?=\ndef |\Z)", "", text, count=1, flags=re.S)
        else:
            text = re.sub(rf"def {re.escape(fn)}\(.*?\n(?=\ndef |\Z)", "", text, count=1, flags=re.S)

    # add channel imports
    text = text.replace(
        "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (\n    parse_msgraph_teams_channel,\n)",
        "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (\n    MsGraphTeamsChannel,\n    MsGraphTeamsChannelMembershipType,\n    parse_msgraph_teams_channel,\n)",
    )
    text = text.replace(
        "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (",
        "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (\n    MsGraphTeamsChannelMessageKind,\n    MsGraphTeamsChannelMessageChanged,\n    read_and_validate_current_teams_channel_message_observation,\n    validate_msgraph_teams_channel_reply_page,\n    validate_msgraph_teams_channel_replies_continuation,\n",
    )

    # _channel helper
    if "def _channel()" not in text:
        text = text.replace(
            "def _config() -> Ms365GraphIntegrationConfig:",
            f"""def _channel() -> MsGraphTeamsChannel:
    return parse_msgraph_teams_channel(
        {{
            "id": _CHANNEL_ID,
            "displayName": "General",
            "membershipType": "standard",
            "isArchived": False,
        }},
        expected_team_id=_TEAM_ID,
    )


def _config() -> Ms365GraphIntegrationConfig:""",
        )

    # valid active message defaults for channel
    text = re.sub(
        r"def _valid_active_message\(\*\*overrides: object\) -> MsGraphTeamsChannelMessage:.*?return MsGraphTeamsChannelMessage\(\*\*defaults\)",
        f"""def _valid_active_message(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {{
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChannelBodyKind.TEXT,
        "body_content": "Hello",
    }}
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)""",
        text,
        count=1,
        flags=re.S,
    )

    # active message payload
    text = re.sub(
        r"def _active_message_payload\(\*\*overrides: Any\) -> dict\[str, Any\]:.*?return base",
        """def _active_message_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": _MESSAGE_ID,
        "etag": _ETAG,
        "messageType": "message",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-01T11:00:00Z",
        "deletedDateTime": None,
        "importance": "normal",
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
        "body": {"contentType": "text", "content": "Hello"},
        "from": {"user": {"id": "u1", "displayName": "Alice"}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base""",
        text,
        count=1,
        flags=re.S,
    )

    text = text.replace("parse_msgraph_teams_channel_message(\n        _active_message_payload(channelIdentity=", "REMOVE_BAD")
    text = text.replace("test_parse_channel_identity_rejected", "test_parse_chat_id_rejected")

    return text


HOSTED_PAIRS = [
    ("Teams Chat knowledge-read hosted", "Teams Channel knowledge-read hosted"),
    ("teams_chat_hosted_content", "teams_channel_hosted_content"),
    ("teams_chat_messages", "teams_channel_messages"),
    ("MsGraphTeamsChatHostedContentReader", "MsGraphTeamsChannelHostedContentReader"),
    ("MsGraphTeamsChatHostedContentPage", "MsGraphTeamsChannelHostedContentPage"),
    ("MsGraphTeamsChatHostedContentBytes", "MsGraphTeamsChannelHostedContentBytes"),
    ("MsGraphTeamsChatHostedContent", "MsGraphTeamsChannelHostedContent"),
    ("MsGraphTeamsChatHostedContentTooLarge", "MsGraphTeamsChannelHostedContentTooLarge"),
    ("MsGraphTeamsChatMessageChanged", "MsGraphTeamsChannelMessageChanged"),
    ("validate_msgraph_teams_chat_hosted_content", "validate_msgraph_teams_channel_hosted_content"),
    ("validate_msgraph_teams_chat_hosted_content_page", "validate_msgraph_teams_channel_hosted_content_page"),
    ("validate_msgraph_teams_chat_hosted_content_bytes", "validate_msgraph_teams_channel_hosted_content_bytes"),
    ("validate_msgraph_teams_chat_hosted_contents_continuation", "validate_msgraph_teams_channel_hosted_contents_continuation"),
    ("read_teams_chat_hosted_contents_page", "read_teams_channel_hosted_contents_page"),
    ("read_teams_chat_hosted_content_bytes", "read_teams_channel_hosted_content_bytes"),
    ("ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES", "ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES"),
    ("DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES", "DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES"),
    ("_MAILBOX_USER_ID", "_TEAM_ID"),
    ("_OTHER_MAILBOX_USER_ID", "_OTHER_TEAM_ID"),
    ("_CHAT_ID", "_CHANNEL_ID"),
    ("_OTHER_CHAT_ID", "_OTHER_CHANNEL_ID"),
    ("_QUOTED_MAILBOX", "_QUOTED_TEAM"),
    ("_QUOTED_OTHER_MAILBOX", "_QUOTED_OTHER_TEAM"),
    ("_QUOTED_CHAT", "_QUOTED_CHANNEL"),
    ("chat_remote_id", "channel_remote_id"),
    ("mailbox_user_id", "team_remote_id"),
    ("unexpected Microsoft Graph Teams hosted content", "unexpected Microsoft Graph Teams hosted content"),
    ("invalid Microsoft Graph Teams hosted content continuation", "invalid Microsoft Graph Teams hosted content continuation"),
    ("invalid Microsoft Graph Teams hosted content request", "invalid Microsoft Graph Teams hosted content request"),
    ("_teams_chat_hosted_content_reader", "_teams_channel_hosted_content_reader"),
    ("_graph_base_url_for_teams_chat_validation", "_graph_base_url_for_teams_channel_validation"),
    ("/users/", "/teams/"),
    ("/chats/", "/channels/"),
    ("_TEAM_ID = \"user@contoso.com\"", f"_TEAM_ID = \"{TEAM}\""),
    ("_CHANNEL_ID = \"19:chat", f"_CHANNEL_ID = \"{CHANNEL}\""),
    ("_MESSAGE_ID = \"msg-immutable-opaque-id\"", f"_MESSAGE_ID = \"{ROOT_MSG}\""),
    ("_OTHER_MESSAGE_ID = \"msg-other-opaque-id\"", f"_OTHER_MESSAGE_ID = \"{REPLY_MSG}\""),
    ("MsGraphTeamsChatMessage", "MsGraphTeamsChannelMessage"),
    ("MsGraphTeamsChatMessageState", "MsGraphTeamsChannelMessageState"),
    ("MsGraphTeamsChatBodyKind", "MsGraphTeamsChannelBodyKind"),
    ("MsGraphTeamsChatImportance", "MsGraphTeamsChannelImportance"),
    ("MsGraphTeamsChatMessageType", "MsGraphTeamsChannelMessageType"),
]


def fix_hosted(text: str) -> str:
    text = replace_all(text, HOSTED_PAIRS)
    text = text.replace(
        "/messages/{_QUOTED_MESSAGE_ID}/hostedContents",
        "/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents",
    )
    # hosted paths need thread root - simplify: messages/{root}/hostedContents stays for root messages
    return text


def main() -> None:
    src_msg = TD / "test_ms365_graph_knowledge_teams_chat_messages.py"
    dst_msg = TD / "test_ms365_graph_knowledge_teams_channel_messages.py"
    dst_msg.write_text(fix_messages(src_msg.read_text(encoding="utf-8")), encoding="utf-8")
    print("messages:", len(dst_msg.read_text(encoding="utf-8").splitlines()))

    src_h = TD / "test_ms365_graph_knowledge_teams_chat_hosted_content.py"
    dst_h = TD / "test_ms365_graph_knowledge_teams_channel_hosted_content.py"
    dst_h.write_text(fix_hosted(src_h.read_text(encoding="utf-8")), encoding="utf-8")
    print("hosted:", len(dst_h.read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    main()
