# Fix messages test from chat template - v2
from __future__ import annotations

import re
from pathlib import Path

TD = Path(__file__).resolve().parents[1] / "tests/unit/integrations/providers/collaboration_suite"
src = (TD / "test_ms365_graph_knowledge_teams_chat_messages.py").read_text(encoding="utf-8")

pairs = [
    ("Teams Chat knowledge-read messages", "Teams Channel knowledge-read messages"),
    ("teams_chat_inventory", "teams_channel_inventory"),
    ("teams_chat_messages", "teams_channel_messages"),
    ("MsGraphTeamsChatMessagesReader", "MsGraphTeamsChannelMessagesReader"),
    ("MsGraphTeamsChatMessageSnapshotPage", "MsGraphTeamsChannelRootMessagePage"),
    ("validate_msgraph_teams_chat_message_snapshot_page", "validate_msgraph_teams_channel_root_message_page"),
    ("validate_msgraph_teams_chat_messages_continuation", "validate_msgraph_teams_channel_root_messages_continuation"),
    ("read_teams_chat_messages_page", "read_teams_channel_root_messages_page"),
    ("parse_msgraph_teams_chat", "parse_msgraph_teams_channel"),
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
    ("MsGraphTeamsChatBodyKind", "MsGraphTeamsChannelBodyKind"),
    ("MsGraphTeamsChatImportance", "MsGraphTeamsChannelImportance"),
    ("MsGraphTeamsChatMessageType", "MsGraphTeamsChannelMessageType"),
    ("MsGraphTeamsChatMessageState", "MsGraphTeamsChannelMessageState"),
    ("MsGraphTeamsChatMessage", "MsGraphTeamsChannelMessage"),
    ("/users/", "/teams/"),
    ("/chats", "/channels"),
    ("_TEAM_ID = \"user@contoso.com\"", "_TEAM_ID = \"team-abc-123\""),
    ("_OTHER_TEAM_ID = \"other@contoso.com\"", "_OTHER_TEAM_ID = \"other-team-456\""),
    ("_CHANNEL_ID = \"19:chat-abc@thread.v2\"", "_CHANNEL_ID = \"channel-abc-123\""),
    ("_OTHER_CHANNEL_ID = \"19:other-chat@thread.v2\"", "_OTHER_CHANNEL_ID = \"other-channel-456\""),
    ("_MESSAGE_ID = \"msg-001\"", "_MESSAGE_ID = \"root-msg-001\""),
    ("\"chatId\": _CHANNEL_ID", "\"channelIdentity\": {\"teamId\": _TEAM_ID, \"channelId\": _CHANNEL_ID}"),
    ("def _chat()", "def _channel()"),
    ("_chat()", "_channel()"),
    ("_valid_snapshot_page", "_valid_root_page"),
    ("snapshot_page", "root_page"),
]
for a, b in pairs:
    src = src.replace(a, b)

# remove window tests and format test
for fn in [
    "test_format_window_datetime_utc_z",
    "test_window_rejects_naive_and_invalid_bounds",
    "test_snapshot_page_rejects_message_outside_window",
    "test_snapshot_page_accepts_message_inside_window",
    "test_snapshot_page_is_complete_when_no_continuation",
]:
    src = re.sub(rf"def {fn}\(.*?\n(?=\ndef |\Z)", "", src, count=1, flags=re.S)

# remove window import and class usage
src = re.sub(r"    MsGraphTeamsChatMessageWindow,\n", "", src)
src = re.sub(r"    format_msgraph_teams_chat_window_datetime,\n", "", src)
src = re.sub(r"def _window\(\).*?return MsGraphTeamsChannelMessageWindow\(.*?\)\n\n", "", src, count=1, flags=re.S)

# fix reader calls: replace window=... with channel=_channel()
src = re.sub(r",\s*window=_window\(\)", ", channel=_channel()", src)
src = re.sub(r"window=_window\(\)", "channel=_channel()", src)

# add channel imports
src = src.replace(
    "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (\n    parse_msgraph_teams_channel,\n)",
    "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (\n    MsGraphTeamsChannel,\n    MsGraphTeamsChannelMembershipType,\n    parse_msgraph_teams_channel,\n)",
)
src = src.replace(
    "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (",
    "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (\n    MsGraphTeamsChannelMessageKind,\n    MsGraphTeamsChannelMessagesReader,\n    MsGraphTeamsChannelRootMessagePage,\n    MsGraphTeamsChannelMessageState,\n",
)

# _channel helper body
src = src.replace(
    "def _channel():\n    return parse_msgraph_teams_channel(\n        {\n            \"id\": _CHANNEL_ID,\n            \"chatType\": \"group\",",
    "def _channel() -> MsGraphTeamsChannel:\n    return parse_msgraph_teams_channel(\n        {\n            \"id\": _CHANNEL_ID,\n            \"displayName\": \"General\",\n            \"membershipType\": \"standard\",\n            \"isArchived\": False,",
)
src = re.sub(r"\"createdDateTime\".*?\n            \"lastUpdatedDateTime\".*?\n            \"isHiddenForAllMembers\": False,\n        },\n        expected_team_id=_TEAM_ID,\n    \)", "        },\n        expected_team_id=_TEAM_ID,\n    )", src, count=1, flags=re.S)

# valid message model fields
src = src.replace(
    '"team_remote_id": _TEAM_ID,\n        "channel_remote_id": _CHANNEL_ID,\n        "remote_id": _MESSAGE_ID,',
    '"team_remote_id": _TEAM_ID,\n        "channel_remote_id": _CHANNEL_ID,\n        "thread_root_remote_id": _MESSAGE_ID,\n        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,\n        "remote_id": _MESSAGE_ID,',
)

# parse calls add message_kind ROOT
src = src.replace(
    "parse_msgraph_teams_channel_message(\n        _active_message_payload(),\n        expected_team_id=_TEAM_ID,\n        expected_channel_id=_CHANNEL_ID,",
    "parse_msgraph_teams_channel_message(\n        _active_message_payload(),\n        expected_team_id=_TEAM_ID,\n        expected_channel_id=_CHANNEL_ID,\n        message_kind=MsGraphTeamsChannelMessageKind.ROOT,",
)

out = TD / "test_ms365_graph_knowledge_teams_channel_messages.py"
out.write_text(src, encoding="utf-8")
print(len(src.splitlines()))
