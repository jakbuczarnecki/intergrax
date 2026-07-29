# One-off generator — not part of product; delete after use.
"""Generate Teams Channel knowledge-read tests from Teams Chat test templates."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests/unit/integrations/providers/collaboration_suite"

COMMON_REPLACEMENTS: list[tuple[str, str]] = [
    ("Teams Chat knowledge-read", "Teams Channel knowledge-read"),
    ("teams_chat_inventory", "teams_channel_inventory"),
    ("teams_chat_members", "teams_channel_members"),
    ("teams_chat_messages", "teams_channel_messages"),
    ("teams_chat_hosted_content", "teams_channel_hosted_content"),
    ("MsGraphTeamsChatsReader", "MsGraphTeamsChannelsReader"),
    ("MsGraphTeamsChatMembersReader", "MsGraphTeamsChannelMembersReader"),
    ("MsGraphTeamsChatMessagesReader", "MsGraphTeamsChannelMessagesReader"),
    ("MsGraphTeamsChatHostedContentReader", "MsGraphTeamsChannelHostedContentReader"),
    ("MsGraphTeamsChatPage", "MsGraphTeamsChannelPage"),
    ("MsGraphTeamsChatMemberPage", "MsGraphTeamsChannelMemberPage"),
    ("MsGraphTeamsChatMessageSnapshotPage", "MsGraphTeamsChannelRootMessagePage"),
    ("MsGraphTeamsChatMessageWindow", "MsGraphTeamsChannelReplyPage"),
    ("MsGraphTeamsChatHostedContentPage", "MsGraphTeamsChannelHostedContentPage"),
    ("MsGraphTeamsChatHostedContentBytes", "MsGraphTeamsChannelHostedContentBytes"),
    ("MsGraphTeamsChatHostedContent", "MsGraphTeamsChannelHostedContent"),
    ("MsGraphTeamsChatMember", "MsGraphTeamsChannelMember"),
    ("MsGraphTeamsChat", "MsGraphTeamsChannel"),
    ("MsGraphTeamsChatType", "MsGraphTeamsChannelMembershipType"),
    ("MsGraphTeamsChatMigrationMode", "REMOVE_ME"),
    ("MsGraphTeamsChatMemberKind", "MsGraphTeamsChannelMemberKind"),
    ("MsGraphTeamsChatMemberRole", "MsGraphTeamsChannelMemberRole"),
    ("MsGraphTeamsChatMessage", "MsGraphTeamsChannelMessage"),
    ("MsGraphTeamsChatMessageState", "MsGraphTeamsChannelMessageState"),
    ("MsGraphTeamsChatMessageType", "MsGraphTeamsChannelMessageType"),
    ("MsGraphTeamsChatBodyKind", "MsGraphTeamsChannelBodyKind"),
    ("MsGraphTeamsChatImportance", "MsGraphTeamsChannelImportance"),
    ("MsGraphTeamsChatAttachmentKind", "MsGraphTeamsChatAttachmentKind"),
    ("MsGraphTeamsChatAttachmentReference", "MsGraphTeamsChatAttachmentReference"),
    ("MsGraphTeamsChatMention", "MsGraphTeamsChatMention"),
    ("MsGraphTeamsChatReaction", "MsGraphTeamsChatReaction"),
    ("MsGraphTeamsChatChanged", "MsGraphTeamsChannelChanged"),
    ("MsGraphTeamsChatMessageChanged", "MsGraphTeamsChannelMessageChanged"),
    ("MsGraphTeamsChatHostedContentTooLarge", "MsGraphTeamsChannelHostedContentTooLarge"),
    ("parse_msgraph_teams_chat", "parse_msgraph_teams_channel"),
    ("validate_msgraph_teams_chat", "validate_msgraph_teams_channel"),
    ("validate_msgraph_teams_chat_page", "validate_msgraph_teams_channel_page"),
    ("validate_msgraph_teams_chats_continuation", "validate_msgraph_teams_channels_continuation"),
    ("validate_msgraph_teams_chat_member", "validate_msgraph_teams_channel_member"),
    ("validate_msgraph_teams_chat_member_page", "validate_msgraph_teams_channel_member_page"),
    ("validate_msgraph_teams_chat_members_continuation", "validate_msgraph_teams_channel_members_continuation"),
    ("parse_msgraph_teams_chat_member", "parse_msgraph_teams_channel_member"),
    ("parse_msgraph_teams_chat_message", "parse_msgraph_teams_channel_message"),
    ("validate_msgraph_teams_chat_message", "validate_msgraph_teams_channel_message"),
    ("validate_msgraph_teams_chat_message_snapshot_page", "validate_msgraph_teams_channel_root_message_page"),
    ("validate_msgraph_teams_chat_messages_continuation", "validate_msgraph_teams_channel_root_messages_continuation"),
    ("validate_msgraph_teams_chat_hosted_content", "validate_msgraph_teams_channel_hosted_content"),
    ("validate_msgraph_teams_chat_hosted_content_page", "validate_msgraph_teams_channel_hosted_content_page"),
    ("validate_msgraph_teams_chat_hosted_content_bytes", "validate_msgraph_teams_channel_hosted_content_bytes"),
    ("validate_msgraph_teams_chat_hosted_contents_continuation", "validate_msgraph_teams_channel_hosted_contents_continuation"),
    ("read_teams_chats_page", "read_teams_channels_page"),
    ("read_teams_chat_members_page", "read_teams_channel_members_page"),
    ("read_teams_chat_messages_page", "read_teams_channel_root_messages_page"),
    ("read_teams_chat_hosted_contents_page", "read_teams_channel_hosted_contents_page"),
    ("read_teams_chat_hosted_content_bytes", "read_teams_channel_hosted_content_bytes"),
    ("read_chats_page", "read_teams_channels_page"),
    ("read_chat_members_page", "read_teams_channel_members_page"),
    ("_teams_chats_reader", "_teams_channels_reader"),
    ("_teams_chat_members_reader", "_teams_channel_members_reader"),
    ("_teams_chat_messages_reader", "_teams_channel_messages_reader"),
    ("_teams_chat_hosted_content_reader", "_teams_channel_hosted_content_reader"),
    ("_graph_base_url_for_teams_chat_validation", "_graph_base_url_for_teams_channel_validation"),
    ("Teams chats capability", "Teams channels capability"),
    ("Teams Chat validation is not configured", "Teams Channel validation is not configured"),
    ("Teams chat changed during read", "Teams channel changed during read"),
    ("Teams chat members response", "Teams channel members response"),
    ("Teams chat members continuation", "Teams channel members continuation"),
    ("Teams chat messages response", "Teams channel messages response"),
    ("Teams chat messages continuation", "Teams channel messages continuation"),
    ("Teams chat messages request", "Teams channel messages request"),
    ("Teams chats response", "Teams channels response"),
    ("Teams chats request", "Teams channels request"),
    ("Teams chats continuation", "Teams channels continuation"),
    ("Teams hosted content response", "Teams hosted content response"),
    ("Teams hosted content continuation", "Teams hosted content continuation"),
    ("Teams hosted content request", "Teams hosted content request"),
    ("ABSOLUTE_TEAMS_CHAT_MESSAGE_MAX_CHARS", "ABSOLUTE_TEAMS_CHANNEL_MESSAGE_MAX_CHARS"),
    ("DEFAULT_TEAMS_CHAT_MESSAGE_MAX_CHARS", "DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS"),
    ("ABSOLUTE_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES", "ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES"),
    ("DEFAULT_TEAMS_CHAT_HOSTED_CONTENT_MAX_BYTES", "DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES"),
    ("mailbox_user_id", "team_remote_id"),
    ("expected_mailbox_user_id", "expected_team_id"),
    ("_MAILBOX_USER_ID", "_TEAM_ID"),
    ("_OTHER_MAILBOX_USER_ID", "_OTHER_TEAM_ID"),
    ("_QUOTED_MAILBOX", "_QUOTED_TEAM"),
    ("_QUOTED_OTHER_MAILBOX", "_QUOTED_OTHER_TEAM"),
    ("chat_remote_id", "channel_remote_id"),
    ("_CHAT_ID", "_CHANNEL_ID"),
    ("_OTHER_CHAT_ID", "_OTHER_CHANNEL_ID"),
    ("_OPAQUE_CHAT_ID", "_OPAQUE_CHANNEL_ID"),
    ("_QUOTED_CHAT", "_QUOTED_CHANNEL"),
    ("_QUOTED_OTHER_CHAT", "_QUOTED_OTHER_CHANNEL"),
    ("chat_id", "channel_id"),
    ("chat_type", "membership_type"),
    ("chatType", "membershipType"),
    ("MsGraphTeamsChatType.ONE_ON_ONE", "MsGraphTeamsChannelMembershipType.STANDARD"),
    ("MsGraphTeamsChatType.GROUP", "MsGraphTeamsChannelMembershipType.PRIVATE"),
    ("MsGraphTeamsChatType.MEETING", "MsGraphTeamsChannelMembershipType.SHARED"),
    ("oneOnOne", "standard"),
    ("group", "private"),
    ("meeting", "shared"),
    ("futureChatType", "futureMembershipType"),
    ("is_hidden_for_all_members", "is_archived"),
    ("isHiddenForAllMembers", "isArchived"),
    ("last_updated_at", "REMOVE_LAST_UPDATED"),
    ("lastUpdatedDateTime", "REMOVE_LAST_UPDATED_FIELD"),
    ("original_created_at", "REMOVE_ORIGINAL"),
    ("originalCreatedDateTime", "REMOVE_ORIGINAL_FIELD"),
    ("migration_mode", "REMOVE_MIGRATION"),
    ("migrationMode", "REMOVE_MIGRATION_FIELD"),
    ("has_online_meeting_info", "REMOVE_ONLINE"),
    ("onlineMeetingInfo", "REMOVE_ONLINE_FIELD"),
    ("topic", "display_name"),
    ("displayName", "displayName"),
    ("_TOPIC", "_DISPLAY_NAME"),
    ("_HIDDEN_TOPIC", "_HIDDEN_DISPLAY_NAME"),
    ("Project Discussion", "General Channel"),
    ("Hidden Topic", "Hidden Display Name"),
    ("/users/", "/teams/"),
    ("/chats", "/channels"),
    ("Chats", "Channels"),
    ("chats", "channels"),
    ("chat members", "channel members"),
    ("chat messages", "channel messages"),
    ("chat inventory", "channel inventory"),
    ("_CustomGraphChatsClient", "_CustomGraphChannelsClient"),
    ("_CustomChatsSuite", "_CustomChannelsSuite"),
    ("_CustomSuiteWithoutChats", "_CustomSuiteWithoutChannels"),
    ("_CountingChatsClient", "_CountingChannelsClient"),
    ("read_teams_chat_replies_page", "read_teams_channel_replies_page"),
    ("validate_msgraph_teams_chat_reply_page", "validate_msgraph_teams_channel_reply_page"),
    ("validate_msgraph_teams_chat_replies_continuation", "validate_msgraph_teams_channel_replies_continuation"),
    ("MsGraphTeamsChannelMessageKind.ROOT", "MsGraphTeamsChannelMessageKind.ROOT"),
    ("thread_root_remote_id", "thread_root_remote_id"),
    ("root_message_remote_id", "root_message_remote_id"),
    ("message_kind", "message_kind"),
    ("MsGraphTeamsChannelMessageKind", "MsGraphTeamsChannelMessageKind"),
    ("format_msgraph_teams_chat_window_datetime", "REMOVE_FORMAT"),
    ("validate_msgraph_teams_chat_attachment_reference", "validate_msgraph_teams_chat_attachment_reference"),
    ("validate_msgraph_teams_chat_mention", "validate_msgraph_teams_chat_mention"),
    ("validate_msgraph_teams_chat_reaction", "validate_msgraph_teams_chat_reaction"),
    ("MsGraphTeamsForwardedMessageReference", "MsGraphTeamsForwardedMessageReference"),
    ("MsGraphTeamsIdentity", "MsGraphTeamsIdentity"),
    ("MsGraphTeamsIdentityKind", "MsGraphTeamsIdentityKind"),
    ("read_teams_chat_messages_snapshot_page", "read_teams_channel_root_messages_page"),
    ("read_teams_chat_message_window_page", "read_teams_channel_replies_page"),
    ("_MEMBERS_PATH", "_ALL_MEMBERS_PATH"),
    ("/members", "/allMembers"),
    ("members?$skiptoken", "allMembers?$skiptoken"),
    ("Members?$skiptoken", "allMembers?$skiptoken"),
    ("'/members'", "'/allMembers'"),
    ("_OBSERVATION_PATH", "_CHANNEL_OBSERVATION_PATH"),
    ("_CHAT_LAST_UPDATED", "_CHANNEL_CREATED"),
    ("_OTHER_CHAT_LAST_UPDATED", "_OTHER_CHANNEL_CREATED"),
    ("chat_last_updated", "channel_created"),
    ("_REVISION", "_REVISION"),
    ("_ROOT_MESSAGE_ID", "_ROOT_MESSAGE_ID"),
    ("_REPLY_ID", "_REPLY_ID"),
    ("_THREAD_ROOT_ID", "_THREAD_ROOT_ID"),
    ("_HOSTED_CONTENT_ID", "_HOSTED_CONTENT_ID"),
    ("_slash_hosted_contents_next_link", "_slash_hosted_contents_next_link"),
    ("_odata_hosted_contents_next_link", "_odata_hosted_contents_next_link"),
    ("_hosted_contents_next_link", "_hosted_contents_next_link"),
    ("_slash_messages_next_link", "_slash_root_messages_next_link"),
    ("_odata_messages_next_link", "_odata_root_messages_next_link"),
    ("_messages_next_link", "_root_messages_next_link"),
    ("_slash_replies_next_link", "_slash_replies_next_link"),
    ("read_teams_chat_messages_page", "read_teams_channel_root_messages_page"),
    ("validate_msgraph_teams_chat_messages_window_continuation", "validate_msgraph_teams_channel_replies_continuation"),
    ("MsGraphTeamsChatMessageWindow", "MsGraphTeamsChannelReplyPage"),
    ("validate_msgraph_teams_chat_message_window_page", "validate_msgraph_teams_channel_reply_page"),
    ("snapshot", "root"),
    ("Snapshot", "Root"),
    ("window", "reply"),
    ("Window", "Reply"),
    ("chatId", "channelIdentity"),
    ("channelIdentity", "channelIdentity"),
]


def apply_replacements(text: str) -> str:
    for old, new in COMMON_REPLACEMENTS:
        if new.startswith("REMOVE"):
            continue
        text = text.replace(old, new)
    return text


def transform_inventory(text: str) -> str:
    text = apply_replacements(text)
    text = text.replace(
        "_ROOT_PATH = f\"/teams/{_QUOTED_TEAM}/channels\"",
        "_ROOT_PATH = f\"/teams/{_QUOTED_TEAM}/channels\"\n"
        "_CHANNEL_SELECT = (\n"
        "    \"id,displayName,description,createdDateTime,membershipType,isArchived,tenantId\"\n"
        ")",
    )
    # Remove limit from read calls
    text = re.sub(r",\s*limit=50", "", text)
    text = re.sub(r"limit: int = 50,\n", "", text)
  # fix request test - channel uses $select not $top
    text = text.replace(
        "assert call.kwargs[\"params\"][\"$top\"] == 50",
        "assert call.kwargs[\"params\"][\"$select\"] == _CHANNEL_SELECT",
    )
    text = text.replace("def test_request_path_top_and_prefer_header", "def test_request_path_select_and_prefer_header")
    text = text.replace("def test_parse_one_on_one_chat", "def test_parse_standard_channel")
    text = text.replace("def test_parse_group_chat", "def test_parse_private_channel")
    text = text.replace("def test_parse_meeting_chat", "def test_parse_shared_channel")
    text = text.replace("def test_parse_unknown_future_chat_type", "def test_parse_unknown_membership_type")
    text = text.replace("def test_parse_topic_present", "def test_parse_description_present")
    text = text.replace("def test_parse_topic_absent", "def test_parse_description_absent")
    text = text.replace("def test_parse_topic_null", "def test_parse_description_null")
    text = text.replace("def test_parse_topic_empty_becomes_none", "def test_parse_description_empty_becomes_none")
    text = text.replace("def test_parse_topic_trimmed", "def test_parse_display_name_trimmed")
    text = text.replace("def test_parse_topic_over_limit_rejected", "def test_parse_description_over_limit_rejected")
    text = text.replace("def test_parse_opaque_chat_id", "def test_parse_opaque_channel_id")
    text = text.replace(
        "def test_parse_expected_mailbox_user_id_preserved",
        "def test_parse_expected_team_id_preserved",
    )
    text = text.replace("def test_page_cross_mailbox_item_rejected", "def test_page_cross_team_item_rejected")
    text = text.replace("def test_page_duplicate_chat_ids_rejected", "def test_page_duplicate_channel_ids_rejected")
    text = text.replace("def test_page_multiple_chats", "def test_page_multiple_channels")
    text = text.replace("def test_malformed_chat_model_construct_rejected", "def test_malformed_channel_model_construct_rejected")
    text = text.replace("def test_security_chat_repr_and_errors", "def test_security_channel_repr_and_errors")
    text = text.replace("def test_custom_client_without_chats_capability_fails", "def test_custom_client_without_channels_capability_fails")
    text = text.replace("description", "description")
    # channel payload helper
    text = re.sub(
        r"def _chat_payload\([^)]*\)[^:]*:",
        "def _channel_payload(\n    *,\n    channel_id: str = _CHANNEL_ID,\n    membership_type: str = \"standard\",\n    created_at: str = _CREATED_AT_STR,\n    is_archived: bool = False,\n    display_name: str = _DISPLAY_NAME,\n    description: str | None | object = _MISSING,\n    tenant_id: str | None | object = _MISSING,\n    extra_field: str | None = None,\n) -> dict[str, Any]:",
        text,
        count=1,
    )
    text = text.replace("_parse_chat(", "_parse_channel(")
    text = text.replace("_valid_chat(", "_valid_channel(")
    text = text.replace("def _parse_chat", "def _parse_channel")
    text = text.replace("def _valid_chat", "def _valid_channel")
    # rejection urls for allChannels/incomingChannels
    text = text.replace(
        "f\"https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x\",",
        "f\"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/allChannels?$skiptoken={_SECRET_TOKEN}\",\n"
        "        f\"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/incomingChannels?$skiptoken={_SECRET_TOKEN}\",\n"
        "        \"https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x\",",
    )
    return text


def main() -> None:
    pairs = [
        ("test_ms365_graph_knowledge_teams_chat_inventory.py", "test_ms365_graph_knowledge_teams_channel_inventory.py", transform_inventory),
    ]
    for src_name, dst_name, fn in pairs:
        src = TEST_DIR / src_name
        dst = TEST_DIR / dst_name
        content = fn(src.read_text(encoding="utf-8"))
        dst.write_text(content, encoding="utf-8")
        print(f"Wrote {dst_name} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
