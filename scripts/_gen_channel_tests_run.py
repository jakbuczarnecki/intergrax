# Temporary generator — delete after use.
"""Transform Teams Chat knowledge-read tests into Teams Channel tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "tests/unit/integrations/providers/collaboration_suite"

TEAM = "team-abc-123"
OTHER_TEAM = "other-team-456"
CHANNEL = "channel-abc-123"
OTHER_CHANNEL = "other-channel-456"
OPAQUE = "channels/messages/allMembers/replies"


def replace_all(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        text = text.replace(old, new)
    return text


INVENTORY_PAIRS = [
    ("Teams Chat knowledge-read inventory", "Teams Channel knowledge-read inventory"),
    (
        "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (",
        "from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (",
    ),
    ("MsGraphTeamsChatMigrationMode,\n    ", ""),
    ("MsGraphTeamsChatType", "MsGraphTeamsChannelMembershipType"),
    ("MsGraphTeamsChatsReader", "MsGraphTeamsChannelsReader"),
    ("MsGraphTeamsChatPage", "MsGraphTeamsChannelPage"),
    ("MsGraphTeamsChat", "MsGraphTeamsChannel"),
    ("parse_msgraph_teams_chat", "parse_msgraph_teams_channel"),
    ("validate_msgraph_teams_chat_page", "validate_msgraph_teams_channel_page"),
    ("validate_msgraph_teams_chats_continuation", "validate_msgraph_teams_channels_continuation"),
    ("read_teams_chats_page", "read_teams_channels_page"),
    ("read_chats_page", "read_teams_channels_page"),
    ("_teams_chats_reader", "_teams_channels_reader"),
    ("_graph_base_url_for_teams_chat_validation", "_graph_base_url_for_teams_channel_validation"),
    ("Teams chats capability", "Teams channels capability"),
    ("_CustomGraphChatsClient", "_CustomGraphChannelsClient"),
    ("_CustomChatsSuite", "_CustomChannelsSuite"),
    ("_CustomSuiteWithoutChats", "_CustomSuiteWithoutChannels"),
    ("_CountingChatsClient", "_CountingChannelsClient"),
    ("_MAILBOX_USER_ID", "_TEAM_ID"),
    ("_OTHER_MAILBOX_USER_ID", "_OTHER_TEAM_ID"),
    ("_CHAT_ID", "_CHANNEL_ID"),
    ("_OTHER_CHAT_ID", "_OTHER_CHANNEL_ID"),
    ("_OPAQUE_CHAT_ID", "_OPAQUE_CHANNEL_ID"),
    ("_QUOTED_MAILBOX", "_QUOTED_TEAM"),
    ("_QUOTED_OTHER_MAILBOX", "_QUOTED_OTHER_TEAM"),
    ("_TOPIC", "_DISPLAY_NAME"),
    ("_HIDDEN_TOPIC", "_HIDDEN_DISPLAY_NAME"),
    ("Project Discussion", "General Channel"),
    ("Hidden Topic", "Hidden Display Name"),
    ("unexpected Microsoft Graph Teams chats response", "unexpected Microsoft Graph Teams channels response"),
    ("invalid Microsoft Graph Teams chats request", "invalid Microsoft Graph Teams channels request"),
    ("invalid Microsoft Graph Teams chats continuation", "invalid Microsoft Graph Teams channels continuation"),
    ("Microsoft Graph Teams Chat validation is not configured", "Microsoft Graph Teams Channel validation is not configured"),
    ("mailbox_user_id", "team_id"),
    ("expected_mailbox_user_id", "expected_team_id"),
    ("/users/", "/teams/"),
    ("/chats", "/channels"),
    ("Chats", "Channels"),
    ("_TEAM_ID = \"user@contoso.com\"", f"_TEAM_ID = \"{TEAM}\""),
    ("_OTHER_TEAM_ID = \"other@contoso.com\"", f"_OTHER_TEAM_ID = \"{OTHER_TEAM}\""),
    ("_CHANNEL_ID = \"chat-abc-123\"", f"_CHANNEL_ID = \"{CHANNEL}\""),
    ("_OTHER_CHANNEL_ID = \"other-chat\"", f"_OTHER_CHANNEL_ID = \"{OTHER_CHANNEL}\""),
    (
        "_OPAQUE_CHANNEL_ID = \"19:meeting_abc@thread.v2/special+id\"",
        f"_OPAQUE_CHANNEL_ID = \"{OPAQUE}\"",
    ),
    ("def _chat_payload", "def _channel_payload"),
    ("_chat_payload", "_channel_payload"),
    ("def _parse_chat", "def _parse_channel"),
    ("_parse_chat", "_parse_channel"),
    ("def _valid_chat", "def _valid_channel"),
    ("_valid_chat", "_valid_channel"),
    ("chat_type: str = \"oneOnOne\"", "membership_type: str = \"standard\""),
    ("chatType", "membershipType"),
    ("is_hidden: bool = False", "is_archived: bool = False"),
    ("isHiddenForAllMembers", "isArchived"),
    ("chat_type=\"oneOnOne\"", "membership_type=\"standard\""),
    ("chat_type=\"group\"", "membership_type=\"private\""),
    ("chat_type=\"meeting\"", "membership_type=\"shared\""),
    ("futureChatType", "futureMembershipType"),
    ("MsGraphTeamsChannelMembershipType.ONE_ON_ONE", "MsGraphTeamsChannelMembershipType.STANDARD"),
    ("MsGraphTeamsChannelMembershipType.GROUP", "MsGraphTeamsChannelMembershipType.PRIVATE"),
    ("MsGraphTeamsChannelMembershipType.MEETING", "MsGraphTeamsChannelMembershipType.SHARED"),
    ("chat_type", "membership_type"),
    ("is_hidden_for_all_members", "is_archived"),
    ("test_parse_one_on_one_chat", "test_parse_standard_channel"),
    ("test_parse_group_chat", "test_parse_private_channel"),
    ("test_parse_meeting_chat", "test_parse_shared_channel"),
    ("test_parse_unknown_future_chat_type", "test_parse_unknown_membership_type"),
    ("test_parse_topic", "test_parse_description"),
    ("test_parse_opaque_chat_id", "test_parse_opaque_channel_id"),
    ("test_parse_expected_mailbox_user_id_preserved", "test_parse_expected_team_id_preserved"),
    ("test_page_cross_mailbox_item_rejected", "test_page_cross_team_item_rejected"),
    ("test_page_duplicate_chat_ids_rejected", "test_page_duplicate_channel_ids_rejected"),
    ("test_page_multiple_chats", "test_page_multiple_channels"),
    ("test_malformed_chat_model_construct_rejected", "test_malformed_channel_model_construct_rejected"),
    ("test_security_chat_repr_and_errors", "test_security_channel_repr_and_errors"),
    ("test_custom_client_without_chats_capability_fails", "test_custom_client_without_channels_capability_fails"),
    ("test_request_path_top_and_prefer_header", "test_request_path_select_and_prefer_header"),
    ('assert call.kwargs["params"]["$top"] == 50', 'assert call.kwargs["params"]["$select"] == _CHANNEL_SELECT'),
    ("assert chat.topic", "assert channel.description"),
    ("chat.topic", "channel.description"),
    ("topic=", "description="),
    ("    chat = ", "    channel = "),
    ("    assert chat.", "    assert channel."),
    ("chat_id=", "channel_id="),
    ("last_updated_at: str = _UPDATED_AT_STR,", ""),
    ("        \"lastUpdatedDateTime\": last_updated_at,", ""),
    ("original_created_at", "REMOVE_ORIGINAL_CREATED"),
    ("originalCreatedDateTime", "REMOVE_ORIGINAL_CREATED_FIELD"),
    ("migration_mode", "REMOVE_MIGRATION"),
    ("migrationMode", "REMOVE_MIGRATION_FIELD"),
    ("has_online_meeting_info", "REMOVE_ONLINE"),
    ("onlineMeetingInfo", "REMOVE_ONLINE_FIELD"),
    ("MsGraphTeamsChannelMigrationMode", "REMOVED"),
    ("last_updated_at", "REMOVE_LAST_UPDATED"),
    ("lastUpdatedDateTime", "REMOVE_LAST_UPDATED_FIELD"),
]


def fix_inventory(text: str) -> str:
    text = replace_all(text, INVENTORY_PAIRS)
    text = text.replace(
        '_ROOT_PATH = f"/teams/{_QUOTED_TEAM}/channels"',
        '_ROOT_PATH = f"/teams/{_QUOTED_TEAM}/channels"\n_CHANNEL_SELECT = (\n'
        '    "id,displayName,description,createdDateTime,membershipType,isArchived,tenantId"\n'
        ")",
    )
    text = re.sub(r",\s*limit=50", "", text)
    text = re.sub(r"limit: int = 50,\n", "", text)
    text = re.sub(
        r"def _valid_channel\(\*\*overrides: object\) -> MsGraphTeamsChannel:.*?return MsGraphTeamsChannel\(\*\*defaults\)  # type: ignore\[arg-type\]",
        """def _valid_channel(**overrides: object) -> MsGraphTeamsChannel:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "remote_id": _CHANNEL_ID,
        "display_name": _DISPLAY_NAME,
        "description": None,
        "created_at": _CREATED_AT,
        "membership_type": MsGraphTeamsChannelMembershipType.STANDARD,
        "is_archived": False,
        "tenant_id": None,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannel(**defaults)  # type: ignore[arg-type]""",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        '"https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",',
        'f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/allChannels?$skiptoken={_SECRET_TOKEN}",\n'
        '        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/incomingChannels?$skiptoken={_SECRET_TOKEN}",\n'
        '        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",',
    )
    return text


MEMBERS_PAIRS = [
    ("Teams Chat knowledge-read members", "Teams Channel knowledge-read members"),
    ("teams_chat_inventory", "teams_channel_inventory"),
    ("teams_chat_members", "teams_channel_members"),
    ("MsGraphTeamsChatChanged", "MsGraphTeamsChannelChanged"),
    ("MsGraphTeamsChatType", "MsGraphTeamsChannelMembershipType"),
    ("MsGraphTeamsChatMembersReader", "MsGraphTeamsChannelMembersReader"),
    ("MsGraphTeamsChatMemberPage", "MsGraphTeamsChannelMemberPage"),
    ("MsGraphTeamsChatMember", "MsGraphTeamsChannelMember"),
    ("MsGraphTeamsChatMemberKind", "MsGraphTeamsChannelMemberKind"),
    ("MsGraphTeamsChatMemberRole", "MsGraphTeamsChannelMemberRole"),
    ("parse_msgraph_teams_chat_member", "parse_msgraph_teams_channel_member"),
    ("validate_msgraph_teams_chat_member", "validate_msgraph_teams_channel_member"),
    ("validate_msgraph_teams_chat_member_page", "validate_msgraph_teams_channel_member_page"),
    ("validate_msgraph_teams_chat_members_continuation", "validate_msgraph_teams_channel_members_continuation"),
    ("read_teams_chat_members_page", "read_teams_channel_members_page"),
    ("read_chat_members_page", "read_teams_channel_members_page"),
    ("_MAILBOX_USER_ID", "_TEAM_ID"),
    ("_OTHER_MAILBOX_USER_ID", "_OTHER_TEAM_ID"),
    ("_CHAT_ID", "_CHANNEL_ID"),
    ("_OTHER_CHAT_ID", "_OTHER_CHANNEL_ID"),
    ("_QUOTED_MAILBOX", "_QUOTED_TEAM"),
    ("_QUOTED_OTHER_MAILBOX", "_QUOTED_OTHER_TEAM"),
    ("_QUOTED_CHAT", "_QUOTED_CHANNEL"),
    ("_QUOTED_OTHER_CHAT", "_QUOTED_OTHER_CHANNEL"),
    ("unexpected Microsoft Graph Teams chat members response", "unexpected Microsoft Graph Teams channel members response"),
    ("invalid Microsoft Graph Teams chat members continuation", "invalid Microsoft Graph Teams channel members continuation"),
    ("Microsoft Graph Teams chat changed during read", "Microsoft Graph Teams channel changed during read"),
    ("_MEMBERS_PATH", "_ALL_MEMBERS_PATH"),
    ("/members", "/allMembers"),
    ("Members?$skiptoken", "allMembers?$skiptoken"),
    ("'/members'", "'/allMembers'"),
    ("_OBSERVATION_PATH", "_CHANNEL_OBSERVATION_PATH"),
    ("_CHAT_LAST_UPDATED", "_CHANNEL_CREATED_STR"),
    ("_OTHER_CHAT_LAST_UPDATED", "_OTHER_CHANNEL_CREATED_STR"),
    ("chatType", "membershipType"),
    ("chat_type", "membership_type"),
    ("lastUpdatedDateTime", "REMOVE_LAST_UPDATED"),
    ("isHiddenForAllMembers", "isArchived"),
    ("mailbox_user_id", "team_remote_id"),
    ("expected_mailbox_user_id", "expected_team_id"),
    ("chat_remote_id", "channel_remote_id"),
    ("chat_id", "channel_id"),
    ("chat=", "channel="),
    ("_chat(", "_channel("),
    ("def _chat(", "def _channel("),
    ("parse_msgraph_teams_chat", "parse_msgraph_teams_channel"),
    ("/users/", "/teams/"),
    ("/chats", "/channels"),
    ("_TEAM_ID = \"user@contoso.com\"", f"_TEAM_ID = \"{TEAM}\""),
    ("_OTHER_TEAM_ID = \"other@contoso.com\"", f"_OTHER_TEAM_ID = \"{OTHER_TEAM}\""),
    ("_CHANNEL_ID = \"19:chat-abc-123@thread.v2\"", f"_CHANNEL_ID = \"{CHANNEL}\""),
    ("_OTHER_CHANNEL_ID = \"19:other-chat@thread.v2\"", f"_OTHER_CHANNEL_ID = \"{OTHER_CHANNEL}\""),
    ("_graph_base_url_for_teams_chat_validation", "_graph_base_url_for_teams_channel_validation"),
    ("Teams Chat validation is not configured", "Teams Channel validation is not configured"),
    ("read_teams_chat_members_page", "read_teams_channel_members_page"),
    ("_teams_chat_members_reader", "_teams_channel_members_reader"),
    ("chat_last_updated", "channel_created"),
    ("chat_revision", "REMOVE_REVISION"),
    ("chat_remote_id", "channel_remote_id"),
    ("MsGraphTeamsChat", "MsGraphTeamsChannel"),
]


def fix_members(text: str) -> str:
    text = replace_all(text, MEMBERS_PAIRS)
    text = text.replace(
        '_ALL_MEMBERS_PATH = f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/allMembers"',
        '_ALL_MEMBERS_PATH = f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/allMembers"\n'
        f'_CHANNEL_OBSERVATION_PATH = f"/teams/{{_QUOTED_TEAM}}/channels/{{_QUOTED_CHANNEL}}"',
    )
    text = text.replace(
        '_CHANNEL_OBSERVATION_PATH = f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}"',
        f'_CHANNEL_OBSERVATION_PATH = f"/teams/{{_QUOTED_TEAM}}/channels/{{_QUOTED_CHANNEL}}"',
    )
    # observation payload for channel
    text = re.sub(
        r"def _observation_payload\(.*?return \{.*?\}\n",
        """def _observation_payload(
    *,
    channel_id: str = _CHANNEL_ID,
    membership_type: str = "standard",
    created_at: str = "2024-06-01T12:00:00Z",
) -> dict[str, Any]:
    return {
        "id": channel_id,
        "displayName": "General",
        "membershipType": membership_type,
        "isArchived": False,
        "createdDateTime": created_at,
    }


""",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"def _channel\(\):.*?return parse_msgraph_teams_channel\(.*?\)\n",
        f"""def _channel() -> MsGraphTeamsChannel:
    return parse_msgraph_teams_channel(
        {{
            "id": _CHANNEL_ID,
            "displayName": "General",
            "membershipType": "standard",
            "isArchived": False,
            "createdDateTime": "2024-06-01T12:00:00Z",
        }},
        expected_team_id=_TEAM_ID,
    )


""",
        text,
        count=1,
        flags=re.S,
    )
    # indirect member tests - append if missing
    indirect_tests = '''

_ORIGINAL_SOURCE_URL = "https://teams.microsoft.com/l/team/team-id/channel/channel-id"


def test_parse_direct_member() -> None:
    member = parse_msgraph_teams_channel_member(
        {
            "@odata.type": _ODATA_AAD,
            "id": _MEMBER_ID,
            "roles": ["owner"],
            "userId": _PROVIDER_USER_ID,
        },
        channel=_channel(),
    )
    assert member.is_indirect_member is False
    assert member.original_source_membership_url is None


def test_parse_indirect_member_with_url() -> None:
    member = parse_msgraph_teams_channel_member(
        {
            "@odata.type": _ODATA_AAD,
            "id": _MEMBER_ID,
            "roles": ["owner"],
            "@microsoft.graph.originalSourceMembershipUrl": _ORIGINAL_SOURCE_URL,
            "@microsoft.graph.isIndirectMember": True,
        },
        channel=_channel(),
    )
    assert member.is_indirect_member is True
    assert member.original_source_membership_url == _ORIGINAL_SOURCE_URL


def test_parse_contradictory_indirect_boolean_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        parse_msgraph_teams_channel_member(
            {
                "@odata.type": _ODATA_AAD,
                "id": _MEMBER_ID,
                "roles": [],
                "@microsoft.graph.originalSourceMembershipUrl": _ORIGINAL_SOURCE_URL,
                "isIndirectMember": False,
            },
            channel=_channel(),
        )


def test_parse_url_without_indirect_true_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        parse_msgraph_teams_channel_member(
            {
                "@odata.type": _ODATA_AAD,
                "id": _MEMBER_ID,
                "roles": [],
                "@microsoft.graph.originalSourceMembershipUrl": _ORIGINAL_SOURCE_URL,
                "isIndirectMember": False,
            },
            channel=_channel(),
        )


def test_parse_indirect_true_without_url_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        parse_msgraph_teams_channel_member(
            {
                "@odata.type": _ODATA_AAD,
                "id": _MEMBER_ID,
                "roles": [],
                "isIndirectMember": True,
            },
            channel=_channel(),
        )


def test_model_construct_inconsistent_indirect_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_channel_member(
            MsGraphTeamsChannelMember.model_construct(
                team_remote_id=_TEAM_ID,
                channel_remote_id=_CHANNEL_ID,
                remote_id=_MEMBER_ID,
                member_kind=MsGraphTeamsChannelMemberKind.AAD_USER,
                is_indirect_member=True,
                original_source_membership_url=None,
            )
        )


def test_security_repr_hides_provider_user_and_visible_history() -> None:
    member = parse_msgraph_teams_channel_member(
        {
            "@odata.type": _ODATA_AAD,
            "id": _MEMBER_ID,
            "roles": ["owner"],
            "userId": _PROVIDER_USER_ID,
            "visibleHistoryStartDateTime": _VISIBLE_HISTORY,
        },
        channel=_channel(),
    )
    rendered = repr(member)
    assert _PROVIDER_USER_ID not in rendered
    assert _VISIBLE_HISTORY not in rendered
'''
    if "test_parse_indirect_member_with_url" not in text:
        text += indirect_tests
    return text


def main() -> None:
    src_inv = TD / "test_ms365_graph_knowledge_teams_chat_inventory.py"
    dst_inv = TD / "test_ms365_graph_knowledge_teams_channel_inventory.py"
    dst_inv.write_text(fix_inventory(src_inv.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"inventory: {len(dst_inv.read_text(encoding='utf-8').splitlines())} lines")

    src_mem = TD / "test_ms365_graph_knowledge_teams_chat_members.py"
    dst_mem = TD / "test_ms365_graph_knowledge_teams_channel_members.py"
    dst_mem.write_text(fix_members(src_mem.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"members: {len(dst_mem.read_text(encoding='utf-8').splitlines())} lines")


if __name__ == "__main__":
    main()
