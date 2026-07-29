# Temporary script — do not commit
from pathlib import Path

p = Path("tests/unit/integrations/providers/collaboration_suite/test_ms365_graph_knowledge_teams_channel_hosted_content.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    "Microsoft Graph Teams Chat validation is not configured",
    "Microsoft Graph Teams Channel validation is not configured",
)
text = text.replace("team_remote_id=_TEAM_ID", "team_id=_TEAM_ID")
text = text.replace("chat_id=", "channel_id=")
text = text.replace("chats(", "channels(")
text = text.replace("/chats/", "/channels/")
text = text.replace(
    "_CustomGraphTeamsChatHostedContentClient",
    "_CustomGraphTeamsChannelHostedContentClient",
)

if "MsGraphTeamsChannelMessageKind" not in text:
    text = text.replace(
        "    MsGraphTeamsChannelMessageType,",
        "    MsGraphTeamsChannelMessageKind,\n    MsGraphTeamsChannelMessageType,",
    )

active_marker = (
    '    defaults: dict[str, object] = {\n'
    '        "team_remote_id": _TEAM_ID,\n'
    '        "channel_remote_id": _CHANNEL_ID,\n'
    '        "remote_id": _MESSAGE_ID,'
)
active_replacement = (
    '    defaults: dict[str, object] = {\n'
    '        "team_remote_id": _TEAM_ID,\n'
    '        "channel_remote_id": _CHANNEL_ID,\n'
    '        "thread_root_remote_id": _MESSAGE_ID,\n'
    '        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,\n'
    '        "remote_id": _MESSAGE_ID,'
)
text = text.replace(active_marker, active_replacement, 1)

deleted_marker = (
    '        "team_remote_id": _TEAM_ID,\n'
    '        "channel_remote_id": _CHANNEL_ID,\n'
    '        "remote_id": _MESSAGE_ID,\n'
    '        "revision": _REVISION,\n'
    '        "state": MsGraphTeamsChannelMessageState.DELETED,'
)
deleted_replacement = (
    '        "team_remote_id": _TEAM_ID,\n'
    '        "channel_remote_id": _CHANNEL_ID,\n'
    '        "thread_root_remote_id": _MESSAGE_ID,\n'
    '        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,\n'
    '        "remote_id": _MESSAGE_ID,\n'
    '        "revision": _REVISION,\n'
    '        "state": MsGraphTeamsChannelMessageState.DELETED,'
)
text = text.replace(deleted_marker, deleted_replacement, 1)

old_obs = (
    '    return {\n'
    '        "id": message_id,\n'
    '        "chatId": channel_id,\n'
    '        "etag": revision,\n'
    '        "deletedDateTime": "2024-06-01T12:00:00Z" if deleted else None,\n'
    '    }'
)
new_obs = (
    '    return {\n'
    '        "id": message_id,\n'
    '        "channelIdentity": {"teamId": _TEAM_ID, "channelId": channel_id},\n'
    '        "etag": revision,\n'
    '        "deletedDateTime": "2024-06-01T12:00:00Z" if deleted else None,\n'
    '    }'
)
text = text.replace(old_obs, new_obs)

# Add message_kind to continuation calls that lack it
import re

def add_message_kind(match: re.Match[str]) -> str:
    block = match.group(0)
    if "message_kind" in block:
        return block
    if "thread_root_id" not in block:
        block = block.replace(
            "message_id=_MESSAGE_ID,",
            "thread_root_id=_MESSAGE_ID,\n        message_id=_MESSAGE_ID,\n        message_kind=MsGraphTeamsChannelMessageKind.ROOT,",
        )
        block = block.replace(
            "message_id=message_id,",
            "thread_root_id=message_id,\n        message_id=message_id,\n        message_kind=MsGraphTeamsChannelMessageKind.ROOT,",
        )
    else:
        if "message_kind" not in block:
            block = block.replace(
                "graph_base_url=_GRAPH_BASE,",
                "message_kind=MsGraphTeamsChannelMessageKind.ROOT,\n        graph_base_url=_GRAPH_BASE,",
            )
    return block

text = re.sub(
    r"validate_msgraph_teams_channel_hosted_contents_continuation\(\s*[\s\S]*?\)",
    add_message_kind,
    text,
)

p.write_text(text, encoding="utf-8")
print("patched", p)
