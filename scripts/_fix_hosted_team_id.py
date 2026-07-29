# Temporary — do not commit
from pathlib import Path

p = Path(
    "tests/unit/integrations/providers/collaboration_suite/"
    "test_ms365_graph_knowledge_teams_channel_hosted_content.py"
)
lines = p.read_text(encoding="utf-8").splitlines()

out: list[str] = []
in_continuation = False
for line in lines:
    if "validate_msgraph_teams_channel_hosted_contents_continuation(" in line:
        in_continuation = True
    if in_continuation and line.strip() == ")":
        in_continuation = False
    if "team_id=_TEAM_ID" in line and not in_continuation:
        line = line.replace("team_id=_TEAM_ID", "team_remote_id=_TEAM_ID")
    out.append(line)

text = "\n".join(out) + "\n"

# Fix page/bytes defaults
text = text.replace(
    '        "message_remote_id": _MESSAGE_ID,\n'
    '        "message_revision": _REVISION,\n'
    '        "items": (_valid_hosted_content(),),',
    '        "message_remote_id": _MESSAGE_ID,\n'
    '        "thread_root_remote_id": _MESSAGE_ID,\n'
    '        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,\n'
    '        "message_revision": _REVISION,\n'
    '        "items": (_valid_hosted_content(),),',
    1,
)
text = text.replace(
    '        "message_remote_id": _MESSAGE_ID,\n'
    '        "message_revision": _REVISION,\n'
    '        "hosted_content_remote_id": _HOSTED_CONTENT_ID,',
    '        "message_remote_id": _MESSAGE_ID,\n'
    '        "thread_root_remote_id": _MESSAGE_ID,\n'
    '        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,\n'
    '        "message_revision": _REVISION,\n'
    '        "hosted_content_remote_id": _HOSTED_CONTENT_ID,',
    1,
)

p.write_text(text, encoding="utf-8")
print("fixed team_remote_id in models")
