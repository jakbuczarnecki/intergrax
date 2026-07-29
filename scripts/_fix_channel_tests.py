# Fix generated channel test files.
from __future__ import annotations

import re
from pathlib import Path

TD = Path(__file__).resolve().parents[1] / "tests/unit/integrations/providers/collaboration_suite"

CHANNEL_PAYLOAD = '''def _channel_payload(
    *,
    channel_id: str = _CHANNEL_ID,
    membership_type: str = "standard",
    created_at: str = _CREATED_AT_STR,
    is_archived: bool = False,
    display_name: str = _DISPLAY_NAME,
    description: str | None | object = _MISSING,
    tenant_id: str | None | object = _MISSING,
    extra_field: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": channel_id,
        "displayName": display_name,
        "membershipType": membership_type,
        "isArchived": is_archived,
        "createdDateTime": created_at,
    }
    if description is not _MISSING:
        payload["description"] = description
    if tenant_id is not _MISSING:
        payload["tenantId"] = tenant_id
    if extra_field is not None:
        payload["unknownField"] = extra_field
    return payload
'''

inv = TD / "test_ms365_graph_knowledge_teams_channel_inventory.py"
text = inv.read_text(encoding="utf-8")
text = re.sub(r"def _channel_payload\(.*?return payload\n", CHANNEL_PAYLOAD + "\n", text, count=1, flags=re.S)

for fn in [
    "test_parse_migration_in_progress",
    "test_parse_migration_completed",
    "test_parse_unknown_REMOVE_MIGRATION",
    "test_parse_migration_absent",
    "test_parse_online_meeting_info_present",
    "test_parse_online_meeting_info_absent",
    "test_parse_online_meeting_info_null",
    "test_parse_timestamps_normalized_to_utc",
    "test_parse_REMOVE_ORIGINAL_CREATED_present",
    "test_parse_is_hidden_flag",
]:
    text = re.sub(rf"def {re.escape(fn)}\(.*?\n(?=\ndef |\Z)", "", text, count=1, flags=re.S)

text = text.replace("REMOVE_LAST_UPDATED_FIELD", "createdDateTime")
text = text.replace("assert channel.REMOVE_LAST_UPDATED", "assert channel.created_at")
text = text.replace("REMOVE_LAST_UPDATED=", "created_at=")
text = text.replace('"REMOVE_LAST_UPDATED":', '"created_at":')
text = text.replace("REMOVE_LAST_UPDATED", "created_at")
text = text.replace("channel_id=_OTHER_CHANNEL_ID", "channel_id=_OTHER_CHANNEL_ID")
inv.write_text(text, encoding="utf-8")
print("fixed inventory", len(text.splitlines()))

mem = TD / "test_ms365_graph_knowledge_teams_channel_members.py"
mtext = mem.read_text(encoding="utf-8")
mtext = mtext.replace("REMOVE_REVISION", "STALE_FIELD")
mtext = re.sub(r"def test_page_stale_STALE_FIELD_rejected.*?\n(?=\ndef |\Z)", "", mtext, count=1, flags=re.S)
mtext = re.sub(r",\s*STALE_FIELD=_CHANNEL_CREATED_STR", "", mtext)
mtext = re.sub(r",\s*STALE_FIELD=_OTHER_CHANNEL_CREATED_STR", "", mtext)
mtext = re.sub(r"STALE_FIELD=_CHANNEL_CREATED_STR,\n", "", mtext)
mtext = re.sub(r"STALE_FIELD=_OTHER_CHANNEL_CREATED_STR,\n", "", mtext)
mtext = re.sub(r"\s*STALE_FIELD=_CHANNEL_CREATED_STR,\n", "", mtext)
mtext = re.sub(r"\s*STALE_FIELD=_OTHER_CHANNEL_CREATED_STR,\n", "", mtext)
mtext = re.sub(r"assert member\.STALE_FIELD.*\n", "", mtext)
mtext = re.sub(r"\{\"STALE_FIELD\": None\},?\n", "", mtext)
mtext = re.sub(r"test_page_cross_mailbox_item_rejected", "test_page_cross_team_item_rejected", mtext)
mtext = mtext.replace("team_remote_id=_OTHER_TEAM_ID", "team_remote_id=_OTHER_TEAM_ID")
mem.write_text(mtext, encoding="utf-8")
print("fixed members", len(mtext.splitlines()))
