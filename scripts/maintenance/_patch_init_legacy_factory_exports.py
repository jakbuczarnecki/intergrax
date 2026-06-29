from __future__ import annotations

import re
from pathlib import Path

from intergrax.integrations.providers.layout import SLUG_CATEGORY

UPDATES = {
    "pagerduty": "create_pagerduty_notification_channel",
    "log": "create_log_notification_channel",
    "rabbitmq": "create_rabbitmq_message_bus",
    "slash_command": "create_slash_command_interaction_surface",
    "vespa": "create_vespa_vector_store",
    "gitlab": "create_gitlab_issue_tracker",
}

for slug, legacy in UPDATES.items():
    category = SLUG_CATEGORY[slug]
    path = Path(f"intergrax/integrations/providers/{category}/{slug}/__init__.py")
    text = path.read_text(encoding="utf-8")
    block = text.split("_BUNDLE_EXPORTS", 1)[1].split(")", 1)[0]
    if legacy in block:
        print(slug, "already ok")
        continue
    text = re.sub(
        r"(_BUNDLE_EXPORTS = frozenset\(\s*\{)([^}]*)(\})",
        lambda m, l=legacy: m.group(1) + m.group(2).rstrip() + f'\n        "{l}",\n    ' + m.group(3),
        text,
        count=1,
    )
    all_block = text.split("__all__", 1)[1].split("]", 1)[0]
    if legacy not in all_block:
        text = re.sub(
            r"(__all__ = \[)([^\]]*)",
            lambda m, l=legacy: m.group(1) + m.group(2).rstrip() + f'\n    "{l}",\n',
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print("patched", slug)
