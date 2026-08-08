from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_generic_lkw_surfaces_do_not_import_slack_runtime_directly() -> None:
    root = Path(__file__).resolve().parents[4]
    paths = (
        root / "applications/local_workspace_application/serving/workspace_routes.py",
        root
        / "applications/local_workspace_application/workspaces/connected_source_host_wiring.py",
        root
        / "applications/local_workspace_application/workspaces/knowledge_plugin_configuration_service.py",
    )
    forbidden = (
        "slack_sdk",
        "SlackConversationChannelIntegration",
        "intergrax.integrations.providers.conversation_channel.slack",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(marker in source for marker in forbidden), path
