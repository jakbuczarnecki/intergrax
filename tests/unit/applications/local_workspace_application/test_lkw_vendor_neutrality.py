from __future__ import annotations

import ast
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
        "build_shared_slack_integration_for_host",
        "build_default_slack_integration_from_env",
        'provider_id="slack"',
        "provider_id='slack'",
        "len(rehydration) == 1",
    )

    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(marker in source for marker in forbidden), path

    host_path = paths[1]
    host_source = host_path.read_text(encoding="utf-8")
    host_tree = ast.parse(host_source)
    host_bootstrap = next(
        node
        for node in host_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_connected_source_host_bundle"
    )
    durable_bootstrap_source = ast.get_source_segment(host_source, host_bootstrap)
    assert durable_bootstrap_source is not None
    assert "slack_tenant_id" not in durable_bootstrap_source
    assert "connected_source_slack_connection_ref" not in durable_bootstrap_source
    assert 'provider_id = "slack"' not in durable_bootstrap_source
    assert "build_default_slack" not in durable_bootstrap_source
    assert "LEGACY_LOCAL_BOOTSTRAP" in host_source
