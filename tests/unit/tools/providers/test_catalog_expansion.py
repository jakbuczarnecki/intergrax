# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids

pytestmark = pytest.mark.unit

NEW_TOOL_IDS = frozenset(
    {
        "workspace.write_file",
        "workspace.read_file",
        "workspace.list_files",
        "workspace.snapshot",
        "memory.read",
        "memory.write",
        "memory.list_keys",
        "knowledge.get_page",
        "knowledge.search",
        "document.parse",
        "browser.fetch_page",
        "storage.get",
        "storage.put",
        "storage.presigned_url",
        "storage.delete",
        "issues.get_issue",
        "issues.add_comment",
        "issues.search",
        "platform.get_secret",
        "platform.evaluate_feature_flag",
        "platform.get_workflow_run",
        "platform.list_check_suites",
        "message_bus.enqueue",
        "message_bus.get_status",
        "message_bus.get_result",
        "graph.run_query",
        "graph.get_node",
        "collaboration.send_mail",
        "cache.get",
        "cache.set",
    }
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_register_default_tools_expanded_catalog() -> None:
    register_default_tools()
    registered = frozenset(list_catalog_tool_ids())
    assert len(registered) == 66
    assert NEW_TOOL_IDS <= registered


def test_new_bundles_present_in_catalog() -> None:
    register_default_tools()
    for bundle_id in (
        "workspace",
        "memory",
        "knowledge",
        "document",
        "browser",
        "storage",
        "issues",
        "platform",
        "message_bus",
        "graph",
        "collaboration",
        "cache",
    ):
        bundle = get_bundle(bundle_id)
        assert bundle.bundle_id == bundle_id
        assert bundle.tool_ids
