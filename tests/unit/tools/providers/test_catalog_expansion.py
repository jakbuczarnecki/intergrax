# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids

pytestmark = pytest.mark.unit

T7_NEW_TOOL_IDS = frozenset(
    {
        "message_bus.list_tasks",
        "message_bus.cancel",
        "rag.list_documents",
        "rag.get_document",
        "rag.check_index_status",
        "document.parse_preview",
        "metrics.query_range",
        "logs.tail",
        "eval.compare_releases",
        "cost.forecast_spend",
    }
)

T6_NEW_TOOL_IDS = frozenset(
    {
        "filesystem.list",
        "filesystem.glob",
        "filesystem.read_text",
        "filesystem.stat",
        "billing.record_usage",
        "billing.list_usage",
        "cost.get_run_budget",
        "cost.check_quota",
        "platform.delete_secret",
        "rag.rerank",
        "cache.delete",
        "cache.list_keys",
        "crm.get_account",
        "crm.list_contacts",
        "crm.list_tickets",
    }
)

T5_NEW_TOOL_IDS = frozenset(
    {
        "identity.verify_token",
        "identity.get_user",
        "identity.list_tenants",
        "harness.get_run",
        "harness.list_runs",
        "harness.get_run_cost",
        "harness.get_run_events",
        "health.check_integration",
        "health.check_profile",
        "eval.record_observation",
        "eval.list_observations",
        "eval.summarize_release",
        "security.summarize_findings",
        "platform.put_secret",
    }
)

T4_NEW_TOOL_IDS = frozenset(
    {
        "database.query",
        "database.execute",
        "records.get",
        "records.put",
        "records.delete",
        "records.query",
        "rag.delete_documents",
        "rag.describe_collection",
        "workspace.delete_file",
        "workspace.search",
        "collaboration.list_messages",
        "collaboration.get_message",
        "collaboration.list_calendar",
        "collaboration.get_user",
    }
)

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
        "issues.create_issue",
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
    assert len(registered) == 120
    assert NEW_TOOL_IDS <= registered
    assert T4_NEW_TOOL_IDS <= registered
    assert T5_NEW_TOOL_IDS <= registered
    assert T6_NEW_TOOL_IDS <= registered
    assert T7_NEW_TOOL_IDS <= registered


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
        "database",
        "records",
        "identity",
        "harness",
        "health",
        "eval",
        "filesystem",
        "billing",
        "cost",
        "crm",
    ):
        bundle = get_bundle(bundle_id)
        assert bundle.bundle_id == bundle_id
        assert bundle.tool_ids

    assert len(get_bundle("rag").tool_ids) == 9
    assert len(get_bundle("observability").tool_ids) == 6
    assert len(get_bundle("message_bus").tool_ids) == 5
    assert len(get_bundle("document").tool_ids) == 2
    assert len(get_bundle("eval").tool_ids) == 4
    assert len(get_bundle("cost").tool_ids) == 3
