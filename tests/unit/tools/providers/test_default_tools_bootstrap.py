# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, list_catalog_tool_ids

pytestmark = pytest.mark.unit

EXPECTED_TOOL_IDS = frozenset(
    {
        "rag.retrieve",
        "websearch.query",
        "jira.get_issue",
        "jira.add_comment",
        "jira.search_tasks",
        "confluence.get_page",
        "confluence.search_pages",
        "notify.send",
        "metrics.query_instant",
        "logs.search",
        "sandbox.exec",
    }
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_register_default_tools_registers_full_catalog() -> None:
    register_default_tools()
    registered = frozenset(list_catalog_tool_ids())
    assert EXPECTED_TOOL_IDS <= registered
