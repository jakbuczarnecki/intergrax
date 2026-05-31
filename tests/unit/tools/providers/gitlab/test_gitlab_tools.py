# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.registry.bootstrap import register_default_tools
from intergrax.tools.registry.catalog import get_bundle, list_catalog_tool_ids


def test_gitlab_tool_registered_in_catalog() -> None:
    register_default_tools()
    assert "gitlab.create_issue" in list_catalog_tool_ids()
    assert get_bundle("gitlab").tool_ids == ("gitlab.create_issue",)
