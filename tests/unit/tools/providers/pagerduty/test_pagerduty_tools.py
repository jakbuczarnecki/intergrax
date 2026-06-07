# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.tools.registry.bootstrap import register_default_tools
from intergrax.tools.registry.catalog import get_bundle, list_catalog_tool_ids


def test_pagerduty_tool_registered_in_catalog() -> None:
    register_default_tools()
    assert "pagerduty.trigger_incident" in list_catalog_tool_ids()
    assert get_bundle("pagerduty").tool_ids == (
        "pagerduty.trigger_incident",
        "pagerduty.acknowledge_incident",
    )
