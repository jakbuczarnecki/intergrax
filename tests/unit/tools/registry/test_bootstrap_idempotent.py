# © Artur Czarnecki. All rights reserved.

"""CVL-LC-3 — register_default_tools idempotency when catalog is pre-populated."""

from __future__ import annotations

import pytest

from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, is_tool_bundle_registered, list_bundle_ids

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_register_default_tools_twice_without_clearing_catalog() -> None:
    register_default_tools()
    first_ids = list_bundle_ids()
    assert first_ids
    reset_default_tools_bootstrap()
    register_default_tools()
    assert list_bundle_ids() == first_ids


def test_register_default_tools_after_partial_eval_bundle() -> None:
    register_default_tools(bundle_ids=["eval"])
    assert is_tool_bundle_registered("eval")
    reset_default_tools_bootstrap()
    register_default_tools()
    assert is_tool_bundle_registered("catalog")
