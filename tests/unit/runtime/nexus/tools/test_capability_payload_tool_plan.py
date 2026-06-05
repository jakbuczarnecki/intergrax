# © Artur Czarnecki. All rights reserved.

"""LEG-1: capability payload → explicit tool_ids (no from_legacy on gateway path)."""

from __future__ import annotations

import warnings

import pytest

from intergrax.runtime.nexus.tools.tool_runtime import (
    capability_payload_to_tool_ids,
    tool_invocation_plan_from_capability_payload,
)
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_capability_payload_to_tool_ids_prefers_explicit_list() -> None:
    ids = capability_payload_to_tool_ids(
        {"tool_ids": [WEBSEARCH_QUERY_TOOL_ID, RAG_RETRIEVE_TOOL_ID]},
    )
    assert ids == (WEBSEARCH_QUERY_TOOL_ID, RAG_RETRIEVE_TOOL_ID)


def test_capability_payload_to_tool_ids_maps_legacy_booleans() -> None:
    ids = capability_payload_to_tool_ids({"use_rag": True, "use_websearch": False})
    assert ids == (RAG_RETRIEVE_TOOL_ID,)


def test_tool_invocation_plan_from_capability_payload_without_deprecation() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        plan = tool_invocation_plan_from_capability_payload(
            {"use_rag": True, "use_websearch": True, "use_tools": False},
        )
    assert RAG_RETRIEVE_TOOL_ID in plan.tool_ids
    assert WEBSEARCH_QUERY_TOOL_ID in plan.tool_ids
    assert plan.uses_legacy_booleans_only() is False
