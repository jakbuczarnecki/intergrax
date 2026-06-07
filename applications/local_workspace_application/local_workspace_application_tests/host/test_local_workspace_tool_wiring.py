# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.providers.rag.index_lifecycle_service import (
    RAG_CHECK_INDEX_STATUS_TOOL_ID,
    RAG_GET_DOCUMENT_TOOL_ID,
    RAG_LIST_DOCUMENTS_TOOL_ID,
)
from local_workspace_application.host.tool_wiring import wire_local_workspace_tools

pytestmark = pytest.mark.unit


def test_lkw_base_tool_profile_includes_t7_rag_and_document_tools() -> None:
    wiring = wire_local_workspace_tools()
    enabled = set(wiring.profile.enabled)
    for tool_id in (
        "document.parse_preview",
        RAG_LIST_DOCUMENTS_TOOL_ID,
        RAG_GET_DOCUMENT_TOOL_ID,
        RAG_CHECK_INDEX_STATUS_TOOL_ID,
    ):
        assert tool_id in enabled
