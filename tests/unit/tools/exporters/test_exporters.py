# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.exporters.mcp import to_mcp_tools
from intergrax.tools.exporters.openai import to_openai_tools
from intergrax.tools.providers.rag.bundle import rag_retrieve_contract
from intergrax.tools.registry.runtime import ToolRegistry

pytestmark = pytest.mark.unit


def test_openai_exporter_matches_contract() -> None:
    contract = rag_retrieve_contract()
    schemas = to_openai_tools([contract])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "rag.retrieve"
    assert "query" in schemas[0]["function"]["parameters"]["properties"]


def test_mcp_exporter_includes_annotations() -> None:
    contract = rag_retrieve_contract()
    items = to_mcp_tools([contract])
    assert items[0]["name"] == "rag.retrieve"
    assert items[0]["annotations"]["injects_context"] is True


def test_exporters_from_registry() -> None:
    registry = ToolRegistry()
    contract = rag_retrieve_contract()
    from intergrax.tools.providers.rag.handler import RagRetrieveHandler
    from intergrax.tools.registry.wiring import ToolWiringContext

    registry.register(contract, RagRetrieveHandler(ToolWiringContext()))
    assert len(to_openai_tools(registry)) == 1
    assert len(to_mcp_tools(registry)) == 1
