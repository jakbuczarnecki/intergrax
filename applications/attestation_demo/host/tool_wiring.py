# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for attestation_demo (records bundle + lab document store)."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.tools.providers.records.service import (
    RECORDS_COUNT_TOOL_ID,
    RECORDS_DELETE_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
)
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext

_RECORDS_TOOL_IDS = [
    RECORDS_GET_TOOL_ID,
    RECORDS_PUT_TOOL_ID,
    RECORDS_DELETE_TOOL_ID,
    RECORDS_QUERY_TOOL_ID,
    RECORDS_DESCRIBE_COLLECTION_TOOL_ID,
    RECORDS_COUNT_TOOL_ID,
]


def attestation_demo_tool_profile() -> ToolProfile:
    """Records bundle only — explicit enabled list survives integration profile extension."""
    return ToolProfile(
        enabled=list(_RECORDS_TOOL_IDS),
        enabled_bundles=["records"],
    )


def wire_attestation_demo_tools(
    *,
    document_store: DocumentStore | None = None,
) -> ApplicationToolWiring:
    """Lab PoC — in-memory document store backing ``records.put``."""
    store = document_store or InMemoryDocumentStore()
    profile = attestation_demo_tool_profile()
    wiring_context = ToolWiringContext(document_store=store)
    return build_application_tool_wiring(
        profile,
        wiring_context=wiring_context,
    )
