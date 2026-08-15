from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from intergrax.distributed.source_operation import (
    DocumentStoreSourceOperationCoordinator,
    InProcessSourceOperationCoordinator,
    SourceOperationCoordinator,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.rag.source_operation_wiring import (
    bind_source_operation_coordinator,
    shared_source_operation_coordinator,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


def test_tool_wiring_context_exposes_typed_coordinator_slot() -> None:
    field_names = {field.name for field in fields(ToolWiringContext)}
    assert "source_operation_coordinator" in field_names

    ctx = ToolWiringContext()
    assert ctx.source_operation_coordinator is None


def test_source_operation_wiring_does_not_use_extras_for_coordinator() -> None:
    root = Path(__file__).resolve().parents[5]
    source = (root / "intergrax/tools/providers/rag/source_operation_wiring.py").read_text(
        encoding="utf-8",
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "extras":
                raise AssertionError("source_operation_wiring must not access ctx.extras")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "extras":
                raise AssertionError("source_operation_wiring must not access ctx.extras")
    assert "source_operation_coordinator" not in source or "extras" not in source.split(
        "source_operation_coordinator",
        maxsplit=1,
    )[0]


def test_shared_coordinator_is_singleton_per_context() -> None:
    ctx = ToolWiringContext()

    first = shared_source_operation_coordinator(ctx)
    second = shared_source_operation_coordinator(ctx)

    assert first is second
    assert ctx.source_operation_coordinator is first
    assert isinstance(first, SourceOperationCoordinator)


def test_ingest_and_retrieve_paths_share_typed_coordinator() -> None:
    ctx = ToolWiringContext()
    scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="namespace-a",
        workspace_id="workspace-a",
    )
    manager = VectorstoreManager(object(), scope=scope)

    bind_source_operation_coordinator(ctx, manager)
    ingest_coordinator = shared_source_operation_coordinator(ctx)
    retrieve_coordinator = shared_source_operation_coordinator(ctx)

    assert ingest_coordinator is retrieve_coordinator
    assert manager._source_coordinator is ingest_coordinator


def test_document_store_coordinator_recovers_after_restart() -> None:
    document_store = InMemoryDocumentStore()
    ctx = ToolWiringContext(document_store=document_store)

    before_restart = shared_source_operation_coordinator(ctx)
    assert isinstance(before_restart, DocumentStoreSourceOperationCoordinator)

    restarted_ctx = ToolWiringContext(document_store=document_store)
    after_restart = shared_source_operation_coordinator(restarted_ctx)

    assert isinstance(after_restart, DocumentStoreSourceOperationCoordinator)
    assert after_restart is not before_restart
    assert type(after_restart) is type(before_restart)


def test_inprocess_fallback_when_document_store_missing() -> None:
    ctx = ToolWiringContext(document_store=None)
    coordinator = shared_source_operation_coordinator(ctx)
    assert isinstance(coordinator, InProcessSourceOperationCoordinator)
