# © Artur Czarnecki. All rights reserved.

"""Persistence tests for Ask Workspace runs (MVP-2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.ask_models import (
    AskCitation,
    AskError,
    AskRunStatus,
    WorkspaceAskRun,
)
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository

pytestmark = pytest.mark.unit


def _run(**overrides: object) -> WorkspaceAskRun:
    payload = {
        "run_id": "run-1",
        "tenant_id": "tenant-a",
        "workspace_id": "ws-1",
        "question": "When does the contract terminate?",
        "status": AskRunStatus.COMPLETED,
        "evidence": [
            WorkspaceSearchHitV1(
                document_id="doc-1",
                source_id="src-1",
                workspace_id="ws-1",
                source_path="C:/docs/a.txt",
                file_name="a.txt",
                score=0.9,
                snippet="Terminates on 30 June 2026.",
                metadata={"provider_vector_id": "vec-1"},
            )
        ],
        "answer": "Terminates on 30 June 2026.",
        "citations": [
            AskCitation(
                evidence_id="E1",
                document_id="doc-1",
                source_id="src-1",
                workspace_id="ws-1",
                source_path="C:/docs/a.txt",
                file_name="a.txt",
                excerpt="Terminates on 30 June 2026.",
                score=0.9,
            )
        ],
        "created_at": datetime(2026, 7, 22, 10, 0, tzinfo=UTC),
        "completed_at": datetime(2026, 7, 22, 10, 1, tzinfo=UTC),
        "error": None,
    }
    payload.update(overrides)
    return WorkspaceAskRun.model_validate(payload)


def test_ask_run_persists_and_reads_for_owning_tenant() -> None:
    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    run = _run()
    repo.put_run(run)

    loaded = repo.get_run(tenant_id="tenant-a", run_id="run-1")
    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.answer == "Terminates on 30 June 2026."
    assert loaded.citations[0].document_id == "doc-1"
    assert loaded.evidence[0].snippet.startswith("Terminates")


def test_ask_run_foreign_tenant_read_is_not_found() -> None:
    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    repo.put_run(_run())
    assert repo.get_run(tenant_id="tenant-b", run_id="run-1") is None


def test_ask_run_restart_persistence_reloads_from_same_store() -> None:
    store = InMemoryDocumentStore()
    first = WorkspaceAskRepository(store)
    first.put_run(_run())

    # Simulate process restart: dispose repository instance, keep durable store.
    del first
    second = WorkspaceAskRepository(store)
    loaded = second.get_run(tenant_id="tenant-a", run_id="run-1")
    assert loaded is not None
    assert loaded.status == AskRunStatus.COMPLETED
    assert loaded.citations


def test_ask_run_failed_state_persists_bounded_error() -> None:
    store = InMemoryDocumentStore()
    repo = WorkspaceAskRepository(store)
    repo.put_run(
        _run(
            status=AskRunStatus.FAILED,
            answer=None,
            citations=[],
            error=AskError(code="unknown_evidence_reference", message="unknown evidence reference: E99"),
        )
    )
    loaded = repo.get_run(tenant_id="tenant-a", run_id="run-1")
    assert loaded is not None
    assert loaded.status == AskRunStatus.FAILED
    assert loaded.error is not None
    assert loaded.error.code == "unknown_evidence_reference"
    assert "Traceback" not in loaded.error.message
