# © Artur Czarnecki. All rights reserved.

"""TOKEN-10E-4 proof for Memory/Session-owned durable CAS activation."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.runtime.context_lifecycle import SQLiteOptimizationArtifactRepository
from intergrax.runtime.nexus.session.context_revision import (
    SQLiteSessionContextRevisionStore,
    SessionContextRevisionActivationError,
    SessionContextRevisionActivationRequest,
    SessionContextRevisionActivationService,
    SessionContextRevisionActivationStatus,
)
from tests.unit.runtime.token_optimization.test_durable_compaction_validation import (
    _compiler,
    _messages,
    _request,
    _validation_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


def _prepared(tmp_path: Path):
    source_repository, compaction_request, compaction_result = _request(
        _messages("source content with a protected https://example.com/value"),
        summary="safe durable summary https://example.com/value",
    )
    validation_request = _validation_request(
        compaction_request,
        compaction_result,
    )
    outcome = _compiler(source_repository).compile(validation_request)
    stored = source_repository.resolve(outcome.candidate.artifact_reference)
    assert stored is not None
    db_path = str(tmp_path / "activation.sqlite")
    repository = SQLiteOptimizationArtifactRepository(db_path, clock=lambda: _NOW)
    reservation = repository.try_acquire_creation_reservation(
        compaction_request.snapshot.source_identity.artifact_lookup_key,
        owner_operation_id="operation-1",
        lease_seconds=60,
    )
    assert reservation.reservation is not None
    repository.store_validated_artifact(reservation=reservation.reservation, artifact=stored)
    repository.close()
    schema_store = SQLiteSessionContextRevisionStore(db_path, clock=lambda: _NOW)
    schema_store.close()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO active_context_revision_pointers (
                tenant_id, context_scope_id, active_revision,
                active_artifact_id, updated_at, state_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "tenant-1",
                "context-1",
                4,
                "prior-artifact",
                _NOW.isoformat(),
                4,
            ),
        )
        connection.commit()
    durable_repository = SQLiteOptimizationArtifactRepository(db_path, clock=lambda: _NOW)
    revision_store = SQLiteSessionContextRevisionStore(db_path, clock=lambda: _NOW)
    service = SessionContextRevisionActivationService(
        repository=durable_repository,
        revision_store=revision_store,
        clock=lambda: _NOW,
    )
    request = SessionContextRevisionActivationRequest(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        operation_id="operation-1",
        outcome=outcome,
        expected_active_revision=4,
    )
    return service, revision_store, durable_repository, request


def test_passed_outcome_activates_immutable_manifest_and_pointer(tmp_path: Path) -> None:
    service, store, repository, request = _prepared(tmp_path)
    result = service.activate(request)
    assert result.status is SessionContextRevisionActivationStatus.ACTIVATED
    assert result.previous_revision == 4
    assert result.active_revision == 5
    assert result.raw_content_included is False
    pointer = store.get_active_pointer(tenant_id="tenant-1", context_scope_id="context-1")
    manifest = store.get_revision(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        revision=5,
    )
    assert pointer.active_revision == 5
    assert pointer.active_artifact_id == result.artifact_id
    assert manifest is not None
    assert manifest.active is False
    assert manifest.parent_revision == 4
    assert manifest.prior_artifact_id is None
    assert "safe durable summary" not in repr(result)
    repository.close()
    store.close()


def test_reopen_recovers_pointer_manifest_and_idempotent_replay(tmp_path: Path) -> None:
    service, store, repository, request = _prepared(tmp_path)
    first = service.activate(request)
    repository.close()
    store.close()

    db_path = str(tmp_path / "activation.sqlite")
    reopened_repository = SQLiteOptimizationArtifactRepository(db_path, clock=lambda: _NOW)
    reopened_store = SQLiteSessionContextRevisionStore(db_path, clock=lambda: _NOW)
    reopened_service = SessionContextRevisionActivationService(
        repository=reopened_repository,
        revision_store=reopened_store,
        clock=lambda: _NOW,
    )
    replay = reopened_service.activate(request)
    assert replay.status is SessionContextRevisionActivationStatus.ALREADY_ACTIVATED
    assert replay.idempotent_replay is True
    assert replay.revision_manifest_id == first.revision_manifest_id
    assert (
        reopened_store.get_active_pointer(
            tenant_id="tenant-1",
            context_scope_id="context-1",
        ).active_revision
        == 5
    )
    reopened_repository.close()
    reopened_store.close()


def test_stale_expected_revision_does_not_create_manifest_or_retry(tmp_path: Path) -> None:
    service, store, repository, request = _prepared(tmp_path)
    assert service.activate(request).status is SessionContextRevisionActivationStatus.ACTIVATED
    stale = replace(request, operation_id="operation-2")
    result = service.activate(stale)
    assert result.status is SessionContextRevisionActivationStatus.STALE_CONTEXT_REVISION
    assert result.active_revision == 5
    assert store.get_revision(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        revision=6,
    ) is None
    repository.close()
    store.close()


def test_same_operation_conflict_fails_closed(tmp_path: Path) -> None:
    service, store, repository, request = _prepared(tmp_path)
    assert service.activate(request).status is SessionContextRevisionActivationStatus.ACTIVATED
    changed_outcome = replace(
        request.outcome,
        activation_requirements=replace(
            request.outcome.activation_requirements,
            lineage_reference="lineage-conflict",
        ),
    )
    conflicting = replace(request, outcome=changed_outcome)
    with pytest.raises(SessionContextRevisionActivationError, match="OPERATION_ID_CONFLICT"):
        service.activate(conflicting)
    assert (
        store.get_active_pointer(
            tenant_id="tenant-1",
            context_scope_id="context-1",
        ).active_revision
        == 5
    )
    repository.close()
    store.close()


def test_manifest_insert_failure_rolls_back_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, store, repository, request = _prepared(tmp_path)

    def fail_insert(_: object) -> None:
        raise RuntimeError("injected manifest failure")

    monkeypatch.setattr(store, "_insert_manifest", fail_insert)
    with pytest.raises(RuntimeError, match="injected manifest failure"):
        service.activate(request)
    assert (
        store.get_active_pointer(
            tenant_id="tenant-1",
            context_scope_id="context-1",
        ).active_revision
        == 4
    )
    assert store.get_revision(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        revision=5,
    ) is None
    repository.close()
    store.close()


def test_invalid_outcome_and_tenant_scope_fail_closed(tmp_path: Path) -> None:
    service, store, repository, request = _prepared(tmp_path)
    rejected = replace(
        request.outcome,
        status=request.outcome.status.__class__.REJECTED,
        protected_region_validation=replace(
            request.outcome.protected_region_validation,
            status=request.outcome.protected_region_validation.status.__class__.FAILED,
        ),
        receipt=replace(
            request.outcome.receipt,
            protected_region_status=request.outcome.protected_region_validation.status.__class__.FAILED,
            validation_passed=False,
            rollback_metadata_present=False,
        ),
        rollback_metadata=None,
        activation_requirements=None,
    )
    with pytest.raises(SessionContextRevisionActivationError, match="VALIDATION_OUTCOME_NOT_PASSED"):
        service.activate(replace(request, outcome=rejected))
    with pytest.raises(SessionContextRevisionActivationError, match="TENANT_SCOPE_MISMATCH"):
        service.activate(replace(request, tenant_id="tenant-2"))
    repository.close()
    store.close()


def test_activation_ownership_has_no_executor_or_model_dependency() -> None:
    source = Path(
        "intergrax/runtime/nexus/session/context_revision.py"
    ).read_text(encoding="utf-8")
    assert "MessageSequenceArtifactExecutor" not in source
    assert "LLMAdapter" not in source
