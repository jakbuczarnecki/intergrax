# © Artur Czarnecki. All rights reserved.

"""PDC-D2 R012: durable compaction round-trip and reconstruction (existing CE-02 / UCL path)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from intergrax.context.session_history import SessionHistoryMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.context_lifecycle import SQLiteOptimizationArtifactRepository
from intergrax.runtime.nexus.session.context_revision import (
    SessionContextRevisionActivationRequest,
    SessionContextRevisionActivationService,
    SessionContextRevisionActivationStatus,
    SQLiteSessionContextRevisionStore,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    CompactionCandidateStatus,
    DurableCompactionCandidateBuilder,
)
from intergrax.runtime.token_optimization.durable_compaction_validation import (
    DurableCompactionValidationCompiler,
    DurableCompactionValidationStatus,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutor,
)

# Reuse the existing TOKEN-10E fixtures (real builder, validator, SQLite repository, activation).
from tests.unit.runtime.token_optimization import test_durable_compaction_closeout as fx

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _repository(db_path: Path) -> SQLiteOptimizationArtifactRepository:
    return SQLiteOptimizationArtifactRepository(
        str(db_path),
        clock=lambda: fx._NOW,
        reservation_id_factory=lambda: "reservation-pdc-d2",
    )


def _compile(repository: SQLiteOptimizationArtifactRepository, request, result):  # type: ignore[no-untyped-def]
    return DurableCompactionValidationCompiler(
        repository=repository,
        clock=lambda: fx._NOW,
        receipt_id_factory=lambda: "durable-receipt-pdc-d2",
    ).compile(fx._validation_request(request, result))


def _builder_with_summary(
    repository: SQLiteOptimizationArtifactRepository,
    summary: str,
) -> DurableCompactionCandidateBuilder:
    def invoke_model(_: object) -> LLMAdapterResponse:
        return LLMAdapterResponse(content=summary)

    executor = MessageSequenceArtifactExecutor(
        preflight=lambda _: None,
        invoke_model=invoke_model,
        count_tokens=lambda text: max(1, len(text.split())),
        clock=lambda: fx._NOW,
        operation_id_factory=lambda: "internal-operation-pdc-d2",
        receipt_id_factory=lambda: "message-receipt-pdc-d2",
    )
    return DurableCompactionCandidateBuilder(
        repository=repository,
        message_sequence_executor=executor,
        artifact_id_factory=lambda: "artifact-pdc-d2",
        wait_timeout_seconds=0.0,
    )


def test_compaction_round_trip_survives_restart_and_reconstructs_the_source(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "pdc-d2-roundtrip.sqlite"
    request = fx._request()
    originals = fx._messages()

    repository = _repository(db_path)
    built = fx._builder(repository, [0], []).build(request)
    assert built.candidate is not None
    assert built.candidate.status is CompactionCandidateStatus.CREATED_NEW_ARTIFACT
    outcome = _compile(repository, request, built)
    assert outcome.status is DurableCompactionValidationStatus.PASSED
    repository.close()

    # Activate through the existing CAS service, then "restart" every durable store.
    lifecycle_store = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: fx._NOW)
    lifecycle_store.close()
    fx._seed_active_pointer(db_path)
    repo_a = SQLiteOptimizationArtifactRepository(str(db_path), clock=lambda: fx._NOW)
    store_a = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: fx._NOW)
    activation = SessionContextRevisionActivationRequest(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        operation_id="operation-1",
        outcome=outcome,
        expected_active_revision=4,
    )
    activated = SessionContextRevisionActivationService(
        repository=repo_a, revision_store=store_a, clock=lambda: fx._NOW
    ).activate(activation)
    assert activated.status is SessionContextRevisionActivationStatus.ACTIVATED
    repo_a.close()
    store_a.close()

    recovered_repo = SQLiteOptimizationArtifactRepository(str(db_path), clock=lambda: fx._NOW)
    recovered_store = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: fx._NOW)
    try:
        pointer = recovered_store.get_active_pointer(tenant_id="tenant-1", context_scope_id="context-1")
        assert pointer.active_artifact_id == outcome.candidate.artifact_reference.artifact_id

        # Reconstruction chain: active pointer -> artifact -> source refs -> original messages.
        artifact = recovered_repo.resolve(outcome.candidate.artifact_reference)
        assert artifact is not None
        payload = json.loads(artifact.payload.decode("utf-8"))
        assert artifact.metadata.artifact_content_hash == outcome.candidate.artifact_content_hash
        assert tuple(payload["source_refs"]) == tuple(m.message_id for m in originals)
        by_id = {m.message_id: m for m in originals}
        assert [by_id[ref].sequence for ref in payload["source_refs"]] == [0, 1]
        assert payload["source_content_hash"] == request.snapshot.source_identity.source_content_hash

        # Every protected region of the source survives verbatim in the compacted summary.
        for region in (fx._URL, fx._PATH, fx._COMMAND, fx._HASH):
            assert region in payload["summary"]
        assert outcome.protected_region_validation.regions_failed == 0

        # Rollback owner has the safe references it needs and nothing raw.
        assert outcome.rollback_metadata is not None
        assert outcome.rollback_metadata.rollback_source_reference.startswith("revision://")
        assert outcome.rollback_metadata.raw_content_included is False
        assert fx._SECRET not in repr((outcome, payload["summary"], pointer))
    finally:
        recovered_repo.close()
        recovered_store.close()


def test_lossy_summary_that_drops_a_protected_region_is_never_activatable(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "pdc-d2-lossy.sqlite")
    request = fx._request()
    # Adversarial summary: keeps the URL but silently drops path, command and hash.
    builder = _builder_with_summary(repository, f"user ran something against {fx._URL}")
    try:
        try:
            built = builder.build(request)
        except ValueError:
            return  # rejected while building: nothing durable can be activated
        outcome = _compile(repository, request, built)
        assert outcome.status is not DurableCompactionValidationStatus.PASSED
        assert outcome.activation_requirements is None
        assert outcome.rollback_metadata is None
        assert outcome.receipt.validation_passed is False
        assert outcome.protected_region_validation.regions_failed >= 1
    finally:
        repository.close()


def test_source_drift_after_compaction_is_rejected_not_silently_reused(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "pdc-d2-drift.sqlite")
    request = fx._request()
    try:
        fx._builder(repository, [0], []).build(request)

        original_first = request.snapshot.messages[0]
        # A message that keeps its identity but whose content changed after the source was hashed.
        tampered_first = SessionHistoryMessage(
            message_id=original_first.message_id,
            sequence=original_first.sequence,
            role=original_first.role,
            content=original_first.content + " tampered-after-compaction",
        )
        # The snapshot proves per-message content hashes; drifted content cannot even be admitted.
        with pytest.raises(ValueError, match="source_group_proof content hash mismatch"):
            replace(
                request.snapshot,
                messages=(tampered_first, *request.snapshot.messages[1:]),
            )
    finally:
        repository.close()


def test_corrupted_durable_payload_is_never_served_as_a_valid_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "pdc-d2-corrupt.sqlite"
    repository = _repository(db_path)
    request = fx._request()
    built = fx._builder(repository, [0], []).build(request)
    assert built.candidate is not None
    reference = built.candidate.artifact_reference
    repository.close()

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE optimization_artifacts SET payload = ? WHERE artifact_id = ?",
            (b'{"summary":"forged instruction"}', reference.artifact_id),
        )
        connection.commit()

    reopened = _repository(db_path)
    try:
        try:
            resolved = reopened.resolve(reference)
        except Exception:
            return  # fail-closed with an error is acceptable
        assert resolved is None or b"forged instruction" not in resolved.payload
    finally:
        reopened.close()
