# © Artur Czarnecki. All rights reserved.

"""TOKEN-10E-CLOSEOUT-1 end-to-end closure proof."""

from __future__ import annotations

import ast
import hashlib
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.context.session_history import SessionHistoryMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.context_lifecycle import (
    ArtifactCompressionTarget,
    ArtifactLookupKey,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    DurableCompactionPolicy,
    DurableCompactionSourceIdentity,
    DurableCompactionStabilityEvidence,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    SQLiteOptimizationArtifactRepository,
    assess_durable_compaction_eligibility,
)
from intergrax.runtime.nexus.session.context_revision import (
    SQLiteSessionContextRevisionStore,
    SessionContextRevisionActivationRequest,
    SessionContextRevisionActivationService,
    SessionContextRevisionActivationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    CompactionCandidateStatus,
    CompactionInputSnapshot,
    CompactionRequest,
    DurableCompactionCandidateBuilder,
)
from intergrax.runtime.token_optimization.durable_compaction_validation import (
    DurableCompactionValidationCompiler,
    DurableCompactionValidationRequest,
    DurableCompactionValidationStatus,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutor,
    MessageSequenceArtifactSourceGroupProof,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
_POLICY_VERSION = "durable-policy.v1"
_VALIDATION_VERSION = "validation.v1"
_STRATEGY = "message_sequence_summarization.v1"
_STRATEGY_VERSION = "1.0.0"
_URL = "https://example.com/token-10e-closeout"
_PATH = "/srv/intergrax/config.yaml"
_COMMAND = "curl --config /srv/intergrax/config.yaml"
_HASH = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
_SECRET = "TOKEN_10E_CLOSEOUT_SECRET_MARKER"


def _messages() -> tuple[SessionHistoryMessage, ...]:
    return (
        SessionHistoryMessage(
            message_id="message-a",
            sequence=0,
            role="user",
            content=f"run {_COMMAND} against {_URL} with hash {_HASH} {_SECRET}",
        ),
        SessionHistoryMessage(
            message_id="message-b",
            sequence=1,
            role="assistant",
            content="confirm the bounded durable compaction candidate",
        ),
    )


def _group_hash(messages: tuple[SessionHistoryMessage, ...]) -> str:
    return hashlib.sha256(
        "|".join(f"{message.message_id}:{message.content_hash}" for message in messages).encode(
            "utf-8"
        )
    ).hexdigest()


def _request() -> CompactionRequest:
    messages = _messages()
    proofs = tuple(
        MessageSequenceArtifactSourceGroupProof(
            source_refs=(message.message_id,),
            source_content_hash=_group_hash((message,)),
        )
        for message in messages
    )
    source_hash = hashlib.sha256(
        "|".join(proof.source_content_hash for proof in proofs).encode("utf-8")
    ).hexdigest()
    source_refs = tuple(message.message_id for message in messages)
    lookup_key = ArtifactLookupKey(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash=source_hash,
        strategy_id=_STRATEGY,
        strategy_version=_STRATEGY_VERSION,
        policy_version=_POLICY_VERSION,
        validation_contract_version=_VALIDATION_VERSION,
        compression_target=ArtifactCompressionTarget(target_tokens=40),
        lossiness_profile="lossy_summary",
        source_refs=source_refs,
    )
    identity = DurableCompactionSourceIdentity(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        source_revision=4,
        expected_active_revision=4,
        source_refs=source_refs,
        source_content_hash=source_hash,
        artifact_lookup_key=lookup_key,
        strategy_id=_STRATEGY,
        strategy_version=_STRATEGY_VERSION,
        lossiness_profile="lossy_summary",
    )
    snapshot = CompactionInputSnapshot(
        source_identity=identity,
        stability_evidence=DurableCompactionStabilityEvidence(
            observed_stable_revision_count=2,
            observed_source_revision=4,
            observed_source_content_hash=source_hash,
        ),
        messages=messages,
        source_group_proofs=proofs,
    )
    policy = ContextOptimizationPolicy(
        policy_version=_POLICY_VERSION,
        validation_contract_version=_VALIDATION_VERSION,
        enabled=True,
        mode=ContextOptimizationMode.DURABLE_COMPACTION,
        allow_lossy=True,
        allow_llm_summarization=True,
        allow_artifact_reuse=True,
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        allowed_strategy_ids=(_STRATEGY,),
        require_receipt=True,
        require_rollback_metadata=True,
        durable_compaction=DurableCompactionPolicy(
            enabled=True,
            allowed_strategy_ids=(_STRATEGY,),
            allowed_lossiness_profiles=("lossy_summary",),
            minimum_stable_revision_count=1,
        ),
    )
    eligibility = assess_durable_compaction_eligibility(
        policy=policy,
        target=identity,
        stability_evidence=snapshot.stability_evidence,
    )
    return CompactionRequest(
        operation_id="operation-1",
        policy=policy,
        eligibility=eligibility,
        snapshot=snapshot,
        execution_guard=OptimizationExecutionGuard(
            execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
            operation_id="operation-1",
            parent_operation_id=None,
            optimization_depth=0,
        ),
    )


def _executor(calls: list[int]) -> MessageSequenceArtifactExecutor:
    summary = f"preserved {_URL} {_PATH} {_COMMAND} {_HASH}"

    def invoke_model(_: object) -> LLMAdapterResponse:
        calls[0] += 1
        return LLMAdapterResponse(content=summary)

    return MessageSequenceArtifactExecutor(
        preflight=lambda _: None,
        invoke_model=invoke_model,
        count_tokens=lambda text: max(1, len(text.split())),
        clock=lambda: _NOW,
        operation_id_factory=lambda: "internal-operation-1",
        receipt_id_factory=lambda: "message-receipt-1",
    )


def _builder(
    repository: SQLiteOptimizationArtifactRepository,
    calls: list[int],
    artifact_ids: list[str],
) -> DurableCompactionCandidateBuilder:
    def artifact_id_factory() -> str:
        artifact_ids.append("artifact-1")
        return "artifact-1"

    return DurableCompactionCandidateBuilder(
        repository=repository,
        message_sequence_executor=_executor(calls),
        artifact_id_factory=artifact_id_factory,
        wait_timeout_seconds=0.0,
    )


def _validation_request(
    request: CompactionRequest,
    result: object,
) -> DurableCompactionValidationRequest:
    return DurableCompactionValidationRequest(
        compaction_request=request,
        compaction_result=result,
        rollback_source_reference="revision://tenant-1/context-1/revision-3",
        protected_regions=(
            ProtectedRegion(ProtectedRegionKind.URL, _URL),
            ProtectedRegion(ProtectedRegionKind.PATH, _PATH),
            ProtectedRegion(ProtectedRegionKind.COMMAND, _COMMAND),
            ProtectedRegion(ProtectedRegionKind.HASH, _HASH),
        ),
    )


def _seed_active_pointer(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO active_context_revision_pointers (
                tenant_id, context_scope_id, active_revision,
                active_artifact_id, updated_at, state_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("tenant-1", "context-1", 4, "prior-artifact", _NOW.isoformat(), 4),
        )
        connection.commit()


def test_token_10e_closes_policy_to_durable_cas_activation(tmp_path: Path) -> None:
    db_path = tmp_path / "token-10e-closeout.sqlite"
    request = _request()
    first_calls = [0]
    first_artifact_ids: list[str] = []
    repository = SQLiteOptimizationArtifactRepository(
        str(db_path),
        clock=lambda: _NOW,
        reservation_id_factory=lambda: "reservation-1",
    )

    first = _builder(repository, first_calls, first_artifact_ids).build(request)
    assert first.candidate is not None
    assert first.created is True
    assert first.reused is False
    assert first.candidate.status is CompactionCandidateStatus.CREATED_NEW_ARTIFACT
    assert first_calls == [1]
    assert first_artifact_ids == ["artifact-1"]

    first_outcome = DurableCompactionValidationCompiler(
        repository=repository,
        clock=lambda: _NOW,
        receipt_id_factory=lambda: "durable-receipt-1",
    ).compile(_validation_request(request, first))
    assert first_outcome.status is DurableCompactionValidationStatus.PASSED
    assert first_outcome.protected_region_validation.regions_failed == 0
    assert first_outcome.rollback_metadata is not None
    assert first_outcome.activation_requirements is not None
    assert first_outcome.receipt.raw_content_included is False
    repository.close()

    second_calls = [0]
    second_artifact_ids: list[str] = []
    reopened_repository = SQLiteOptimizationArtifactRepository(
        str(db_path),
        clock=lambda: _NOW,
        reservation_id_factory=lambda: "reservation-2",
    )
    second = _builder(reopened_repository, second_calls, second_artifact_ids).build(request)
    assert second.candidate is not None
    assert second.reused is True
    assert second.created is False
    assert second.llm_invoked is False
    assert second.candidate.status is CompactionCandidateStatus.REUSED_EXISTING_ARTIFACT
    assert second_calls == [0]
    assert second_artifact_ids == []
    assert second.candidate.artifact_reference.artifact_id == (
        first.candidate.artifact_reference.artifact_id
    )

    outcome = DurableCompactionValidationCompiler(
        repository=reopened_repository,
        clock=lambda: _NOW,
        receipt_id_factory=lambda: "durable-receipt-2",
    ).compile(_validation_request(request, second))
    assert outcome.status is DurableCompactionValidationStatus.PASSED
    assert outcome.protected_region_validation.regions_preserved >= 4
    assert outcome.receipt.reused_existing_artifact is True
    assert outcome.receipt.created_new_artifact is False
    assert outcome.receipt.llm_invoked is False
    assert outcome.receipt.invalidated_prior_artifact is False
    assert outcome.receipt.raw_content_included is False
    assert outcome.rollback_metadata is not None
    assert outcome.rollback_metadata.raw_content_included is False
    assert outcome.activation_requirements is not None
    assert outcome.activation_requirements.candidate_artifact_id == (
        outcome.candidate.artifact_reference.artifact_id
    )
    assert outcome.activation_requirements.validated_artifact_id == (
        outcome.candidate.artifact_reference.artifact_id
    )
    assert outcome.activation_requirements.creation_receipt_reference == outcome.receipt.receipt_id

    raw_outputs = repr((first, first_outcome, second, outcome))
    assert _SECRET not in raw_outputs
    assert _messages()[0].content not in raw_outputs
    assert all(value in f"preserved {_URL} {_PATH} {_COMMAND} {_HASH}" for value in (
        _URL,
        _PATH,
        _COMMAND,
        _HASH,
    ))
    reopened_repository.close()

    lifecycle_store = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: _NOW)
    lifecycle_store.close()
    _seed_active_pointer(db_path)

    durable_repository = SQLiteOptimizationArtifactRepository(str(db_path), clock=lambda: _NOW)
    revision_store = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: _NOW)
    service = SessionContextRevisionActivationService(
        repository=durable_repository,
        revision_store=revision_store,
        clock=lambda: _NOW,
    )
    activation_request = SessionContextRevisionActivationRequest(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        operation_id="operation-1",
        outcome=outcome,
        expected_active_revision=4,
    )
    activated = service.activate(activation_request)
    assert activated.status is SessionContextRevisionActivationStatus.ACTIVATED
    assert activated.active_revision == 5
    assert second_calls == [0]
    durable_repository.close()
    revision_store.close()

    recovered_repository = SQLiteOptimizationArtifactRepository(str(db_path), clock=lambda: _NOW)
    recovered_store = SQLiteSessionContextRevisionStore(str(db_path), clock=lambda: _NOW)
    recovered_service = SessionContextRevisionActivationService(
        repository=recovered_repository,
        revision_store=recovered_store,
        clock=lambda: _NOW,
    )
    recovered_artifact = recovered_repository.resolve(outcome.candidate.artifact_reference)
    recovered_pointer = recovered_store.get_active_pointer(
        tenant_id="tenant-1",
        context_scope_id="context-1",
    )
    recovered_manifest = recovered_store.get_revision(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        revision=5,
    )
    assert recovered_artifact is not None
    assert recovered_pointer.active_revision == 5
    assert recovered_pointer.active_artifact_id == outcome.candidate.artifact_reference.artifact_id
    assert recovered_manifest is not None

    replay = recovered_service.activate(activation_request)
    assert replay.status is SessionContextRevisionActivationStatus.ALREADY_ACTIVATED
    assert replay.idempotent_replay is True
    stale = recovered_service.activate(
        replace(activation_request, operation_id="operation-2")
    )
    assert stale.status is SessionContextRevisionActivationStatus.STALE_CONTEXT_REVISION
    assert recovered_store.get_revision(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        revision=6,
    ) is None
    raw_activation_outputs = repr((activated, replay, stale, recovered_pointer, recovered_manifest))
    assert _SECRET not in raw_activation_outputs
    assert _messages()[0].content not in raw_activation_outputs
    recovered_repository.close()
    recovered_store.close()


def test_token_10e_closeout_preserves_layer_ownership_and_dependency_direction() -> None:
    root = Path(__file__).resolve().parents[4]
    sources = {
        "candidate": root / "intergrax" / "runtime" / "token_optimization" / "durable_compaction_candidate.py",
        "validation": root / "intergrax" / "runtime" / "token_optimization" / "durable_compaction_validation.py",
        "activation": root / "intergrax" / "runtime" / "nexus" / "session" / "context_revision.py",
        "package": root / "intergrax" / "runtime" / "token_optimization" / "__init__.py",
    }

    def imported_names(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(alias.name for alias in node.names)
        return names

    assert "SessionContextRevisionActivationService" not in imported_names(sources["candidate"])
    assert "SessionContextRevisionActivationService" not in imported_names(sources["validation"])
    assert "MessageSequenceArtifactExecutor" not in imported_names(sources["activation"])
    package_text = sources["package"].read_text(encoding="utf-8")
    assert "SQLiteOptimizationArtifactRepository" not in package_text
    assert "SQLiteSessionContextRevisionStore" not in package_text
    assert "SessionContextRevisionActivationService" not in package_text
    for path in sources.values():
        lowered = path.read_text(encoding="utf-8").lower()
        assert "applications/" not in lowered
        assert "local_workspace" not in lowered
        assert "lkw" not in lowered
