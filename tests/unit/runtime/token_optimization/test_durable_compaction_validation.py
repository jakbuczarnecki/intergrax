# © Artur Czarnecki. All rights reserved.

"""TOKEN-10E-3 proof for durable candidate validation and safe compilation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    InMemoryOptimizationArtifactRepository,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    assess_durable_compaction_eligibility,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationStatus,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    CompactionCandidateStatus,
    CompactionInputSnapshot,
    CompactionRequest,
    DurableCompactionCandidateBuilder,
)
from intergrax.runtime.token_optimization.durable_compaction_validation import (
    DurableCompactionValidationCompiler,
    DurableCompactionValidationError,
    DurableCompactionValidationReason,
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
_PATH = "/srv/intergrax/config.yaml"
_URL = "https://example.com/incident"
_HASH = "abcdef0123456789abcdef0123456789"


def _messages(*contents: str) -> tuple[SessionHistoryMessage, ...]:
    return tuple(
        SessionHistoryMessage(
            message_id=f"message-{index}",
            sequence=index,
            role="user",
            content=content,
        )
        for index, content in enumerate(contents)
    )


def _group_hash(messages: tuple[SessionHistoryMessage, ...]) -> str:
    return hashlib.sha256(
        "|".join(f"{message.message_id}:{message.content_hash}" for message in messages).encode(
            "utf-8"
        )
    ).hexdigest()


def _request(
    messages: tuple[SessionHistoryMessage, ...],
    *,
    summary: str,
) -> tuple[InMemoryOptimizationArtifactRepository, CompactionRequest, Any]:
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
    key = ArtifactLookupKey(
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
        artifact_lookup_key=key,
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
        ),
    )
    eligibility = assess_durable_compaction_eligibility(
        policy=policy,
        target=identity,
        stability_evidence=snapshot.stability_evidence,
    )
    request = CompactionRequest(
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
    executor = MessageSequenceArtifactExecutor(
        preflight=lambda _: None,
        invoke_model=lambda _: LLMAdapterResponse(content=summary),
        count_tokens=lambda text: len(text.split()),
        clock=lambda: _NOW,
        operation_id_factory=lambda: "internal-operation-1",
        receipt_id_factory=lambda: "internal-receipt-1",
    )
    repository = InMemoryOptimizationArtifactRepository()
    result = DurableCompactionCandidateBuilder(
        repository=repository,
        message_sequence_executor=executor,
        artifact_id_factory=lambda: "artifact-1",
        wait_timeout_seconds=0.0,
    ).build(request)
    return repository, request, result


def _compiler(repository: InMemoryOptimizationArtifactRepository) -> DurableCompactionValidationCompiler:
    return DurableCompactionValidationCompiler(
        repository=repository,
        clock=lambda: _NOW,
        receipt_id_factory=lambda: "durable-receipt-1",
    )


def _validation_request(
    request: CompactionRequest,
    result: Any,
    *,
    regions: tuple[ProtectedRegion, ...] | None = None,
    prior: Any = None,
    rollback_source_reference: str = "revision://tenant-1/context-1/revision-3",
) -> DurableCompactionValidationRequest:
    return DurableCompactionValidationRequest(
        compaction_request=request,
        compaction_result=result,
        rollback_source_reference=rollback_source_reference,
        prior_artifact_reference=prior,
        protected_regions=regions,
    )


def test_passed_outcome_revalidates_payload_and_compiles_safe_metadata() -> None:
    source = f"run curl {_URL} using {_PATH} hash {_HASH}"
    summary = f"incident preserved: {_URL} {_PATH} {_HASH}"
    repository, request, result = _request(_messages(source), summary=summary)

    outcome = _compiler(repository).compile(_validation_request(request, result))

    assert outcome.status is DurableCompactionValidationStatus.PASSED
    assert outcome.protected_region_validation.status is ProtectedRegionValidationStatus.PASSED
    assert outcome.rollback_metadata is not None
    assert outcome.activation_requirements is not None
    assert outcome.activation_requirements.candidate_artifact_id == (
        outcome.activation_requirements.validated_artifact_id
    )
    assert outcome.receipt.validation_passed is True
    assert outcome.receipt.original_chars == len(source)
    assert outcome.receipt.candidate_chars == len(summary)
    assert outcome.receipt.saved_chars == len(source) - len(summary)
    assert outcome.receipt.input_tokens is not None
    assert outcome.receipt.saved_tokens == (
        outcome.receipt.input_tokens - outcome.receipt.output_tokens
    )
    assert outcome.candidate.active is False
    assert outcome.raw_content_included is False
    assert source not in repr(outcome)
    assert summary not in repr(outcome)


def test_missing_protected_value_rejects_without_raw_failure_details() -> None:
    source = f"keep {_URL} and {_PATH}"
    summary = f"keep {_URL}"
    repository, request, result = _request(_messages(source), summary=summary)

    outcome = _compiler(repository).compile(_validation_request(request, result))

    assert outcome.status is DurableCompactionValidationStatus.REJECTED
    assert outcome.activation_requirements is None
    assert outcome.rollback_metadata is None
    assert outcome.receipt.validation_passed is False
    assert outcome.receipt.regions_failed > 0
    assert _PATH not in repr(outcome)
    assert _PATH not in str(outcome.protected_region_validation.failures)


def test_explicit_regions_extend_detection_and_deduplicate() -> None:
    source = f"keep {_URL}"
    summary = f"keep {_URL} and EXACT_ERROR"
    explicit = (
        ProtectedRegion(ProtectedRegionKind.EXACT_ERROR, "EXACT_ERROR"),
        ProtectedRegion(ProtectedRegionKind.URL, _URL),
    )
    repository, request, result = _request(_messages(source), summary=summary)

    outcome = _compiler(repository).compile(
        _validation_request(request, result, regions=explicit)
    )

    assert outcome.status is DurableCompactionValidationStatus.PASSED
    assert outcome.receipt.regions_checked == 2
    assert outcome.receipt.regions_failed == 0


def test_reuse_and_create_attribution_is_truthful() -> None:
    source = "plain source"
    repository, request, created = _request(_messages(source), summary="plain summary")
    compiler = _compiler(repository)

    first = compiler.compile(_validation_request(request, created))
    reused_candidate = replace(
        created.candidate,
        status=CompactionCandidateStatus.REUSED_EXISTING_ARTIFACT,
    )
    reused = replace(
        created,
        reused=True,
        created=False,
        llm_invoked=False,
        candidate=reused_candidate,
        coordination_status=None,
    )
    second = compiler.compile(_validation_request(request, reused))

    assert first.receipt.created_new_artifact is True
    assert first.receipt.reused_existing_artifact is False
    assert first.receipt.llm_invoked is True
    assert second.receipt.reused_existing_artifact is True
    assert second.receipt.created_new_artifact is False
    assert second.receipt.llm_invoked is False
    assert first.receipt.invalidated_prior_artifact is False


def test_missing_artifact_is_typed_and_prior_reference_is_scoped() -> None:
    repository, request, result = _request(_messages("plain source"), summary="plain summary")
    assert result.candidate is not None
    missing = replace(
        result.candidate.artifact_reference,
        artifact_id="missing-artifact",
    )
    compiler = _compiler(repository)
    with pytest.raises(DurableCompactionValidationError) as exc_info:
        compiler.compile(
            _validation_request(
                request,
                replace(
                    result,
                    candidate=replace(result.candidate, artifact_reference=missing),
                ),
            )
        )
    assert exc_info.value.reason is DurableCompactionValidationReason.CANDIDATE_ARTIFACT_NOT_FOUND

    with pytest.raises(ValueError, match="rollback_source_reference"):
        _validation_request(request, result, rollback_source_reference=" ")


def test_rollback_and_receipt_contracts_are_immutable_and_strict() -> None:
    repository, request, result = _request(_messages("plain source"), summary="plain summary")
    outcome = _compiler(repository).compile(_validation_request(request, result))
    assert outcome.rollback_metadata is not None

    with pytest.raises((AttributeError, TypeError)):
        outcome.receipt.receipt_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        replace(outcome.receipt, invalidated_prior_artifact=True)
    with pytest.raises(ValueError):
        replace(outcome.receipt, created_at=datetime(2026, 8, 4, 12, 0))
    with pytest.raises((AttributeError, TypeError)):
        outcome.rollback_metadata.rollback_source_reference = "changed"  # type: ignore[misc]
    assert "metadata" not in outcome.receipt.__slots__
    assert "summary" not in outcome.receipt.__slots__


def test_new_module_has_no_runtime_activation_or_execution_calls() -> None:
    module_path = (
        Path(__file__).resolve().parents[4]
        / "intergrax"
        / "runtime"
        / "token_optimization"
        / "durable_compaction_validation.py"
    )
    source = module_path.read_text(encoding="utf-8")
    ast.parse(source)
    forbidden = (
        "activate_revision",
        "ActiveContextRevisionPointer",
        "SessionContextRevision",
        "invalidate_artifact(",
        "store_validated_artifact(",
        "direct model client",
    )
    assert not any(term in source for term in forbidden)
    assert "resolve(candidate.artifact_reference)" in source
    assert "rollback" in source
