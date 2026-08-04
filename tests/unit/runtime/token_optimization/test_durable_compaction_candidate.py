# © Artur Czarnecki. All rights reserved.

"""Unit proof for TOKEN-10E-2 durable compaction candidate flow."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from intergrax.context.session_history import SessionHistoryMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.context_lifecycle import (
    ArtifactCompressionTarget,
    ArtifactCreationCoordinationResult,
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    DurableCompactionPolicy,
    DurableCompactionSourceIdentity,
    DurableCompactionStabilityEvidence,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    InMemoryOptimizationArtifactRepository,
    compute_artifact_lookup_key_hash,
    assess_durable_compaction_eligibility,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    CompactionCandidateStatus,
    CompactionInputSnapshot,
    CompactionRequest,
    CompactionResult,
    DurableCompactionCandidateBuilder,
    DurableCompactionCandidateError,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutionResult,
    MessageSequenceArtifactExecutionRequest,
    MessageSequenceArtifactExecutor,
    MessageSequenceArtifactSourceGroupProof,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
_STRATEGY = "message_sequence_summarization.v1"
_STRATEGY_VERSION = "1.0.0"
_POLICY_VERSION = "durable-policy.v1"
_VALIDATION_VERSION = "validation.v1"
_SECRET = "TOKEN_10E_2_SECRET_MARKER"


def _messages(suffix: str = "") -> tuple[SessionHistoryMessage, ...]:
    return (
        SessionHistoryMessage(
            message_id=f"message-a{suffix}",
            sequence=0,
            role="user",
            content=f"first {_SECRET}{suffix}",
        ),
        SessionHistoryMessage(
            message_id=f"message-b{suffix}",
            sequence=1,
            role="assistant",
            content=f"second content{suffix}",
        ),
    )


def _group_hash(messages: tuple[SessionHistoryMessage, ...]) -> str:
    return hashlib.sha256(
        "|".join(f"{message.message_id}:{message.content_hash}" for message in messages).encode(
            "utf-8"
        )
    ).hexdigest()


def _snapshot(suffix: str = "") -> CompactionInputSnapshot:
    messages = _messages(suffix)
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
    key = ArtifactLookupKey(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash=source_hash,
        strategy_id=_STRATEGY,
        strategy_version=_STRATEGY_VERSION,
        policy_version=_POLICY_VERSION,
        validation_contract_version=_VALIDATION_VERSION,
        compression_target=ArtifactCompressionTarget(target_tokens=20),
        lossiness_profile="lossy_summary",
        source_refs=tuple(message.message_id for message in messages),
    )
    identity = DurableCompactionSourceIdentity(
        tenant_id="tenant-1",
        context_scope_id="context-1",
        source_revision=4,
        expected_active_revision=4,
        source_refs=key.source_refs,
        source_content_hash=source_hash,
        artifact_lookup_key=key,
        strategy_id=_STRATEGY,
        strategy_version=_STRATEGY_VERSION,
        lossiness_profile="lossy_summary",
    )
    return CompactionInputSnapshot(
        source_identity=identity,
        stability_evidence=DurableCompactionStabilityEvidence(
            observed_stable_revision_count=2,
            observed_source_revision=4,
            observed_source_content_hash=source_hash,
        ),
        messages=messages,
        source_group_proofs=proofs,
    )


def _policy() -> ContextOptimizationPolicy:
    return ContextOptimizationPolicy(
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


def _request(
    *,
    operation_id: str = "operation-1",
    suffix: str = "",
    snapshot: CompactionInputSnapshot | None = None,
) -> CompactionRequest:
    snapshot = snapshot or _snapshot(suffix)
    policy = _policy()
    eligibility = assess_durable_compaction_eligibility(
        policy=policy,
        target=snapshot.source_identity,
        stability_evidence=snapshot.stability_evidence,
    )
    guard = OptimizationExecutionGuard(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        operation_id=operation_id,
        parent_operation_id=None,
        optimization_depth=0,
    )
    return CompactionRequest(
        operation_id=operation_id,
        policy=policy,
        eligibility=eligibility,
        snapshot=snapshot,
        execution_guard=guard,
    )


class _SpyRepository(InMemoryOptimizationArtifactRepository):
    def __init__(self) -> None:
        super().__init__()
        self.lookup_calls = 0
        self.reservation_calls = 0
        self.wait_calls = 0
        self.store_calls = 0
        self.release_calls = 0
        self.lookup_sequence: list[Any] = []
        self.coordination: ArtifactCreationCoordinationResult | None = None
        self.store_error = False

    def lookup(self, key: ArtifactLookupKey) -> Any:
        self.lookup_calls += 1
        if self.lookup_sequence:
            return self.lookup_sequence.pop(0)
        return super().lookup(key)

    def try_acquire_creation_reservation(
        self,
        key: ArtifactLookupKey,
        *,
        owner_operation_id: str,
        lease_seconds: int,
    ) -> ArtifactCreationCoordinationResult:
        self.reservation_calls += 1
        if self.coordination is not None:
            return self.coordination
        return super().try_acquire_creation_reservation(
            key,
            owner_operation_id=owner_operation_id,
            lease_seconds=lease_seconds,
        )

    def wait_for_artifact_or_reservation_change(
        self,
        key: ArtifactLookupKey,
        *,
        observed_state_version: int,
        timeout_seconds: float,
    ) -> bool:
        self.wait_calls += 1
        return super().wait_for_artifact_or_reservation_change(
            key,
            observed_state_version=observed_state_version,
            timeout_seconds=timeout_seconds,
        )

    def store_validated_artifact(self, *, reservation: Any, artifact: Any) -> Any:
        self.store_calls += 1
        if self.store_error:
            raise RuntimeError("store failure")
        return super().store_validated_artifact(reservation=reservation, artifact=artifact)

    def release_creation_reservation(self, *, reservation: Any, reason_code: Any = None) -> bool:
        self.release_calls += 1
        return super().release_creation_reservation(
            reservation=reservation,
            reason_code=reason_code,
        )


def _executor(
    *,
    invoke_model: Any | None = None,
) -> tuple[MessageSequenceArtifactExecutor, list[int]]:
    calls = [0]

    def _invoke(call: Any) -> LLMAdapterResponse:
        calls[0] += 1
        if invoke_model is not None:
            return invoke_model(call)
        return LLMAdapterResponse(content="safe durable summary")

    return (
        MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=_invoke,
            count_tokens=lambda text: max(1, len(text.split())),
            clock=lambda: _NOW,
            operation_id_factory=lambda: "internal-operation-1",
            receipt_id_factory=lambda: "receipt-1",
        ),
        calls,
    )


def _builder(
    repository: _SpyRepository,
    *,
    invoke_model: Any | None = None,
    wait_timeout_seconds: float = 0.0,
) -> tuple[DurableCompactionCandidateBuilder, list[int]]:
    executor, calls = _executor(invoke_model=invoke_model)
    artifact_counter = [0]

    def _artifact_id_factory() -> str:
        artifact_counter[0] += 1
        return f"artifact-{artifact_counter[0]}"

    return (
        DurableCompactionCandidateBuilder(
            repository=repository,
            message_sequence_executor=executor,
            artifact_id_factory=_artifact_id_factory,
            wait_timeout_seconds=wait_timeout_seconds,
        ),
        calls,
    )


def _create(
    repository: _SpyRepository | None = None,
    *,
    suffix: str = "",
) -> tuple[_SpyRepository, CompactionRequest, CompactionResult, list[int]]:
    repository = repository or _SpyRepository()
    builder, executor_calls = _builder(repository)
    result = builder.build(_request(suffix=suffix))
    return repository, _request(suffix=suffix), result, executor_calls


def _reservation(
    request: CompactionRequest,
    *,
    reservation_id: str = "reservation-1",
) -> ArtifactCreationReservation:
    key_hash = compute_artifact_lookup_key_hash(
        request.snapshot.source_identity.artifact_lookup_key
    )
    return ArtifactCreationReservation(
        reservation_id=reservation_id,
        artifact_lookup_key_hash=key_hash,
        tenant_id="tenant-1",
        owner_operation_id=request.operation_id,
        acquired_at=_NOW,
        lease_deadline=_NOW + timedelta(seconds=60),
    )


def _coordination(
    request: CompactionRequest,
    status: ArtifactCreationCoordinationStatus,
    *,
    reference: Any = None,
) -> ArtifactCreationCoordinationResult:
    key_hash = compute_artifact_lookup_key_hash(
        request.snapshot.source_identity.artifact_lookup_key
    )
    if status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE:
        return ArtifactCreationCoordinationResult(
            status=status,
            artifact_lookup_key_hash=key_hash,
            state_version=2,
            artifact_reference=reference,
        )
    reason = {
        ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS: (
            "artifact_creation_in_progress"
        ),
        ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED: (
            "artifact_creation_lease_expired"
        ),
        ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT: (
            "artifact_creation_reservation_conflict"
        ),
    }.get(status)
    return ArtifactCreationCoordinationResult(
        status=status,
        artifact_lookup_key_hash=key_hash,
        state_version=2,
        reservation=_reservation(request),
        reason_code=__import__(
            "intergrax.runtime.context_lifecycle.contracts",
            fromlist=["ContextOptimizationReasonCode"],
        ).ContextOptimizationReasonCode(reason),
    )


def _valid_execution_result(request: CompactionRequest) -> MessageSequenceArtifactExecutionResult:
    key = request.snapshot.source_identity.artifact_lookup_key
    coordination = ArtifactCreationCoordinationResult(
        status=ArtifactCreationCoordinationStatus.ACQUIRED,
        artifact_lookup_key_hash=compute_artifact_lookup_key_hash(key),
        state_version=1,
        reservation=_reservation(request),
    )
    execution_request = MessageSequenceArtifactExecutionRequest(
        decision=__import__(
            "intergrax.runtime.context_lifecycle.contracts",
            fromlist=["ContextOptimizationDecision"],
        ).ContextOptimizationDecision.CREATE_ARTIFACT,
        coordination=coordination,
        lookup_key=key,
        policy=request.policy,
        parent_guard=request.execution_guard,
        source_messages=request.snapshot.messages,
        source_group_proofs=request.snapshot.source_group_proofs,
    )
    executor, _ = _executor()
    return executor.execute(execution_request)


class _ReturningExecutor(MessageSequenceArtifactExecutor):
    def __init__(self, result: MessageSequenceArtifactExecutionResult) -> None:
        super().__init__(
            preflight=lambda _call: None,
            invoke_model=lambda _call: LLMAdapterResponse(content="unused"),
            count_tokens=lambda _text: 1,
        )
        self.result = result
        self.calls = 0

    def execute(self, request: MessageSequenceArtifactExecutionRequest) -> Any:
        self.calls += 1
        return self.result


def test_contracts_are_immutable_and_redacted() -> None:
    request = _request()
    rendered = f"{request.snapshot!r} {request!r}"
    assert _SECRET not in rendered
    with pytest.raises((AttributeError, TypeError)):
        request.snapshot.messages = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="order|proof"):
        CompactionInputSnapshot(
            source_identity=request.snapshot.source_identity,
            stability_evidence=request.snapshot.stability_evidence,
            messages=tuple(reversed(request.snapshot.messages)),
            source_group_proofs=request.snapshot.source_group_proofs,
        )
        with pytest.raises(ValueError, match="duplicate"):
            CompactionInputSnapshot(
                source_identity=request.snapshot.source_identity,
                stability_evidence=request.snapshot.stability_evidence,
                messages=(request.snapshot.messages[0], request.snapshot.messages[0]),
                source_group_proofs=request.snapshot.source_group_proofs,
            )


def test_request_requires_eligible_primary_depth_zero_and_matching_hashes() -> None:
    snapshot = _snapshot()
    policy = _policy()
    decision = assess_durable_compaction_eligibility(
        policy=policy,
        target=snapshot.source_identity,
        stability_evidence=snapshot.stability_evidence,
    )
    with pytest.raises(ValueError, match="eligible"):
        CompactionRequest(
            operation_id="operation-1",
            policy=policy,
            eligibility=decision.__class__(
                eligible=False,
                reason_code=__import__(
                    "intergrax.runtime.context_lifecycle.contracts",
                    fromlist=["DurableCompactionEligibilityReasonCode"],
                ).DurableCompactionEligibilityReasonCode.POLICY_DISABLED,
                policy_hash=decision.policy_hash,
                target_identity_hash=decision.target_identity_hash,
                evaluated_mode=decision.evaluated_mode,
            ),
            snapshot=snapshot,
            execution_guard=_request().execution_guard,
        )
    object.__setattr__(decision, "target_identity_hash", "0" * 64)
    with pytest.raises(ValueError, match="hash"):
        CompactionRequest(
            operation_id="operation-1",
            policy=policy,
            eligibility=decision,
            snapshot=snapshot,
            execution_guard=_request().execution_guard,
        )


def test_create_persists_validated_reference_without_activation() -> None:
    repository, request, result, executor_calls = _create()
    assert result.created is True
    assert result.reused is False
    assert result.llm_invoked is True
    assert result.candidate is not None
    assert result.candidate.status is CompactionCandidateStatus.CREATED_NEW_ARTIFACT
    assert result.candidate.active is False
    assert result.raw_content_included is False
    assert result.active_revision_changed is False
    assert executor_calls == [1]
    assert repository.store_calls == 1
    assert repository.reservation_calls == 1
    assert repository.release_calls == 0
    assert request.snapshot.messages == _messages()
    assert not {"summary", "messages", "payload", "prompt"} & set(
        result.candidate.__slots__
    )
    assert _SECRET not in repr(result)


def test_lookup_reuses_before_reservation_or_executor() -> None:
    repository, request, first, executor_calls = _create()
    builder, second_calls = _builder(repository)
    second = builder.build(request)
    assert first.candidate is not None
    assert second.reused is True
    assert second.created is False
    assert second.llm_invoked is False
    assert second.coordination_status is None
    assert repository.reservation_calls == 1
    assert repository.store_calls == 1
    assert executor_calls == [1]
    assert second_calls == [0]


@pytest.mark.parametrize(
    "status",
    [
        ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED,
        ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT,
    ],
)
def test_fail_closed_coordination_statuses(status: ArtifactCreationCoordinationStatus) -> None:
    repository = _SpyRepository()
    request = _request()
    repository.coordination = _coordination(request, status)
    builder, calls = _builder(repository)
    with pytest.raises(DurableCompactionCandidateError) as exc_info:
        builder.build(request)
    assert exc_info.value.reason.value.endswith(status.value)
    assert calls == [0]


def test_artifact_available_and_in_progress_with_artifact_reuse() -> None:
    source_repo, request, created, _ = _create()
    stored = source_repo.lookup(request.snapshot.source_identity.artifact_lookup_key)
    assert stored is not None
    for status in (
        ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
        ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS,
    ):
        repository = _SpyRepository()
        repository.lookup_sequence = [None, stored]
        repository.coordination = _coordination(
            request,
            status,
            reference=created.candidate.artifact_reference if created.candidate else None,
        ) if status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE else _coordination(
            request, status
        )
        builder, calls = _builder(repository)
        result = builder.build(request)
        assert result.reused is True
        assert result.coordination_status is status
        assert calls == [0]
        assert repository.wait_calls == (1 if status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS else 0)


def test_already_in_progress_without_artifact_returns_bounded_status() -> None:
    repository = _SpyRepository()
    request = _request()
    repository.lookup_sequence = [None, None]
    repository.coordination = _coordination(
        request,
        ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS,
    )
    builder, calls = _builder(repository, wait_timeout_seconds=0.0)
    result = builder.build(request)
    assert result.reused is False
    assert result.created is False
    assert result.llm_invoked is False
    assert result.candidate is None
    assert result.coordination_status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS
    assert repository.wait_calls == 1
    assert calls == [0]


def test_result_rejects_contradictory_flags() -> None:
    with pytest.raises(ValueError, match="both"):
        CompactionResult(
            reused=True,
            created=True,
            llm_invoked=True,
            coordination_status=ArtifactCreationCoordinationStatus.ACQUIRED,
            candidate=None,
        )


@pytest.mark.parametrize("failure", ["executor", "store", "factory"])
def test_failures_after_acquired_release_reservation(failure: str) -> None:
    repository = _SpyRepository()
    if failure == "store":
        repository.store_error = True
    if failure == "factory":
        def artifact_factory() -> str:
            return "   "
    else:
        def artifact_factory() -> str:
            return "artifact-1"

    def _invoke(_call: Any) -> LLMAdapterResponse:
        if failure == "executor":
            raise RuntimeError("provider error")
        return LLMAdapterResponse(content="safe durable summary")

    executor, calls = _executor(invoke_model=_invoke)
    builder = DurableCompactionCandidateBuilder(
        repository=repository,
        message_sequence_executor=executor,
        artifact_id_factory=artifact_factory,
        wait_timeout_seconds=0.0,
    )
    with pytest.raises(DurableCompactionCandidateError) as exc_info:
        builder.build(_request())
    assert exc_info.value.reason.value == "durable_compaction_artifact_creation_failed"
    assert repository.release_calls == 1
    assert _SECRET not in str(exc_info.value)
    assert calls == [1]


@pytest.mark.parametrize("failure", ["malformed", "hash_mismatch", "validation"])
def test_invalid_executor_result_releases_reservation(failure: str) -> None:
    repository = _SpyRepository()
    request = _request()
    result = _valid_execution_result(request)
    if failure == "malformed":
        object.__setattr__(result, "payload", b"malformed")
    elif failure == "hash_mismatch":
        object.__setattr__(result, "artifact_content_hash", "0" * 64)
    else:
        failed_validation = ArtifactValidationSummary(
            status=ArtifactValidationStatus.FAILED,
            validation_contract_version=_VALIDATION_VERSION,
            validated_at=result.receipt.created_at,
        )
        object.__setattr__(result, "validation", failed_validation)
    executor = _ReturningExecutor(result)
    builder = DurableCompactionCandidateBuilder(
        repository=repository,
        message_sequence_executor=executor,
        artifact_id_factory=lambda: "artifact-1",
        wait_timeout_seconds=0.0,
    )
    with pytest.raises(DurableCompactionCandidateError) as exc_info:
        builder.build(request)
    assert exc_info.value.reason.value in {
        "durable_compaction_artifact_payload_invalid",
        "durable_compaction_artifact_creation_failed",
    }
    assert repository.release_calls == 1
    assert repository.store_calls == 0
    assert executor.calls == 1
    assert _SECRET not in str(exc_info.value)


def test_same_key_concurrency_has_one_executor_and_one_store() -> None:
    repository = _SpyRepository()
    entered = threading.Event()
    release = threading.Event()

    def _invoke(_call: Any) -> LLMAdapterResponse:
        entered.set()
        release.wait(timeout=5)
        return LLMAdapterResponse(content="single-flight summary")

    builder, calls = _builder(repository, invoke_model=_invoke, wait_timeout_seconds=0.0)
    request_a = _request(operation_id="operation-a")
    request_b = _request(operation_id="operation-b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(builder.build, request_a)
        assert entered.wait(timeout=5)
        second = pool.submit(builder.build, request_b)
        second_result = second.result(timeout=5)
        release.set()
        first_result = first.result(timeout=5)
    assert first_result.created is True
    assert second_result.coordination_status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS
    assert calls == [1]
    assert repository.store_calls == 1


def test_different_keys_can_execute_independently() -> None:
    repository = _SpyRepository()
    barrier = threading.Barrier(2)

    def _invoke(_call: Any) -> LLMAdapterResponse:
        barrier.wait(timeout=5)
        return LLMAdapterResponse(content="independent summary")

    builder, calls = _builder(repository, invoke_model=_invoke)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                builder.build,
                (_request(operation_id="operation-a", suffix="-a"), _request(operation_id="operation-b", suffix="-b")),
            )
        )
    assert all(result.created for result in results)
    assert calls == [2]
    assert repository.store_calls == 2
