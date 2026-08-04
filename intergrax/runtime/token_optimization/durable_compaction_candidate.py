# © Artur Czarnecki. All rights reserved.

"""Immutable durable-compaction candidate flow over the existing UCL repository."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from intergrax.context.session_history import SessionHistoryMessage
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCreationCoordinationStatus,
    ArtifactValidationStatus,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    ContextOptimizationReasonCode,
    DurableCompactionEligibilityDecision,
    DurableCompactionSourceIdentity,
    DurableCompactionStabilityEvidence,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    ReusableOptimizationArtifact,
    ReusableArtifactStatus,
    assess_durable_compaction_eligibility,
)
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    OptimizationArtifactReference,
    OptimizationArtifactRepository,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.serialization import (
    compute_artifact_lookup_key_hash,
    compute_durable_compaction_policy_hash,
    compute_durable_compaction_source_identity_hash,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutionResult,
    MessageSequenceArtifactExecutor,
    MessageSequenceArtifactSourceGroupProof,
    MessageSequenceArtifactExecutionRequest,
)

_MEDIA_TYPE = "application/vnd.intergrax.message-sequence-summary+json"
_ENCODING = "utf-8"
_PAYLOAD_SCHEMA_VERSION = "message_sequence_artifact.v1"
_PAYLOAD_ARTIFACT_TYPE = "message_sequence"
_REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_refs",
        "source_content_hash",
        "strategy_id",
        "strategy_version",
        "lossiness_profile",
        "summary",
    }
)


class MessageSequenceArtifactValidationError(ValueError):
    """Safe validation failure for an opaque message-sequence artifact."""

    def __init__(self) -> None:
        super().__init__("message_sequence_artifact_validation_failed")


class DurableCompactionCandidateReason(StrEnum):
    INVALID_REQUEST = "durable_compaction_invalid_request"
    ARTIFACT_PAYLOAD_INVALID = "durable_compaction_artifact_payload_invalid"
    ARTIFACT_AVAILABLE_WITHOUT_ARTIFACT = "durable_compaction_artifact_available_without_artifact"
    CREATION_IN_PROGRESS = "durable_compaction_creation_in_progress"
    RESERVATION_EXPIRED = "durable_compaction_reservation_expired"
    RESERVATION_CONFLICT = "durable_compaction_reservation_conflict"
    ARTIFACT_CREATION_FAILED = "durable_compaction_artifact_creation_failed"


class DurableCompactionCandidateError(ValueError):
    """Fail-closed, raw-content-safe candidate-flow error."""

    reason: DurableCompactionCandidateReason

    def __init__(self, reason: DurableCompactionCandidateReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _require_non_empty_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    digest = _require_non_empty_text(value, field_name)
    if len(digest) != 64 or digest != digest.lower():
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest") from exc
    return digest


def _require_tuple(value: object, field_name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be a tuple")
    return value


def _validate_message_sequence_payload(
    *,
    payload: bytes,
    media_type: str,
    encoding: str | None,
    artifact_content_hash: str,
    lookup_key: Any,
) -> tuple[str, str]:
    try:
        if type(payload) is not bytes or not payload:
            raise ValueError
        if media_type != _MEDIA_TYPE or encoding != _ENCODING:
            raise ValueError
        if compute_artifact_content_hash(payload) != artifact_content_hash:
            raise ValueError
        parsed = json.loads(payload.decode(_ENCODING))
        if not isinstance(parsed, dict) or set(parsed) != _REQUIRED_PAYLOAD_KEYS:
            raise ValueError
        if parsed["schema_version"] != _PAYLOAD_SCHEMA_VERSION:
            raise ValueError
        if parsed["artifact_type"] != _PAYLOAD_ARTIFACT_TYPE:
            raise ValueError
        source_refs = parsed["source_refs"]
        if (
            type(source_refs) is not list
            or any(type(source_ref) is not str for source_ref in source_refs)
            or tuple(source_refs) != lookup_key.source_refs
        ):
            raise ValueError
        if parsed["source_content_hash"] != lookup_key.source_content_hash:
            raise ValueError
        if parsed["strategy_id"] != lookup_key.strategy_id:
            raise ValueError
        if parsed["strategy_version"] != lookup_key.strategy_version:
            raise ValueError
        if parsed["lossiness_profile"] != lookup_key.lossiness_profile:
            raise ValueError
        summary = parsed["summary"]
        if type(summary) is not str or not summary.strip():
            raise ValueError
        return summary.strip(), artifact_content_hash
    except MessageSequenceArtifactValidationError:
        raise
    except Exception:
        raise MessageSequenceArtifactValidationError() from None


def validate_stored_message_sequence_artifact(
    stored: StoredOptimizationArtifact,
    *,
    lookup_key: Any,
) -> tuple[str, str]:
    """Validate an exact reusable artifact without exposing its opaque payload."""
    try:
        if type(stored) is not StoredOptimizationArtifact:
            raise ValueError
        metadata = stored.metadata
        if metadata.lookup_key != lookup_key:
            raise ValueError
        if metadata.status is not ReusableArtifactStatus.VALIDATED:
            raise ValueError
        if metadata.validation.status is not ArtifactValidationStatus.PASSED:
            raise ValueError
        return _validate_message_sequence_payload(
            payload=stored.payload,
            media_type=stored.media_type,
            encoding=stored.encoding,
            artifact_content_hash=metadata.artifact_content_hash,
            lookup_key=lookup_key,
        )
    except MessageSequenceArtifactValidationError:
        raise
    except Exception:
        raise MessageSequenceArtifactValidationError() from None


def validate_message_sequence_payload(
    *,
    payload: bytes,
    media_type: str,
    encoding: str | None,
    artifact_content_hash: str,
    lookup_key: Any,
) -> tuple[str, str]:
    """Validate an opaque message-sequence payload without retaining its content."""
    return _validate_message_sequence_payload(
        payload=payload,
        media_type=media_type,
        encoding=encoding,
        artifact_content_hash=artifact_content_hash,
        lookup_key=lookup_key,
    )


def validate_message_sequence_execution_result(
    result: MessageSequenceArtifactExecutionResult,
    *,
    lookup_key: Any,
) -> tuple[str, str]:
    """Validate executor output using the same payload validator as repository reuse."""
    try:
        if type(result) is not MessageSequenceArtifactExecutionResult:
            raise ValueError
        if result.validation.status is not ArtifactValidationStatus.PASSED:
            raise ValueError
        return _validate_message_sequence_payload(
            payload=result.payload,
            media_type=result.media_type,
            encoding=result.encoding,
            artifact_content_hash=result.artifact_content_hash,
            lookup_key=lookup_key,
        )
    except MessageSequenceArtifactValidationError:
        raise
    except Exception:
        raise MessageSequenceArtifactValidationError() from None


@dataclass(frozen=True, slots=True)
class CompactionInputSnapshot:
    """Immutable source snapshot used for one durable candidate operation."""

    source_identity: DurableCompactionSourceIdentity
    stability_evidence: DurableCompactionStabilityEvidence
    messages: tuple[SessionHistoryMessage, ...] = field(repr=False)
    source_group_proofs: tuple[MessageSequenceArtifactSourceGroupProof, ...] = field(
        repr=False
    )

    def __post_init__(self) -> None:
        if type(self.source_identity) is not DurableCompactionSourceIdentity:
            raise ValueError("source_identity must be DurableCompactionSourceIdentity")
        if type(self.stability_evidence) is not DurableCompactionStabilityEvidence:
            raise ValueError("stability_evidence must be DurableCompactionStabilityEvidence")
        messages = _require_tuple(self.messages, "messages")
        proofs = _require_tuple(self.source_group_proofs, "source_group_proofs")
        if not messages or not proofs:
            raise ValueError("messages and source_group_proofs must be non-empty")
        message_ids: list[str] = []
        for message in messages:
            if type(message) is not SessionHistoryMessage:
                raise ValueError("messages must contain SessionHistoryMessage values")
            message_ids.append(message.message_id)
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("messages must not contain duplicate message IDs")
        proof_refs: list[str] = []
        for proof in proofs:
            if type(proof) is not MessageSequenceArtifactSourceGroupProof:
                raise ValueError("source_group_proofs must contain source-group proofs")
            expected_group_hash = hashlib.sha256(
                "|".join(
                    f"{message.message_id}:{message.content_hash}"
                    for message in messages[
                        len(proof_refs) : len(proof_refs) + len(proof.source_refs)
                    ]
                ).encode("utf-8")
            ).hexdigest()
            if expected_group_hash != proof.source_content_hash:
                raise ValueError("source_group_proof content hash mismatch")
            proof_refs.extend(proof.source_refs)
        identity = self.source_identity
        if tuple(message_ids) != identity.source_refs or tuple(proof_refs) != identity.source_refs:
            raise ValueError("message IDs and source refs must have identical order")
        expected_source_hash = hashlib.sha256(
            "|".join(proof.source_content_hash for proof in proofs).encode("utf-8")
        ).hexdigest()
        if expected_source_hash != identity.source_content_hash:
            raise ValueError("source-group proofs do not match source content hash")
        evidence = self.stability_evidence
        if (
            evidence.observed_source_revision != identity.source_revision
            or evidence.observed_source_content_hash != identity.source_content_hash
        ):
            raise ValueError("stability evidence does not match source identity")


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    """Immutable, primary-call request for durable candidate construction."""

    operation_id: str
    policy: ContextOptimizationPolicy
    eligibility: DurableCompactionEligibilityDecision
    snapshot: CompactionInputSnapshot
    execution_guard: OptimizationExecutionGuard

    def __post_init__(self) -> None:
        _require_non_empty_text(self.operation_id, "operation_id")
        if type(self.policy) is not ContextOptimizationPolicy:
            raise ValueError("policy must be ContextOptimizationPolicy")
        if type(self.eligibility) is not DurableCompactionEligibilityDecision:
            raise ValueError(
                "eligibility must be DurableCompactionEligibilityDecision"
            )
        if type(self.snapshot) is not CompactionInputSnapshot:
            raise ValueError("snapshot must be CompactionInputSnapshot")
        if type(self.execution_guard) is not OptimizationExecutionGuard:
            raise ValueError("execution_guard must be OptimizationExecutionGuard")
        if not self.eligibility.eligible:
            raise ValueError("eligible=True is required")
        if self.policy.mode is not ContextOptimizationMode.DURABLE_COMPACTION:
            raise ValueError("policy.mode must be DURABLE_COMPACTION")
        if self.eligibility.evaluated_mode is not ContextOptimizationMode.DURABLE_COMPACTION:
            raise ValueError("eligibility.evaluated_mode must be DURABLE_COMPACTION")
        guard = self.execution_guard
        if guard.execution_scope is not ModelCallExecutionScope.PRIMARY_MODEL_CALL:
            raise ValueError("execution_guard must use PRIMARY_MODEL_CALL")
        if guard.optimization_depth != 0 or guard.parent_operation_id is not None:
            raise ValueError("execution_guard must be a depth-zero primary guard")
        if guard.operation_id != self.operation_id:
            raise ValueError("execution_guard.operation_id must match operation_id")

        durable_policy = self.policy.durable_compaction
        if durable_policy is None or not durable_policy.enabled:
            raise ValueError("enabled durable_compaction policy is required")
        if not self.policy.allow_artifact_reuse:
            raise ValueError("durable candidate flow requires artifact reuse")
        lookup_key = self.snapshot.source_identity.artifact_lookup_key
        if lookup_key.policy_version != self.policy.policy_version:
            raise ValueError("lookup key policy version mismatch")
        if lookup_key.validation_contract_version != self.policy.validation_contract_version:
            raise ValueError("lookup key validation contract version mismatch")

        policy_hash = compute_durable_compaction_policy_hash(durable_policy)
        source_hash = compute_durable_compaction_source_identity_hash(
            self.snapshot.source_identity
        )
        if self.eligibility.policy_hash != policy_hash:
            raise ValueError("eligibility policy hash mismatch")
        if self.eligibility.target_identity_hash != source_hash:
            raise ValueError("eligibility source identity hash mismatch")
        evaluated = assess_durable_compaction_eligibility(
            policy=self.policy,
            target=self.snapshot.source_identity,
            stability_evidence=self.snapshot.stability_evidence,
            raw_content_included=False,
        )
        if evaluated != self.eligibility:
            raise ValueError("eligibility is not the recomputed decision")


class CompactionCandidateStatus(StrEnum):
    REUSED_EXISTING_ARTIFACT = "reused_existing_artifact"
    CREATED_NEW_ARTIFACT = "created_new_artifact"


@dataclass(frozen=True, slots=True)
class CompactionCandidate:
    """Metadata-only, inactive durable-compaction candidate."""

    artifact_reference: OptimizationArtifactReference
    artifact_lookup_hash: str
    artifact_content_hash: str
    source_identity_hash: str
    durable_policy_hash: str
    status: CompactionCandidateStatus
    validation_status: ArtifactValidationStatus
    active: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if type(self.artifact_reference) is not OptimizationArtifactReference:
            raise ValueError("artifact_reference must be OptimizationArtifactReference")
        _require_sha256(self.artifact_lookup_hash, "artifact_lookup_hash")
        _require_sha256(self.artifact_content_hash, "artifact_content_hash")
        _require_sha256(self.source_identity_hash, "source_identity_hash")
        _require_sha256(self.durable_policy_hash, "durable_policy_hash")
        if type(self.status) is not CompactionCandidateStatus:
            raise ValueError("status must be CompactionCandidateStatus")
        if self.validation_status is not ArtifactValidationStatus.PASSED:
            raise ValueError("candidate validation_status must be PASSED")
        if self.artifact_reference.artifact_type is not OptimizationArtifactType.MESSAGE_SEQUENCE:
            raise ValueError("candidate artifact type must be MESSAGE_SEQUENCE")
        if self.artifact_reference.artifact_lookup_key_hash != self.artifact_lookup_hash:
            raise ValueError("candidate lookup hash mismatch")
        if self.artifact_reference.artifact_content_hash != self.artifact_content_hash:
            raise ValueError("candidate content hash mismatch")
        if self.active is not False:
            raise ValueError("candidate active must be False")
        if self.raw_content_included is not False:
            raise ValueError("candidate raw_content_included must be False")


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Raw-content-safe outcome of a candidate lookup or single-flight attempt."""

    reused: bool
    created: bool
    llm_invoked: bool
    coordination_status: ArtifactCreationCoordinationStatus | None
    candidate: CompactionCandidate | None
    active_revision_changed: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if type(self.reused) is not bool or type(self.created) is not bool:
            raise ValueError("reused and created must be booleans")
        if type(self.llm_invoked) is not bool:
            raise ValueError("llm_invoked must be a boolean")
        if self.coordination_status is not None and type(
            self.coordination_status
        ) is not ArtifactCreationCoordinationStatus:
            raise ValueError("coordination_status must be ArtifactCreationCoordinationStatus")
        if self.candidate is not None and type(self.candidate) is not CompactionCandidate:
            raise ValueError("candidate must be CompactionCandidate or None")
        if self.active_revision_changed is not False:
            raise ValueError("active_revision_changed must be False")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")
        if self.reused and self.created:
            raise ValueError("reused and created cannot both be True")
        if self.reused and self.llm_invoked:
            raise ValueError("reused result cannot invoke the LLM")
        if self.created and not self.llm_invoked:
            raise ValueError("created result must invoke the LLM")
        if not self.created and self.llm_invoked:
            raise ValueError("non-created result cannot invoke the LLM")

        status = self.coordination_status
        if self.reused:
            if self.candidate is None:
                raise ValueError("reused result requires candidate")
            if self.candidate.status is not CompactionCandidateStatus.REUSED_EXISTING_ARTIFACT:
                raise ValueError("reused result requires reused candidate")
            if status not in (
                None,
                ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
                ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS,
            ):
                raise ValueError("invalid coordination status for reused result")
        elif self.created:
            if self.candidate is None:
                raise ValueError("created result requires candidate")
            if self.candidate.status is not CompactionCandidateStatus.CREATED_NEW_ARTIFACT:
                raise ValueError("created result requires created candidate")
            if status is not ArtifactCreationCoordinationStatus.ACQUIRED:
                raise ValueError("created result requires ACQUIRED coordination")
        else:
            if self.candidate is not None:
                raise ValueError("non-success result cannot contain candidate")
            if status is not ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS:
                raise ValueError("candidate-less result requires ALREADY_IN_PROGRESS")


class DurableCompactionCandidateBuilder:
    """Build durable candidates with reuse-before-create and repository single-flight."""

    def __init__(
        self,
        *,
        repository: OptimizationArtifactRepository,
        message_sequence_executor: MessageSequenceArtifactExecutor,
        artifact_id_factory: Callable[[], str],
        wait_timeout_seconds: float,
    ) -> None:
        if not isinstance(repository, OptimizationArtifactRepository):
            raise ValueError("repository must be OptimizationArtifactRepository")
        if not isinstance(message_sequence_executor, MessageSequenceArtifactExecutor):
            raise ValueError(
                "message_sequence_executor must be MessageSequenceArtifactExecutor"
            )
        if not callable(artifact_id_factory):
            raise TypeError("artifact_id_factory must be callable")
        if (
            isinstance(wait_timeout_seconds, bool)
            or not isinstance(wait_timeout_seconds, (int, float))
            or not math.isfinite(float(wait_timeout_seconds))
            or not 0.0 <= float(wait_timeout_seconds) <= 5.0
        ):
            raise ValueError("wait_timeout_seconds must be finite and in [0, 5.0]")
        self._repository = repository
        self._executor = message_sequence_executor
        self._artifact_id_factory = artifact_id_factory
        self._wait_timeout_seconds = float(wait_timeout_seconds)

    def build(self, request: CompactionRequest) -> CompactionResult:
        """Return a reused candidate, a created candidate, or bounded in-progress status."""
        try:
            self._validate_request(request)
        except DurableCompactionCandidateError:
            raise
        except Exception:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.INVALID_REQUEST
            ) from None

        lookup_key = request.snapshot.source_identity.artifact_lookup_key
        lookup_hash = compute_artifact_lookup_key_hash(lookup_key)
        stored = self._safe_lookup(lookup_key)
        if stored is not None:
            return self._reused_result(request, stored, coordination_status=None)

        coordination = self._safe_acquire(request, lookup_key)
        status = coordination.status
        if status is ArtifactCreationCoordinationStatus.ACQUIRED:
            try:
                self._validate_coordination_identity(coordination, request, lookup_hash)
                if coordination.reservation is None:
                    raise DurableCompactionCandidateError(
                        DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
                    )

                execution_request = MessageSequenceArtifactExecutionRequest(
                    decision=ContextOptimizationDecision.CREATE_ARTIFACT,
                    coordination=coordination,
                    lookup_key=lookup_key,
                    policy=request.policy,
                    parent_guard=request.execution_guard,
                    source_messages=request.snapshot.messages,
                    source_group_proofs=request.snapshot.source_group_proofs,
                )
                execution_result = self._executor.execute(execution_request)
                validate_message_sequence_execution_result(
                    execution_result,
                    lookup_key=lookup_key,
                )
                self._validate_execution_identity(execution_result, request, lookup_hash)
                stored = self._stored_from_execution(
                    execution_result,
                    request=request,
                    lookup_key=lookup_key,
                )
                reference = self._repository.store_validated_artifact(
                    reservation=coordination.reservation,
                    artifact=stored,
                )
                self._validate_reference(
                    reference,
                    lookup_key=lookup_key,
                    expected_content_hash=stored.metadata.artifact_content_hash,
                )
                candidate = self._candidate(
                    reference=reference,
                    lookup_key=lookup_key,
                    source_identity=request.snapshot.source_identity,
                    policy=request.policy,
                    status=CompactionCandidateStatus.CREATED_NEW_ARTIFACT,
                    validation_status=stored.metadata.validation.status,
                )
                return CompactionResult(
                    reused=False,
                    created=True,
                    llm_invoked=True,
                    coordination_status=ArtifactCreationCoordinationStatus.ACQUIRED,
                    candidate=candidate,
                )
            except Exception as exc:
                self._release_after_failure(coordination)
                if isinstance(exc, DurableCompactionCandidateError):
                    raise
                raise DurableCompactionCandidateError(
                    DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
                ) from None

        self._validate_coordination_identity(coordination, request, lookup_hash)
        if status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE:
            stored = self._safe_lookup(lookup_key)
            if stored is None:
                raise DurableCompactionCandidateError(
                    DurableCompactionCandidateReason.ARTIFACT_AVAILABLE_WITHOUT_ARTIFACT
                )
            return self._reused_result(request, stored, coordination_status=status)

        if status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS:
            if coordination.reservation is None:
                return CompactionResult(
                    reused=False,
                    created=False,
                    llm_invoked=False,
                    coordination_status=status,
                    candidate=None,
                )
            try:
                self._repository.wait_for_artifact_or_reservation_change(
                    lookup_key,
                    observed_state_version=coordination.state_version,
                    timeout_seconds=self._wait_timeout_seconds,
                )
            except Exception:
                return CompactionResult(
                    reused=False,
                    created=False,
                    llm_invoked=False,
                    coordination_status=status,
                    candidate=None,
                )
            stored = self._safe_lookup(lookup_key)
            if stored is not None:
                return self._reused_result(request, stored, coordination_status=status)
            return CompactionResult(
                reused=False,
                created=False,
                llm_invoked=False,
                coordination_status=status,
                candidate=None,
            )

        if status is ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.RESERVATION_EXPIRED
            )
        if status is ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.RESERVATION_CONFLICT
            )
        if status is not ArtifactCreationCoordinationStatus.ACQUIRED:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            )

    @staticmethod
    def _validate_request(request: CompactionRequest) -> None:
        if type(request) is not CompactionRequest:
            raise ValueError("request must be CompactionRequest")
        request.__post_init__()

    def _safe_lookup(self, lookup_key: Any) -> StoredOptimizationArtifact | None:
        try:
            stored = self._repository.lookup(lookup_key)
        except Exception:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            ) from None
        if stored is not None:
            try:
                validate_stored_message_sequence_artifact(stored, lookup_key=lookup_key)
            except MessageSequenceArtifactValidationError:
                raise DurableCompactionCandidateError(
                    DurableCompactionCandidateReason.ARTIFACT_PAYLOAD_INVALID
                ) from None
        return stored

    def _safe_acquire(
        self,
        request: CompactionRequest,
        lookup_key: Any,
    ) -> ArtifactCreationCoordinationResult:
        try:
            coordination = self._repository.try_acquire_creation_reservation(
                lookup_key,
                owner_operation_id=request.operation_id,
                lease_seconds=request.policy.reservation_lease_seconds,
            )
        except Exception:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            ) from None
        if type(coordination) is not ArtifactCreationCoordinationResult:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            )
        return coordination

    @staticmethod
    def _validate_coordination_identity(
        coordination: ArtifactCreationCoordinationResult,
        request: CompactionRequest,
        lookup_hash: str,
    ) -> None:
        if coordination.artifact_lookup_key_hash != lookup_hash:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            )
        reservation = coordination.reservation
        if reservation is not None and (
            reservation.artifact_lookup_key_hash != lookup_hash
            or reservation.tenant_id != request.snapshot.source_identity.tenant_id
        ):
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            )

    def _reused_result(
        self,
        request: CompactionRequest,
        stored: StoredOptimizationArtifact,
        *,
        coordination_status: ArtifactCreationCoordinationStatus | None,
    ) -> CompactionResult:
        lookup_key = request.snapshot.source_identity.artifact_lookup_key
        lookup_hash = compute_artifact_lookup_key_hash(lookup_key)
        try:
            reference = build_optimization_artifact_reference(stored)
            self._validate_reference(
                reference,
                lookup_key=lookup_key,
                expected_content_hash=stored.metadata.artifact_content_hash,
            )
            candidate = self._candidate(
                reference=reference,
                lookup_key=lookup_key,
                source_identity=request.snapshot.source_identity,
                policy=request.policy,
                status=CompactionCandidateStatus.REUSED_EXISTING_ARTIFACT,
                validation_status=stored.metadata.validation.status,
            )
        except DurableCompactionCandidateError:
            raise
        except Exception:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_PAYLOAD_INVALID
            ) from None
        if candidate.artifact_lookup_hash != lookup_hash:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_PAYLOAD_INVALID
            )
        return CompactionResult(
            reused=True,
            created=False,
            llm_invoked=False,
            coordination_status=coordination_status,
            candidate=candidate,
        )

    @staticmethod
    def _validate_execution_identity(
        result: MessageSequenceArtifactExecutionResult,
        request: CompactionRequest,
        lookup_hash: str,
    ) -> None:
        receipt = result.receipt
        identity = request.snapshot.source_identity
        lookup_key = identity.artifact_lookup_key
        if (
            result.validation.validation_contract_version
            != lookup_key.validation_contract_version
            or receipt.artifact_lookup_key_hash != lookup_hash
            or receipt.source_content_hash != lookup_key.source_content_hash
            or receipt.strategy_id != lookup_key.strategy_id
            or receipt.strategy_version != lookup_key.strategy_version
            or receipt.source_ref_count != len(identity.source_refs)
        ):
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_PAYLOAD_INVALID
            )

    def _stored_from_execution(
        self,
        result: MessageSequenceArtifactExecutionResult,
        *,
        request: CompactionRequest,
        lookup_key: Any,
    ) -> StoredOptimizationArtifact:
        try:
            artifact_id = self._artifact_id_factory()
            if type(artifact_id) is not str or not artifact_id or artifact_id != artifact_id.strip():
                raise ValueError
            metadata = ReusableOptimizationArtifact(
                artifact_id=artifact_id,
                lookup_key=lookup_key,
                artifact_content_hash=result.artifact_content_hash,
                created_at=result.receipt.created_at,
                created_by_executor="message_sequence_artifact_executor.v1",
                validation=result.validation,
                receipt_ref=result.receipt.receipt_id,
                safe_metadata=dict(result.validation.safe_metadata),
            )
            return StoredOptimizationArtifact(
                metadata=metadata,
                payload=result.payload,
                media_type=result.media_type,
                encoding=result.encoding,
            )
        except Exception:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            ) from None

    @staticmethod
    def _validate_reference(
        reference: OptimizationArtifactReference,
        *,
        lookup_key: Any,
        expected_content_hash: str,
    ) -> None:
        expected_lookup_hash = compute_artifact_lookup_key_hash(lookup_key)
        if (
            type(reference) is not OptimizationArtifactReference
            or reference.tenant_id != lookup_key.tenant_id
            or reference.artifact_lookup_key_hash != expected_lookup_hash
            or reference.artifact_content_hash != expected_content_hash
            or reference.artifact_type is not OptimizationArtifactType.MESSAGE_SEQUENCE
        ):
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.ARTIFACT_CREATION_FAILED
            )

    @staticmethod
    def _candidate(
        *,
        reference: OptimizationArtifactReference,
        lookup_key: Any,
        source_identity: DurableCompactionSourceIdentity,
        policy: ContextOptimizationPolicy,
        status: CompactionCandidateStatus,
        validation_status: ArtifactValidationStatus,
    ) -> CompactionCandidate:
        durable_policy = policy.durable_compaction
        if durable_policy is None:
            raise DurableCompactionCandidateError(
                DurableCompactionCandidateReason.INVALID_REQUEST
            )
        return CompactionCandidate(
            artifact_reference=reference,
            artifact_lookup_hash=compute_artifact_lookup_key_hash(lookup_key),
            artifact_content_hash=reference.artifact_content_hash,
            source_identity_hash=compute_durable_compaction_source_identity_hash(
                source_identity
            ),
            durable_policy_hash=compute_durable_compaction_policy_hash(durable_policy),
            status=status,
            validation_status=validation_status,
        )

    def _release_after_failure(
        self,
        coordination: ArtifactCreationCoordinationResult,
    ) -> None:
        if coordination.reservation is None:
            return
        try:
            self._repository.release_creation_reservation(
                reservation=coordination.reservation,
                reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_FAILED,
            )
        except Exception:
            return
