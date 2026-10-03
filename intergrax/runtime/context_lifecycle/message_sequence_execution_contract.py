# © Artur Czarnecki. All rights reserved.

"""Message-sequence artifact execution ABI (UCL / Token Optimization contract surface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.context.session_history import SessionHistoryMessage

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactLookupKey,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationDecision,
    ContextOptimizationPolicy,
    OptimizationExecutionGuard,
)
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    compute_artifact_content_hash,
)

MESSAGE_SEQUENCE_ARTIFACT_MEDIA_TYPE = (
    "application/vnd.intergrax.message-sequence-summary+json"
)
MESSAGE_SEQUENCE_ARTIFACT_ENCODING = "utf-8"

_REQUIRED_VALIDATION_METADATA_KEYS = frozenset(
    {
        "parent_operation_id",
        "internal_operation_id",
        "artifact_lookup_key_hash",
        "strategy_id",
        "source_ref_count",
        "input_tokens",
        "output_tokens",
        "target_tokens",
    }
)


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def _require_timezone_aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be datetime")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _require_strict_int(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class MessageSequenceArtifactSourceGroupProof:
    source_refs: tuple[str, ...]
    source_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_refs, tuple) or not self.source_refs:
            raise ValueError("source_refs must be a non-empty tuple")
        seen_refs: set[str] = set()
        for index, source_ref in enumerate(self.source_refs):
            if not isinstance(source_ref, str):
                raise TypeError(f"source_refs[{index}] must be str")
            stripped = source_ref.strip()
            if not stripped:
                raise ValueError(f"source_refs[{index}] must be non-empty")
            if source_ref in seen_refs:
                raise ValueError("source_refs must not contain duplicates")
            seen_refs.add(source_ref)
        _require_non_empty_str(self.source_content_hash, "source_content_hash")


@dataclass(frozen=True, slots=True)
class MessageSequenceArtifactExecutionRequest:
    decision: ContextOptimizationDecision
    coordination: ArtifactCreationCoordinationResult
    lookup_key: ArtifactLookupKey
    policy: ContextOptimizationPolicy
    parent_guard: OptimizationExecutionGuard
    source_messages: tuple[SessionHistoryMessage, ...] = field(repr=False)
    source_group_proofs: tuple[MessageSequenceArtifactSourceGroupProof, ...] = field(repr=False)

    def __post_init__(self) -> None:
        from intergrax.context.session_history import SessionHistoryMessage

        if not isinstance(self.decision, ContextOptimizationDecision):
            raise TypeError("decision must be ContextOptimizationDecision")
        if not isinstance(self.coordination, ArtifactCreationCoordinationResult):
            raise TypeError("coordination must be ArtifactCreationCoordinationResult")
        if not isinstance(self.lookup_key, ArtifactLookupKey):
            raise TypeError("lookup_key must be ArtifactLookupKey")
        if not isinstance(self.policy, ContextOptimizationPolicy):
            raise TypeError("policy must be ContextOptimizationPolicy")
        if not isinstance(self.parent_guard, OptimizationExecutionGuard):
            raise TypeError("parent_guard must be OptimizationExecutionGuard")
        if not isinstance(self.source_messages, tuple) or not self.source_messages:
            raise ValueError("source_messages must be a non-empty tuple")
        seen_ids: set[str] = set()
        for index, message in enumerate(self.source_messages):
            if not isinstance(message, SessionHistoryMessage):
                raise TypeError(f"source_messages[{index}] must be SessionHistoryMessage")
            if message.message_id in seen_ids:
                raise ValueError("source_messages message IDs must not contain duplicates")
            seen_ids.add(message.message_id)
        if (
            not isinstance(self.source_group_proofs, tuple)
            or not self.source_group_proofs
        ):
            raise ValueError("source_group_proofs must be a non-empty tuple")
        seen_proof_refs: set[str] = set()
        for index, proof in enumerate(self.source_group_proofs):
            if not isinstance(proof, MessageSequenceArtifactSourceGroupProof):
                raise TypeError(
                    f"source_group_proofs[{index}] must be MessageSequenceArtifactSourceGroupProof"
                )
            for source_ref in proof.source_refs:
                if source_ref in seen_proof_refs:
                    raise ValueError("source ref cannot appear in multiple group proofs")
                seen_proof_refs.add(source_ref)
        flattened_source_refs = tuple(
            source_ref
            for proof in self.source_group_proofs
            for source_ref in proof.source_refs
        )
        if flattened_source_refs != tuple(message.message_id for message in self.source_messages):
            raise ValueError("source_group_proofs refs must match source_messages")


@dataclass(frozen=True, slots=True)
class MessageSequenceArtifactExecutionReceipt:
    receipt_id: str
    parent_operation_id: str
    internal_operation_id: str
    artifact_lookup_key_hash: str
    strategy_id: str
    strategy_version: str
    source_content_hash: str
    source_ref_count: int
    input_tokens: int
    output_tokens: int
    target_tokens: int
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _require_non_empty_str(self.receipt_id, "receipt_id"))
        object.__setattr__(
            self,
            "parent_operation_id",
            _require_non_empty_str(self.parent_operation_id, "parent_operation_id"),
        )
        object.__setattr__(
            self,
            "internal_operation_id",
            _require_non_empty_str(self.internal_operation_id, "internal_operation_id"),
        )
        object.__setattr__(
            self,
            "artifact_lookup_key_hash",
            _require_non_empty_str(self.artifact_lookup_key_hash, "artifact_lookup_key_hash"),
        )
        object.__setattr__(self, "strategy_id", _require_non_empty_str(self.strategy_id, "strategy_id"))
        object.__setattr__(
            self,
            "strategy_version",
            _require_non_empty_str(self.strategy_version, "strategy_version"),
        )
        object.__setattr__(
            self,
            "source_content_hash",
            _require_non_empty_str(self.source_content_hash, "source_content_hash"),
        )
        if self.internal_operation_id == self.parent_operation_id:
            raise ValueError("internal_operation_id must differ from parent_operation_id")
        object.__setattr__(
            self,
            "source_ref_count",
            _require_strict_int(self.source_ref_count, "source_ref_count", minimum=1),
        )
        object.__setattr__(
            self,
            "input_tokens",
            _require_strict_int(self.input_tokens, "input_tokens", minimum=0),
        )
        object.__setattr__(
            self,
            "output_tokens",
            _require_strict_int(self.output_tokens, "output_tokens", minimum=0),
        )
        object.__setattr__(
            self,
            "target_tokens",
            _require_strict_int(self.target_tokens, "target_tokens", minimum=1),
        )
        if self.output_tokens > self.target_tokens:
            raise ValueError("output_tokens must not exceed target_tokens")
        _require_timezone_aware(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class MessageSequenceArtifactExecutionResult:
    payload: bytes = field(repr=False)
    media_type: str
    encoding: str
    artifact_content_hash: str
    validation: ArtifactValidationSummary
    receipt: MessageSequenceArtifactExecutionReceipt
    internal_guard: OptimizationExecutionGuard

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or not self.payload:
            raise ValueError("payload must be non-empty bytes")
        if self.media_type != MESSAGE_SEQUENCE_ARTIFACT_MEDIA_TYPE:
            raise ValueError(f"media_type must be {MESSAGE_SEQUENCE_ARTIFACT_MEDIA_TYPE}")
        if self.encoding != MESSAGE_SEQUENCE_ARTIFACT_ENCODING:
            raise ValueError(f"encoding must be {MESSAGE_SEQUENCE_ARTIFACT_ENCODING}")
        computed_hash = compute_artifact_content_hash(self.payload)
        if self.artifact_content_hash != computed_hash:
            raise ValueError("artifact_content_hash must match SHA-256 of payload")
        if not isinstance(self.validation, ArtifactValidationSummary):
            raise TypeError("validation must be ArtifactValidationSummary")
        if self.validation.status is not ArtifactValidationStatus.PASSED:
            raise ValueError("validation.status must be PASSED")
        if not isinstance(self.receipt, MessageSequenceArtifactExecutionReceipt):
            raise TypeError("receipt must be MessageSequenceArtifactExecutionReceipt")
        if not isinstance(self.internal_guard, OptimizationExecutionGuard):
            raise TypeError("internal_guard must be OptimizationExecutionGuard")
        lookup_hash = self.receipt.artifact_lookup_key_hash
        parent_id = self.receipt.parent_operation_id
        internal_id = self.receipt.internal_operation_id
        if self.internal_guard.parent_operation_id != parent_id:
            raise ValueError("internal_guard.parent_operation_id must match receipt.parent_operation_id")
        if self.internal_guard.operation_id != internal_id:
            raise ValueError("internal_guard.operation_id must match receipt.internal_operation_id")
        metadata = self.validation.safe_metadata
        if lookup_hash not in self.internal_guard.active_artifact_lookup_key_hashes:
            raise ValueError("internal_guard must contain receipt artifact_lookup_key_hash")
        strategy_id = self.receipt.strategy_id
        if strategy_id not in self.internal_guard.active_strategy_ids:
            raise ValueError("internal_guard must contain receipt strategy_id")
        if self.validation.validated_at != self.receipt.created_at:
            raise ValueError("validation.validated_at must equal receipt.created_at")
        if set(metadata) != _REQUIRED_VALIDATION_METADATA_KEYS:
            raise ValueError("validation.safe_metadata keys must match required set exactly")
        if metadata.get("parent_operation_id") != parent_id:
            raise ValueError("validation.safe_metadata parent_operation_id mismatch")
        if metadata.get("internal_operation_id") != internal_id:
            raise ValueError("validation.safe_metadata internal_operation_id mismatch")
        if metadata.get("artifact_lookup_key_hash") != lookup_hash:
            raise ValueError("validation.safe_metadata artifact_lookup_key_hash mismatch")
        if metadata.get("strategy_id") != strategy_id:
            raise ValueError("validation.safe_metadata strategy_id mismatch")
        if metadata.get("source_ref_count") != self.receipt.source_ref_count:
            raise ValueError("validation.safe_metadata source_ref_count mismatch")
        if metadata.get("input_tokens") != self.receipt.input_tokens:
            raise ValueError("validation.safe_metadata input_tokens mismatch")
        if metadata.get("output_tokens") != self.receipt.output_tokens:
            raise ValueError("validation.safe_metadata output_tokens mismatch")
        if metadata.get("target_tokens") != self.receipt.target_tokens:
            raise ValueError("validation.safe_metadata target_tokens mismatch")


@runtime_checkable
class MessageSequenceArtifactExecutionPort(Protocol):
    """Minimal execution surface used by UCL artifact materialization."""

    def execute(
        self,
        request: MessageSequenceArtifactExecutionRequest,
    ) -> MessageSequenceArtifactExecutionResult: ...


__all__ = [
    "MESSAGE_SEQUENCE_ARTIFACT_ENCODING",
    "MESSAGE_SEQUENCE_ARTIFACT_MEDIA_TYPE",
    "MessageSequenceArtifactExecutionPort",
    "MessageSequenceArtifactExecutionReceipt",
    "MessageSequenceArtifactExecutionRequest",
    "MessageSequenceArtifactExecutionResult",
    "MessageSequenceArtifactSourceGroupProof",
]
