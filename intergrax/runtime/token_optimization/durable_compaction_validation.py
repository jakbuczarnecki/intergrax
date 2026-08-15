# © Artur Czarnecki. All rights reserved.

"""Fail-closed validation and safe result compilation for durable candidates."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactValidationStatus,
    DurableCompactionActivationRequirements,
)
from intergrax.runtime.context_lifecycle.repository import (
    OptimizationArtifactReference,
    OptimizationArtifactRepository,
    StoredOptimizationArtifact,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.serialization import (
    compute_artifact_lookup_key_hash,
    compute_durable_compaction_policy_hash,
    compute_durable_compaction_source_identity_hash,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    CompactionCandidate,
    CompactionCandidateStatus,
    CompactionRequest,
    CompactionResult,
    validate_stored_message_sequence_artifact,
)
from intergrax.runtime.token_optimization.protected_regions import (
    detect_protected_regions,
    validate_protected_regions,
)


class DurableCompactionValidationReason(StrEnum):
    INVALID_REQUEST = "invalid_request"
    CANDIDATE_ARTIFACT_NOT_FOUND = "candidate_artifact_not_found"
    CANDIDATE_ARTIFACT_INVALID = "candidate_artifact_invalid"
    CANDIDATE_IDENTITY_MISMATCH = "candidate_identity_mismatch"
    PROTECTED_REGION_VALIDATION_FAILED = "protected_region_validation_failed"
    ROLLBACK_METADATA_REQUIRED = "rollback_metadata_required"
    TOKEN_MEASUREMENT_INVALID = "token_measurement_invalid"
    RECEIPT_COMPILATION_FAILED = "receipt_compilation_failed"


class DurableCompactionValidationError(ValueError):
    """Typed, raw-content-safe validation failure."""

    reason: DurableCompactionValidationReason

    def __init__(self, reason: DurableCompactionValidationReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _strict_non_empty(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a strict non-empty string")
    return value


def _sha256(value: object, field_name: str) -> str:
    digest = _strict_non_empty(value, field_name)
    if len(digest) != 64 or digest != digest.lower():
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest") from exc
    return digest


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    result = _non_negative_int(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _aware_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _strict_tuple(value: object, field_name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be an immutable tuple")
    return value


@dataclass(frozen=True, slots=True)
class DurableCompactionValidationRequest:
    """Immutable input for validating one inactive durable candidate."""

    compaction_request: CompactionRequest
    compaction_result: CompactionResult
    rollback_source_reference: str
    prior_artifact_reference: OptimizationArtifactReference | None = None
    protected_regions: tuple[ProtectedRegion, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.compaction_request) is not CompactionRequest:
            raise ValueError("compaction_request must be CompactionRequest")
        if type(self.compaction_result) is not CompactionResult:
            raise ValueError("compaction_result must be CompactionResult")
        _strict_non_empty(self.rollback_source_reference, "rollback_source_reference")

        if self.prior_artifact_reference is not None:
            if type(self.prior_artifact_reference) is not OptimizationArtifactReference:
                raise ValueError(
                    "prior_artifact_reference must be OptimizationArtifactReference"
                )
            tenant_id = self.compaction_request.snapshot.source_identity.tenant_id
            if self.prior_artifact_reference.tenant_id != tenant_id:
                raise ValueError("prior_artifact_reference tenant mismatch")
            _validate_reference(self.prior_artifact_reference, "prior_artifact_reference")

        if self.protected_regions is not None:
            regions = _strict_tuple(self.protected_regions, "protected_regions")
            for region in regions:
                if type(region) is not ProtectedRegion:
                    raise ValueError("protected_regions must contain ProtectedRegion values")
                if type(region.kind) is not ProtectedRegionKind:
                    raise ValueError("protected region kind must be ProtectedRegionKind")
                _strict_non_empty(region.value, "protected region value")

        result = self.compaction_result
        candidate = result.candidate
        if (result.reused, result.created) not in ((True, False), (False, True)):
            raise ValueError("compaction result must be reused XOR created")
        if candidate is None:
            raise ValueError("compaction result must contain a candidate")
        if result.llm_invoked is not result.created:
            raise ValueError("compaction result LLM attribution is inconsistent")
        if result.active_revision_changed is not False or result.raw_content_included is not False:
            raise ValueError("compaction result must not activate or include raw content")
        if candidate.active is not False or candidate.raw_content_included is not False:
            raise ValueError("candidate must remain inactive and redaction-safe")

        source_identity = self.compaction_request.snapshot.source_identity
        lookup_key = source_identity.artifact_lookup_key
        expected_lookup_hash = compute_artifact_lookup_key_hash(lookup_key)
        expected_source_hash = compute_durable_compaction_source_identity_hash(
            source_identity
        )
        durable_policy = self.compaction_request.policy.durable_compaction
        if durable_policy is None:
            raise ValueError("durable compaction policy is required")
        expected_policy_hash = compute_durable_compaction_policy_hash(durable_policy)

        reference = candidate.artifact_reference
        _validate_reference(reference, "candidate_artifact_reference")
        if (
            reference.tenant_id != source_identity.tenant_id
            or reference.artifact_type.value != "message_sequence"
            or reference.artifact_lookup_key_hash != expected_lookup_hash
            or candidate.artifact_lookup_hash != expected_lookup_hash
            or candidate.source_identity_hash != expected_source_hash
            or candidate.durable_policy_hash != expected_policy_hash
            or reference.artifact_content_hash != candidate.artifact_content_hash
        ):
            raise ValueError("candidate identity does not match the request")
        if self.prior_artifact_reference is not None:
            if self.prior_artifact_reference.artifact_id == reference.artifact_id:
                raise ValueError("prior_artifact_reference must differ from candidate")


def _validate_reference(
    reference: OptimizationArtifactReference,
    field_name: str,
) -> None:
    _strict_non_empty(reference.tenant_id, f"{field_name}.tenant_id")
    _strict_non_empty(reference.artifact_id, f"{field_name}.artifact_id")
    _sha256(reference.artifact_lookup_key_hash, f"{field_name}.artifact_lookup_key_hash")
    _sha256(reference.artifact_content_hash, f"{field_name}.artifact_content_hash")


@dataclass(frozen=True, slots=True)
class CompactionRollbackMetadata:
    """Safe references and hashes required by a future activation/rollback owner."""

    tenant_id: str
    context_scope_id: str
    source_revision: int
    expected_active_revision: int
    rollback_source_reference: str
    candidate_artifact_reference: OptimizationArtifactReference
    prior_artifact_reference: OptimizationArtifactReference | None
    source_identity_hash: str
    candidate_artifact_content_hash: str
    compiled_at: datetime
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        _strict_non_empty(self.tenant_id, "tenant_id")
        _strict_non_empty(self.context_scope_id, "context_scope_id")
        _non_negative_int(self.source_revision, "source_revision")
        _positive_int(self.expected_active_revision, "expected_active_revision")
        _strict_non_empty(self.rollback_source_reference, "rollback_source_reference")
        if type(self.candidate_artifact_reference) is not OptimizationArtifactReference:
            raise ValueError(
                "candidate_artifact_reference must be OptimizationArtifactReference"
            )
        if self.prior_artifact_reference is not None and type(
            self.prior_artifact_reference
        ) is not OptimizationArtifactReference:
            raise ValueError(
                "prior_artifact_reference must be OptimizationArtifactReference"
            )
        if self.prior_artifact_reference is not None and (
            self.prior_artifact_reference.tenant_id != self.tenant_id
            or self.prior_artifact_reference.artifact_id
            == self.candidate_artifact_reference.artifact_id
        ):
            raise ValueError("prior_artifact_reference is not compatible")
        if self.candidate_artifact_reference.tenant_id != self.tenant_id:
            raise ValueError("candidate artifact tenant mismatch")
        _sha256(self.source_identity_hash, "source_identity_hash")
        _sha256(self.candidate_artifact_content_hash, "candidate_artifact_content_hash")
        _aware_datetime(self.compiled_at, "compiled_at")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")


@dataclass(frozen=True, slots=True)
class DurableCompactionReceipt:
    """Redaction-safe receipt for one durable candidate validation attempt."""

    receipt_id: str
    created_at: datetime
    tenant_id: str
    context_scope_id: str
    candidate_status: CompactionCandidateStatus
    artifact_id: str
    artifact_lookup_hash: str
    artifact_content_hash: str
    source_identity_hash: str
    durable_policy_hash: str
    source_revision: int
    expected_active_revision: int
    source_ref_count: int
    reused_existing_artifact: bool
    created_new_artifact: bool
    invalidated_prior_artifact: bool
    llm_invoked: bool
    protected_region_status: ProtectedRegionValidationStatus
    validation_passed: bool
    regions_checked: int
    regions_preserved: int
    regions_failed: int
    original_chars: int
    candidate_chars: int
    saved_chars: int
    input_tokens: int | None
    output_tokens: int | None
    saved_tokens: int | None
    token_measurement_available: bool
    rollback_metadata_present: bool
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        _strict_non_empty(self.receipt_id, "receipt_id")
        _aware_datetime(self.created_at, "created_at")
        _strict_non_empty(self.tenant_id, "tenant_id")
        _strict_non_empty(self.context_scope_id, "context_scope_id")
        if type(self.candidate_status) is not CompactionCandidateStatus:
            raise ValueError("candidate_status must be CompactionCandidateStatus")
        _strict_non_empty(self.artifact_id, "artifact_id")
        for value, name in (
            (self.artifact_lookup_hash, "artifact_lookup_hash"),
            (self.artifact_content_hash, "artifact_content_hash"),
            (self.source_identity_hash, "source_identity_hash"),
            (self.durable_policy_hash, "durable_policy_hash"),
        ):
            _sha256(value, name)
        _non_negative_int(self.source_revision, "source_revision")
        _positive_int(self.expected_active_revision, "expected_active_revision")
        _positive_int(self.source_ref_count, "source_ref_count")
        if type(self.protected_region_status) is not ProtectedRegionValidationStatus:
            raise ValueError(
                "protected_region_status must be ProtectedRegionValidationStatus"
            )
        for value, name in (
            (self.regions_checked, "regions_checked"),
            (self.regions_preserved, "regions_preserved"),
            (self.regions_failed, "regions_failed"),
            (self.original_chars, "original_chars"),
            (self.candidate_chars, "candidate_chars"),
        ):
            _non_negative_int(value, name)
        if self.regions_preserved > self.regions_checked or self.regions_failed > self.regions_checked:
            raise ValueError("protected-region counts are inconsistent")
        if self.saved_chars != self.original_chars - self.candidate_chars:
            raise ValueError("saved_chars must equal original_chars - candidate_chars")
        if self.invalidated_prior_artifact is not False:
            raise ValueError("invalidated_prior_artifact must be False")
        if type(self.reused_existing_artifact) is not bool or type(
            self.created_new_artifact
        ) is not bool:
            raise ValueError("reuse/create attribution must be boolean")
        if self.reused_existing_artifact == self.created_new_artifact:
            raise ValueError("receipt must identify exactly one candidate origin")
        if self.candidate_status is CompactionCandidateStatus.REUSED_EXISTING_ARTIFACT:
            if not self.reused_existing_artifact or self.llm_invoked:
                raise ValueError("reuse attribution is inconsistent")
        elif not self.created_new_artifact or not self.llm_invoked:
            raise ValueError("create attribution is inconsistent")
        if type(self.llm_invoked) is not bool or type(self.validation_passed) is not bool:
            raise ValueError("receipt boolean fields must be booleans")
        if self.validation_passed != (
            self.protected_region_status
            in (
                ProtectedRegionValidationStatus.PASSED,
                ProtectedRegionValidationStatus.NOT_APPLICABLE,
            )
        ):
            raise ValueError("validation_passed does not match protected status")
        if self.rollback_metadata_present != self.validation_passed:
            raise ValueError("rollback metadata presence does not match validation")
        if type(self.token_measurement_available) is not bool:
            raise ValueError("token_measurement_available must be a boolean")
        token_values = (self.input_tokens, self.output_tokens, self.saved_tokens)
        if not self.token_measurement_available:
            if any(value is not None for value in token_values):
                raise ValueError("unavailable token measurement must contain no values")
        else:
            if any(
                value is None
                or isinstance(value, bool)
                or type(value) is not int
                or value < 0
                for value in token_values[:2]
            ):
                raise ValueError("available token measurement has invalid counts")
            if self.saved_tokens != self.input_tokens - self.output_tokens:
                raise ValueError("saved_tokens must equal input_tokens - output_tokens")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")


class DurableCompactionValidationStatus(StrEnum):
    PASSED = "passed"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DurableCompactionValidationOutcome:
    """Immutable safe validation package for the next lifecycle stage."""

    status: DurableCompactionValidationStatus
    candidate: CompactionCandidate
    protected_region_validation: ProtectedRegionValidationResult
    receipt: DurableCompactionReceipt
    rollback_metadata: CompactionRollbackMetadata | None
    activation_requirements: DurableCompactionActivationRequirements | None
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if type(self.status) is not DurableCompactionValidationStatus:
            raise ValueError("status must be DurableCompactionValidationStatus")
        if type(self.candidate) is not CompactionCandidate:
            raise ValueError("candidate must be CompactionCandidate")
        if self.candidate.active is not False or self.candidate.raw_content_included is not False:
            raise ValueError("candidate must remain inactive and redaction-safe")
        if type(self.protected_region_validation) is not ProtectedRegionValidationResult:
            raise ValueError("protected_region_validation must be ProtectedRegionValidationResult")
        if type(self.receipt) is not DurableCompactionReceipt:
            raise ValueError("receipt must be DurableCompactionReceipt")
        if self.receipt.artifact_id != self.candidate.artifact_reference.artifact_id:
            raise ValueError("receipt artifact_id must match candidate")
        passed = self.status is DurableCompactionValidationStatus.PASSED
        accepted_region_status = (
            ProtectedRegionValidationStatus.PASSED,
            ProtectedRegionValidationStatus.NOT_APPLICABLE,
        )
        rejected_region_status = (
            ProtectedRegionValidationStatus.FAILED,
            ProtectedRegionValidationStatus.SKIPPED,
        )
        if passed:
            if self.protected_region_validation.status not in accepted_region_status:
                raise ValueError("passed outcome requires accepted protected validation")
            if self.rollback_metadata is None or self.activation_requirements is None:
                raise ValueError("passed outcome requires rollback and activation metadata")
            if not self.receipt.validation_passed:
                raise ValueError("passed outcome requires a passing receipt")
        else:
            if self.protected_region_validation.status not in rejected_region_status:
                raise ValueError("rejected outcome requires failed protected validation")
            if self.rollback_metadata is not None or self.activation_requirements is not None:
                raise ValueError("rejected outcome cannot contain activation metadata")
            if self.receipt.validation_passed:
                raise ValueError("rejected outcome requires a failing receipt")
        if self.raw_content_included is not False:
            raise ValueError("raw_content_included must be False")


def _merge_regions(
    detected: tuple[ProtectedRegion, ...],
    explicit: tuple[ProtectedRegion, ...] | None,
) -> tuple[ProtectedRegion, ...]:
    merged: list[ProtectedRegion] = []
    seen_values: set[str] = set()
    for region in (*detected, *(explicit or ())):
        if region.value in seen_values:
            continue
        seen_values.add(region.value)
        merged.append(region)
    return tuple(merged)


def _safe_validation_result(
    result: ProtectedRegionValidationResult,
    *,
    policy_version: str | None,
) -> ProtectedRegionValidationResult:
    if result.status is ProtectedRegionValidationStatus.FAILED:
        failures = ("protected_region_values_not_preserved",)
    else:
        failures = ()
    metadata = {
        "failure_reason": "protected_region_values_not_preserved"
        if result.status is ProtectedRegionValidationStatus.FAILED
        else None,
        "protected_region_policy_version": policy_version,
    }
    return ProtectedRegionValidationResult(
        status=result.status,
        regions_checked=result.regions_checked,
        regions_preserved=result.regions_preserved,
        regions_failed=result.regions_failed,
        failures=failures,
        metadata=metadata,
    )


def _token_measurements(
    stored: StoredOptimizationArtifact,
) -> tuple[int | None, int | None, int | None, bool]:
    metadata = stored.metadata.validation.safe_metadata
    keys = {"input_tokens", "output_tokens", "saved_tokens"}
    if not keys.intersection(metadata):
        return None, None, None, False
    if "input_tokens" not in metadata or "output_tokens" not in metadata:
        raise DurableCompactionValidationError(
            DurableCompactionValidationReason.TOKEN_MEASUREMENT_INVALID
        )
    input_tokens = _non_negative_int(metadata["input_tokens"], "input_tokens")
    output_tokens = _non_negative_int(metadata["output_tokens"], "output_tokens")
    saved_tokens = input_tokens - output_tokens
    if "saved_tokens" in metadata and metadata["saved_tokens"] != saved_tokens:
        raise DurableCompactionValidationError(
            DurableCompactionValidationReason.TOKEN_MEASUREMENT_INVALID
        )
    return input_tokens, output_tokens, saved_tokens, True


class DurableCompactionValidationCompiler:
    """Compile safe validation, receipt, rollback and activation prerequisites."""

    def __init__(
        self,
        *,
        repository: OptimizationArtifactRepository,
        clock: Callable[[], datetime],
        receipt_id_factory: Callable[[], str],
    ) -> None:
        if not isinstance(repository, OptimizationArtifactRepository):
            raise ValueError("repository must implement OptimizationArtifactRepository")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(receipt_id_factory):
            raise TypeError("receipt_id_factory must be callable")
        self._repository = repository
        self._clock = clock
        self._receipt_id_factory = receipt_id_factory

    def compile(
        self,
        request: DurableCompactionValidationRequest,
    ) -> DurableCompactionValidationOutcome:
        try:
            if type(request) is not DurableCompactionValidationRequest:
                raise ValueError
            request.__post_init__()
        except DurableCompactionValidationError:
            raise
        except Exception:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.INVALID_REQUEST
            ) from None

        candidate = request.compaction_result.candidate
        if candidate is None:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.INVALID_REQUEST
            )
        stored = self._resolve_candidate(request, candidate)
        summary, payload_hash = self._validate_candidate_artifact(request, candidate, stored)
        input_tokens, output_tokens, saved_tokens, tokens_available = _token_measurements(stored)

        try:
            detected: list[ProtectedRegion] = []
            for message in request.compaction_request.snapshot.messages:
                detected.extend(detect_protected_regions(message.content))
            regions = _merge_regions(tuple(detected), request.protected_regions)
            protected_validation = validate_protected_regions(
                "",
                summary,
                regions=regions,
            )
            safe_validation = _safe_validation_result(
                protected_validation,
                policy_version=request.compaction_request.policy.protected_region_policy_version,
            )
        except Exception:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.PROTECTED_REGION_VALIDATION_FAILED
            ) from None

        try:
            created_at = _aware_datetime(self._clock(), "created_at")
            receipt_id = _strict_non_empty(self._receipt_id_factory(), "receipt_id")
            source_identity = request.compaction_request.snapshot.source_identity
            receipt = DurableCompactionReceipt(
                receipt_id=receipt_id,
                created_at=created_at,
                tenant_id=source_identity.tenant_id,
                context_scope_id=source_identity.context_scope_id,
                candidate_status=candidate.status,
                artifact_id=candidate.artifact_reference.artifact_id,
                artifact_lookup_hash=candidate.artifact_lookup_hash,
                artifact_content_hash=payload_hash,
                source_identity_hash=candidate.source_identity_hash,
                durable_policy_hash=candidate.durable_policy_hash,
                source_revision=source_identity.source_revision,
                expected_active_revision=source_identity.expected_active_revision,
                source_ref_count=len(source_identity.source_refs),
                reused_existing_artifact=request.compaction_result.reused,
                created_new_artifact=request.compaction_result.created,
                invalidated_prior_artifact=False,
                llm_invoked=request.compaction_result.llm_invoked,
                protected_region_status=safe_validation.status,
                validation_passed=safe_validation.status
                in (
                    ProtectedRegionValidationStatus.PASSED,
                    ProtectedRegionValidationStatus.NOT_APPLICABLE,
                ),
                regions_checked=safe_validation.regions_checked,
                regions_preserved=safe_validation.regions_preserved,
                regions_failed=safe_validation.regions_failed,
                original_chars=sum(
                    len(message.content)
                    for message in request.compaction_request.snapshot.messages
                ),
                candidate_chars=len(summary),
                saved_chars=sum(
                    len(message.content)
                    for message in request.compaction_request.snapshot.messages
                )
                - len(summary),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                saved_tokens=saved_tokens,
                token_measurement_available=tokens_available,
                rollback_metadata_present=safe_validation.status
                in (
                    ProtectedRegionValidationStatus.PASSED,
                    ProtectedRegionValidationStatus.NOT_APPLICABLE,
                ),
            )
        except DurableCompactionValidationError:
            raise
        except Exception:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.RECEIPT_COMPILATION_FAILED
            ) from None

        if safe_validation.status in (
            ProtectedRegionValidationStatus.FAILED,
            ProtectedRegionValidationStatus.SKIPPED,
        ):
            return DurableCompactionValidationOutcome(
                status=DurableCompactionValidationStatus.REJECTED,
                candidate=candidate,
                protected_region_validation=safe_validation,
                receipt=receipt,
                rollback_metadata=None,
                activation_requirements=None,
            )
        if safe_validation.status not in (
            ProtectedRegionValidationStatus.PASSED,
            ProtectedRegionValidationStatus.NOT_APPLICABLE,
        ):
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.PROTECTED_REGION_VALIDATION_FAILED
            )

        try:
            source_identity = request.compaction_request.snapshot.source_identity
            rollback_metadata = CompactionRollbackMetadata(
                tenant_id=source_identity.tenant_id,
                context_scope_id=source_identity.context_scope_id,
                source_revision=source_identity.source_revision,
                expected_active_revision=source_identity.expected_active_revision,
                rollback_source_reference=request.rollback_source_reference,
                candidate_artifact_reference=candidate.artifact_reference,
                prior_artifact_reference=request.prior_artifact_reference,
                source_identity_hash=candidate.source_identity_hash,
                candidate_artifact_content_hash=candidate.artifact_content_hash,
                compiled_at=created_at,
            )
            lineage_digest = hashlib.sha256(
                "|".join(
                    (
                        candidate.source_identity_hash,
                        candidate.artifact_content_hash,
                        receipt.receipt_id,
                    )
                ).encode("utf-8")
            ).hexdigest()
            activation_requirements = DurableCompactionActivationRequirements(
                expected_active_revision=source_identity.expected_active_revision,
                candidate_artifact_id=candidate.artifact_reference.artifact_id,
                validated_artifact_id=candidate.artifact_reference.artifact_id,
                lineage_reference=f"lineage_{lineage_digest}",
                creation_receipt_reference=receipt.receipt_id,
                rollback_source_reference=request.rollback_source_reference,
            )
        except Exception:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.ROLLBACK_METADATA_REQUIRED
            ) from None

        return DurableCompactionValidationOutcome(
            status=DurableCompactionValidationStatus.PASSED,
            candidate=candidate,
            protected_region_validation=safe_validation,
            receipt=receipt,
            rollback_metadata=rollback_metadata,
            activation_requirements=activation_requirements,
        )

    def _resolve_candidate(
        self,
        request: DurableCompactionValidationRequest,
        candidate: CompactionCandidate,
    ) -> StoredOptimizationArtifact:
        try:
            stored = self._repository.resolve(candidate.artifact_reference)
        except Exception:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_ARTIFACT_INVALID
            ) from None
        if stored is None:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_ARTIFACT_NOT_FOUND
            )
        if type(stored) is not StoredOptimizationArtifact:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_ARTIFACT_INVALID
            )
        return stored

    @staticmethod
    def _validate_candidate_artifact(
        request: DurableCompactionValidationRequest,
        candidate: CompactionCandidate,
        stored: StoredOptimizationArtifact,
    ) -> tuple[str, str]:
        source_identity = request.compaction_request.snapshot.source_identity
        lookup_key = source_identity.artifact_lookup_key
        expected_lookup_hash = compute_artifact_lookup_key_hash(lookup_key)
        reference = candidate.artifact_reference
        metadata = stored.metadata
        if (
            metadata.artifact_id != reference.artifact_id
            or metadata.lookup_key != lookup_key
            or metadata.artifact_content_hash != reference.artifact_content_hash
            or metadata.status.value != "validated"
            or metadata.validation.status is not ArtifactValidationStatus.PASSED
            or reference.artifact_lookup_key_hash != expected_lookup_hash
            or reference.artifact_content_hash != candidate.artifact_content_hash
        ):
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_IDENTITY_MISMATCH
            )
        try:
            summary, payload_hash = validate_stored_message_sequence_artifact(
                stored,
                lookup_key=lookup_key,
            )
        except Exception:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_ARTIFACT_INVALID
            ) from None
        if payload_hash != candidate.artifact_content_hash:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_IDENTITY_MISMATCH
            )
        if compute_artifact_content_hash(stored.payload) != candidate.artifact_content_hash:
            raise DurableCompactionValidationError(
                DurableCompactionValidationReason.CANDIDATE_ARTIFACT_INVALID
            )
        return summary, payload_hash

