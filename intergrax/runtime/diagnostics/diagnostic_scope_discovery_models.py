# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed models for diagnostic execution scope discovery (DG-002 Slice 1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.diagnostics.diagnostic_subject import ExecutionDiagnosticSubjectRef
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId, validate_problem_id

DEFAULT_SCOPE_DISCOVERY_CANDIDATE_LIMIT = 10
MAX_SCOPE_DISCOVERY_CANDIDATE_LIMIT = 100

_MAX_PROVIDER_ID_LENGTH = 64
_MAX_REFERENCE_KIND_LENGTH = 64
_MAX_CANONICAL_RECORD_REF_LENGTH = 256
_MAX_LIMITATION_LENGTH = 512


class DiagnosticScopeReferenceKind(StrEnum):
    """Supported diagnostic scope reference discriminators."""

    PROBLEM = "problem"


class DiagnosticScopeDiscoveryStatus(StrEnum):
    """Canonical scope discovery outcomes."""

    RESOLVED = "RESOLVED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NON_EXECUTION_SUBJECT = "NON_EXECUTION_SUBJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUPPORTED_REFERENCE = "UNSUPPORTED_REFERENCE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ProblemScopeReference:
    """ProblemId-backed diagnostic scope reference."""

    problem_id: ProblemId

    @property
    def kind(self) -> DiagnosticScopeReferenceKind:
        return DiagnosticScopeReferenceKind.PROBLEM


DiagnosticScopeReference = ProblemScopeReference


@dataclass(frozen=True, slots=True)
class DiagnosticScopeDiscoveryRequest:
    """Tenant-scoped scope discovery request."""

    tenant_id: str
    reference: DiagnosticScopeReference
    candidate_limit: int = DEFAULT_SCOPE_DISCOVERY_CANDIDATE_LIMIT


@dataclass(frozen=True, slots=True)
class DiagnosticScopeResolutionProvenance:
    """Bounded provenance for one scope resolution path."""

    provider_id: str
    reference_kind: DiagnosticScopeReferenceKind
    canonical_record_ref: str


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionScopeCandidate:
    """One candidate execution diagnostic scope."""

    subject_ref: ExecutionDiagnosticSubjectRef
    provenance: DiagnosticScopeResolutionProvenance


@dataclass(frozen=True, slots=True)
class DiagnosticScopeDiscoveryResult:
    """Public scope discovery result with explicit status invariants."""

    status: DiagnosticScopeDiscoveryStatus
    resolved_scope: ExecutionDiagnosticSubjectRef | None
    candidates: tuple[DiagnosticExecutionScopeCandidate, ...]
    candidate_count: int
    candidate_count_exact: bool
    provenance: tuple[DiagnosticScopeResolutionProvenance, ...]
    limitations: tuple[str, ...]


class DiagnosticScopeDiscoveryIntegrityError(Exception):
    """Raised when canonical diagnostic scope data violates tenant or integrity rules."""


class DiagnosticScopeDiscoveryConfigurationError(Exception):
    """Raised when discovery provider composition is invalid."""


class DiagnosticScopeDiscoveryValidationError(ValueError):
    """Raised when request or result invariants are violated."""


def validate_scope_discovery_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id must be non-empty and not whitespace-only")
    if tenant_id != normalized:
        raise ValueError("tenant_id must not contain leading or trailing whitespace")
    return normalized


def validate_scope_discovery_candidate_limit(limit: int) -> int:
    if type(limit) is not int or isinstance(limit, bool):
        raise TypeError("candidate_limit must be int")
    if limit < 1 or limit > MAX_SCOPE_DISCOVERY_CANDIDATE_LIMIT:
        raise ValueError(
            f"candidate_limit must be between 1 and {MAX_SCOPE_DISCOVERY_CANDIDATE_LIMIT}",
        )
    return limit


def validate_scope_discovery_request(
    request: DiagnosticScopeDiscoveryRequest,
) -> DiagnosticScopeDiscoveryRequest:
    tenant_id = validate_scope_discovery_tenant_id(request.tenant_id)
    candidate_limit = validate_scope_discovery_candidate_limit(request.candidate_limit)
    if type(request.reference) is not ProblemScopeReference:
        raise TypeError("reference must be ProblemScopeReference")
    problem_id = validate_problem_id(request.reference.problem_id)
    reference = ProblemScopeReference(problem_id=problem_id)
    if (
        tenant_id == request.tenant_id
        and candidate_limit == request.candidate_limit
        and reference == request.reference
    ):
        return request
    return DiagnosticScopeDiscoveryRequest(
        tenant_id=tenant_id,
        reference=reference,
        candidate_limit=candidate_limit,
    )


def _require_bounded_identifier(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if value != normalized:
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length {max_length}")
    return normalized


def validate_scope_resolution_provenance(
    provenance: DiagnosticScopeResolutionProvenance,
) -> DiagnosticScopeResolutionProvenance:
    provider_id = _require_bounded_identifier(
        provenance.provider_id,
        field_name="provider_id",
        max_length=_MAX_PROVIDER_ID_LENGTH,
    )
    if type(provenance.reference_kind) is not DiagnosticScopeReferenceKind:
        raise TypeError("reference_kind must be DiagnosticScopeReferenceKind")
    canonical_record_ref = _require_bounded_identifier(
        provenance.canonical_record_ref,
        field_name="canonical_record_ref",
        max_length=_MAX_CANONICAL_RECORD_REF_LENGTH,
    )
    if (
        provider_id == provenance.provider_id
        and canonical_record_ref == provenance.canonical_record_ref
    ):
        return provenance
    return DiagnosticScopeResolutionProvenance(
        provider_id=provider_id,
        reference_kind=provenance.reference_kind,
        canonical_record_ref=canonical_record_ref,
    )


def _validate_limitations(limitations: tuple[str, ...]) -> tuple[str, ...]:
    validated: list[str] = []
    for limitation in limitations:
        if type(limitation) is not str:
            raise TypeError("limitations entries must be str")
        normalized = limitation.strip()
        if not normalized:
            raise ValueError("limitations entries must be non-empty")
        if len(normalized) > _MAX_LIMITATION_LENGTH:
            raise ValueError(
                f"limitations entries must not exceed {_MAX_LIMITATION_LENGTH} characters",
            )
        validated.append(normalized)
    return tuple(validated)


def build_diagnostic_scope_discovery_result(
    *,
    status: DiagnosticScopeDiscoveryStatus,
    resolved_scope: ExecutionDiagnosticSubjectRef | None,
    candidates: tuple[DiagnosticExecutionScopeCandidate, ...],
    candidate_count: int,
    candidate_count_exact: bool,
    provenance: tuple[DiagnosticScopeResolutionProvenance, ...],
    limitations: tuple[str, ...] = (),
) -> DiagnosticScopeDiscoveryResult:
    if type(status) is not DiagnosticScopeDiscoveryStatus:
        raise TypeError("status must be DiagnosticScopeDiscoveryStatus")
    if type(candidate_count) is not int or isinstance(candidate_count, bool):
        raise TypeError("candidate_count must be int")
    if type(candidate_count_exact) is not bool:
        raise TypeError("candidate_count_exact must be bool")
    if candidate_count < 0:
        raise DiagnosticScopeDiscoveryValidationError("candidate_count must be non-negative")
    if len(candidates) > candidate_count:
        raise DiagnosticScopeDiscoveryValidationError(
            "candidates length must not exceed candidate_count",
        )

    validated_provenance = tuple(
        validate_scope_resolution_provenance(item) for item in provenance
    )
    validated_limitations = _validate_limitations(limitations)

    if status is DiagnosticScopeDiscoveryStatus.RESOLVED:
        if resolved_scope is None:
            raise DiagnosticScopeDiscoveryValidationError(
                "RESOLVED requires resolved_scope",
            )
        if candidate_count != 1:
            raise DiagnosticScopeDiscoveryValidationError(
                "RESOLVED requires candidate_count == 1",
            )
        if len(candidates) != 1:
            raise DiagnosticScopeDiscoveryValidationError(
                "RESOLVED requires exactly one candidate",
            )
    elif status is DiagnosticScopeDiscoveryStatus.NOT_FOUND:
        if resolved_scope is not None:
            raise DiagnosticScopeDiscoveryValidationError(
                "NOT_FOUND requires resolved_scope == None",
            )
        if candidate_count != 0:
            raise DiagnosticScopeDiscoveryValidationError(
                "NOT_FOUND requires candidate_count == 0",
            )
        if candidates:
            raise DiagnosticScopeDiscoveryValidationError(
                "NOT_FOUND requires empty candidates",
            )
    elif status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS:
        if resolved_scope is not None:
            raise DiagnosticScopeDiscoveryValidationError(
                "AMBIGUOUS requires resolved_scope == None",
            )
        if candidate_count <= 1:
            raise DiagnosticScopeDiscoveryValidationError(
                "AMBIGUOUS requires candidate_count > 1",
            )
    elif status is DiagnosticScopeDiscoveryStatus.NON_EXECUTION_SUBJECT:
        if resolved_scope is not None:
            raise DiagnosticScopeDiscoveryValidationError(
                "NON_EXECUTION_SUBJECT requires resolved_scope == None",
            )
    elif status in {
        DiagnosticScopeDiscoveryStatus.UNSUPPORTED_REFERENCE,
        DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE,
    }:
        if candidates:
            raise DiagnosticScopeDiscoveryValidationError(
                f"{status.value} must not include fabricated candidates",
            )

    return DiagnosticScopeDiscoveryResult(
        status=status,
        resolved_scope=resolved_scope,
        candidates=candidates,
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
        provenance=validated_provenance,
        limitations=validated_limitations,
    )


def unsupported_reference_result() -> DiagnosticScopeDiscoveryResult:
    return build_diagnostic_scope_discovery_result(
        status=DiagnosticScopeDiscoveryStatus.UNSUPPORTED_REFERENCE,
        resolved_scope=None,
        candidates=(),
        candidate_count=0,
        candidate_count_exact=True,
        provenance=(),
    )


def provider_unavailable_result(
    *,
    provenance: tuple[DiagnosticScopeResolutionProvenance, ...] = (),
    limitations: tuple[str, ...] = (),
) -> DiagnosticScopeDiscoveryResult:
    return build_diagnostic_scope_discovery_result(
        status=DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE,
        resolved_scope=None,
        candidates=(),
        candidate_count=0,
        candidate_count_exact=True,
        provenance=provenance,
        limitations=limitations,
    )
