# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External contract compatibility assessment contracts (Integrations-owned)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    TaskId,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)


def _require_non_empty_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


def _require_unique_non_empty_refs(
    values: tuple[str, ...], label: str
) -> tuple[str, ...]:
    seen: set[str] = set()
    for item in values:
        text = _require_non_empty_text(item, f"{label} entry")
        if text in seen:
            raise ValueError(f"{label} must be unique")
        seen.add(text)
    return values


def _require_timezone_aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value


class ExternalContractCompatibilityDimension(StrEnum):
    SCHEMA = "schema"
    PROTOCOL = "protocol"
    SEMANTIC = "semantic"


class ExternalContractCompatibilityOutcome(StrEnum):
    COMPATIBLE = "compatible"
    SCHEMA_INCOMPATIBLE = "schema_incompatible"
    PROTOCOL_INCOMPATIBLE = "protocol_incompatible"
    SEMANTIC_INCOMPATIBLE = "semantic_incompatible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DimensionCompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ExternalContractEvidenceAuthority(StrEnum):
    PROVIDER_ADAPTER = "provider_adapter"
    APPLICATION_INVARIANT = "application_invariant"
    CONTRACT_SPECIFICATION = "contract_specification"
    LLM_ADVISORY = "llm_advisory"

    @property
    def is_authoritative(self) -> bool:
        return self is not ExternalContractEvidenceAuthority.LLM_ADVISORY


class SchemaValidationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ProtocolValidationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class SemanticAssertionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ExternalContractCompatibilityReasonCode(StrEnum):
    NONE = "none"

    SCHEMA_MISMATCH = "schema_mismatch"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    SEMANTIC_MISMATCH = "semantic_mismatch"

    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    EVIDENCE_CONFLICT = "evidence_conflict"
    STALE_EVIDENCE = "stale_evidence"
    IDENTITY_MISMATCH = "identity_mismatch"

    UNSUPPORTED_EVALUATOR = "unsupported_evaluator"
    EVALUATOR_AMBIGUITY = "evaluator_ambiguity"


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilitySubject:
    tenant_id: str
    integration_id: str
    provider_id: str
    integration_kind: str
    external_operation_id: str

    host_binding_ref: str | None = None

    execution_task_id: TaskId | None = None
    execution_run_id: RunId | None = None
    execution_id: ExecutionId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant_id", _require_non_empty_text(self.tenant_id, "tenant_id")
        )
        object.__setattr__(
            self,
            "integration_id",
            _require_non_empty_text(self.integration_id, "integration_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            _require_non_empty_text(self.provider_id, "provider_id"),
        )
        object.__setattr__(
            self,
            "integration_kind",
            _require_non_empty_text(self.integration_kind, "integration_kind"),
        )
        object.__setattr__(
            self,
            "external_operation_id",
            _require_non_empty_text(
                self.external_operation_id, "external_operation_id"
            ),
        )
        if self.host_binding_ref is not None:
            object.__setattr__(
                self,
                "host_binding_ref",
                _require_non_empty_text(self.host_binding_ref, "host_binding_ref"),
            )
        expected_integration_id = f"{self.provider_id}:{self.integration_kind}"
        if self.integration_id != expected_integration_id:
            raise ValueError(
                "integration_id must equal f'{provider_id}:{integration_kind}'"
            )
        if self.execution_task_id is not None:
            object.__setattr__(
                self, "execution_task_id", validate_task_id(self.execution_task_id)
            )
        if self.execution_run_id is not None:
            object.__setattr__(
                self, "execution_run_id", validate_run_id(self.execution_run_id)
            )
        if self.execution_id is not None:
            object.__setattr__(
                self, "execution_id", validate_execution_id(self.execution_id)
            )


@dataclass(frozen=True, slots=True)
class ExternalContractPin:
    contract_ref: str
    contract_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_ref",
            _require_non_empty_text(self.contract_ref, "contract_ref"),
        )
        object.__setattr__(
            self,
            "contract_version",
            _require_non_empty_text(self.contract_version, "contract_version"),
        )


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityExpectation:
    expectation_id: str
    subject: ExternalContractCompatibilitySubject
    expected_contract: ExternalContractPin
    required_dimensions: frozenset[ExternalContractCompatibilityDimension]

    schema_expectation_ref: str | None = None
    protocol_expectation_ref: str | None = None
    semantic_expectation_refs: tuple[str, ...] = ()
    domain_extension_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expectation_id",
            _require_non_empty_text(self.expectation_id, "expectation_id"),
        )
        if not self.required_dimensions:
            raise ValueError("required_dimensions must be non-empty")
        object.__setattr__(
            self,
            "semantic_expectation_refs",
            _require_unique_non_empty_refs(
                self.semantic_expectation_refs, "semantic_expectation_refs"
            ),
        )
        if self.schema_expectation_ref is not None:
            object.__setattr__(
                self,
                "schema_expectation_ref",
                _require_non_empty_text(
                    self.schema_expectation_ref, "schema_expectation_ref"
                ),
            )
        if self.protocol_expectation_ref is not None:
            object.__setattr__(
                self,
                "protocol_expectation_ref",
                _require_non_empty_text(
                    self.protocol_expectation_ref, "protocol_expectation_ref"
                ),
            )
        if self.domain_extension_ref is not None:
            object.__setattr__(
                self,
                "domain_extension_ref",
                _require_non_empty_text(
                    self.domain_extension_ref, "domain_extension_ref"
                ),
            )
        if ExternalContractCompatibilityDimension.SCHEMA in self.required_dimensions:
            if self.schema_expectation_ref is None:
                raise ValueError(
                    "schema_expectation_ref is required when SCHEMA dimension is required"
                )
        if ExternalContractCompatibilityDimension.PROTOCOL in self.required_dimensions:
            if self.protocol_expectation_ref is None:
                raise ValueError(
                    "protocol_expectation_ref is required when PROTOCOL dimension is required"
                )
        if ExternalContractCompatibilityDimension.SEMANTIC in self.required_dimensions:
            if not self.semantic_expectation_refs and self.domain_extension_ref is None:
                raise ValueError(
                    "semantic_expectation_refs or domain_extension_ref is required "
                    "when SEMANTIC dimension is required"
                )


@dataclass(frozen=True, slots=True)
class ExternalContractSchemaEvidenceFact:
    schema_ref: str | None
    schema_fingerprint: str | None
    validation_status: SchemaValidationStatus
    violation_codes: tuple[str, ...] = ()
    shape_delta_ref: str | None = None

    def __post_init__(self) -> None:
        if self.schema_ref is not None:
            _require_non_empty_text(self.schema_ref, "schema_ref")
        if self.schema_fingerprint is not None:
            _require_non_empty_text(self.schema_fingerprint, "schema_fingerprint")
        object.__setattr__(
            self,
            "violation_codes",
            _require_unique_non_empty_refs(self.violation_codes, "violation_codes"),
        )
        if self.shape_delta_ref is not None:
            _require_non_empty_text(self.shape_delta_ref, "shape_delta_ref")


@dataclass(frozen=True, slots=True)
class ExternalContractProtocolHeaderInvariantResult:
    invariant_id: str
    status: ProtocolValidationStatus
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "invariant_id",
            _require_non_empty_text(self.invariant_id, "invariant_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _require_unique_non_empty_refs(self.evidence_refs, "evidence_refs"),
        )


@dataclass(frozen=True, slots=True)
class ExternalContractProtocolEvidenceFact:
    protocol_ref: str | None
    protocol_version: str | None
    method: str | None
    content_type: str | None
    validation_status: ProtocolValidationStatus
    violation_codes: tuple[str, ...] = ()
    header_invariant_results: tuple[
        ExternalContractProtocolHeaderInvariantResult, ...
    ] = ()

    def __post_init__(self) -> None:
        if self.protocol_ref is not None:
            _require_non_empty_text(self.protocol_ref, "protocol_ref")
        if self.protocol_version is not None:
            _require_non_empty_text(self.protocol_version, "protocol_version")
        if self.method is not None:
            _require_non_empty_text(self.method, "method")
        if self.content_type is not None:
            _require_non_empty_text(self.content_type, "content_type")
        object.__setattr__(
            self,
            "violation_codes",
            _require_unique_non_empty_refs(self.violation_codes, "violation_codes"),
        )


@dataclass(frozen=True, slots=True)
class ExternalContractSemanticAssertionResult:
    assertion_id: str
    status: SemanticAssertionStatus
    authority: ExternalContractEvidenceAuthority
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assertion_id",
            _require_non_empty_text(self.assertion_id, "assertion_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _require_unique_non_empty_refs(self.evidence_refs, "evidence_refs"),
        )


@dataclass(frozen=True, slots=True)
class ExternalContractSemanticEvidenceFact:
    assertions: tuple[ExternalContractSemanticAssertionResult, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for assertion in self.assertions:
            if assertion.assertion_id in seen:
                raise ValueError("assertion_id must be unique within semantic fact")
            seen.add(assertion.assertion_id)


ExternalContractCompatibilityEvidenceFact = (
    ExternalContractSchemaEvidenceFact
    | ExternalContractProtocolEvidenceFact
    | ExternalContractSemanticEvidenceFact
)


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityEvidence:
    evidence_id: str
    subject: ExternalContractCompatibilitySubject
    observed_contract: ExternalContractPin | None
    dimension: ExternalContractCompatibilityDimension
    observed_at: datetime
    authority: ExternalContractEvidenceAuthority
    evidence_refs: tuple[str, ...]
    fact: ExternalContractCompatibilityEvidenceFact

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _require_non_empty_text(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self,
            "observed_at",
            _require_timezone_aware(self.observed_at, "observed_at"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _require_unique_non_empty_refs(self.evidence_refs, "evidence_refs"),
        )
        if self.dimension is ExternalContractCompatibilityDimension.SCHEMA:
            if type(self.fact) is not ExternalContractSchemaEvidenceFact:
                raise TypeError(
                    "SCHEMA dimension requires ExternalContractSchemaEvidenceFact"
                )
        elif self.dimension is ExternalContractCompatibilityDimension.PROTOCOL:
            if type(self.fact) is not ExternalContractProtocolEvidenceFact:
                raise TypeError(
                    "PROTOCOL dimension requires ExternalContractProtocolEvidenceFact"
                )
        elif self.dimension is ExternalContractCompatibilityDimension.SEMANTIC:
            if type(self.fact) is not ExternalContractSemanticEvidenceFact:
                raise TypeError(
                    "SEMANTIC dimension requires ExternalContractSemanticEvidenceFact"
                )


@dataclass(frozen=True, slots=True)
class ExternalContractAssessmentWindow:
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    max_age: timedelta | None = None
    evidence_ttl_ref: str | None = None

    def __post_init__(self) -> None:
        absolute = (
            self.valid_from is not None
            and self.valid_until is not None
            and self.max_age is None
            and self.evidence_ttl_ref is None
        )
        max_age_mode = (
            self.valid_from is None
            and self.valid_until is None
            and self.max_age is not None
            and self.evidence_ttl_ref is None
        )
        ttl_ref_mode = (
            self.valid_from is None
            and self.valid_until is None
            and self.max_age is None
            and self.evidence_ttl_ref is not None
        )
        if not (absolute or max_age_mode or ttl_ref_mode):
            raise ValueError(
                "assessment window must use exactly one of: absolute, max_age, or evidence_ttl_ref"
            )
        if absolute:
            raw_from = self.valid_from
            raw_until = self.valid_until
            if raw_from is None or raw_until is None:
                raise ValueError("absolute window requires valid_from and valid_until")
            vf = _require_timezone_aware(raw_from, "valid_from")
            vu = _require_timezone_aware(raw_until, "valid_until")
            object.__setattr__(self, "valid_from", vf)
            object.__setattr__(self, "valid_until", vu)
            if vf > vu:
                raise ValueError("valid_from must be <= valid_until")
        if max_age_mode:
            if self.max_age is None or self.max_age <= timedelta(0):
                raise ValueError("max_age must be > 0")
        if ttl_ref_mode:
            object.__setattr__(
                self,
                "evidence_ttl_ref",
                _require_non_empty_text(self.evidence_ttl_ref, "evidence_ttl_ref"),
            )


@runtime_checkable
class ExternalContractCompatibilityEvidencePolicy(Protocol):
    def accepts(
        self,
        evidence: ExternalContractCompatibilityEvidence,
        *,
        assessed_at: datetime,
        window: ExternalContractAssessmentWindow,
    ) -> bool:
        """Return whether evidence is fresh for the assessment window (pure, no I/O)."""
        ...


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityFinding:
    dimension: ExternalContractCompatibilityDimension
    status: DimensionCompatibilityStatus
    reason_code: ExternalContractCompatibilityReasonCode
    evidence_refs: tuple[str, ...]
    evaluator_id: str
    source_authority: ExternalContractEvidenceAuthority

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluator_id",
            _require_non_empty_text(self.evaluator_id, "evaluator_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _require_unique_non_empty_refs(self.evidence_refs, "evidence_refs"),
        )
        if self.status is DimensionCompatibilityStatus.COMPATIBLE:
            if self.reason_code is not ExternalContractCompatibilityReasonCode.NONE:
                raise ValueError("COMPATIBLE findings must use reason_code NONE")
        if self.status is not DimensionCompatibilityStatus.COMPATIBLE:
            if self.reason_code is ExternalContractCompatibilityReasonCode.NONE:
                raise ValueError(
                    "non-COMPATIBLE findings must not use reason_code NONE"
                )


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityEvaluationContext:
    assessment_id: str
    dimension: ExternalContractCompatibilityDimension
    assessed_at: datetime
    assessment_window: ExternalContractAssessmentWindow

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessment_id",
            _require_non_empty_text(self.assessment_id, "assessment_id"),
        )
        object.__setattr__(
            self,
            "assessed_at",
            _require_timezone_aware(self.assessed_at, "assessed_at"),
        )


@runtime_checkable
class ExternalContractCompatibilityEvaluator(Protocol):
    """Pure evaluator SPI: no provider I/O, no mutation, no recovery actions."""

    @property
    def evaluator_id(self) -> str: ...

    @property
    def supported_dimensions(
        self,
    ) -> frozenset[ExternalContractCompatibilityDimension]: ...

    def can_evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        *,
        dimension: ExternalContractCompatibilityDimension,
    ) -> bool: ...

    def evaluate(
        self,
        expectation: ExternalContractCompatibilityExpectation,
        evidence: tuple[ExternalContractCompatibilityEvidence, ...],
        context: ExternalContractCompatibilityEvaluationContext,
    ) -> tuple[ExternalContractCompatibilityFinding, ...]: ...


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityAssessmentRequest:
    assessment_id: str
    expectation: ExternalContractCompatibilityExpectation
    evidence: tuple[ExternalContractCompatibilityEvidence, ...]
    assessed_at: datetime
    assessment_window: ExternalContractAssessmentWindow
    explicit_evaluator_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessment_id",
            _require_non_empty_text(self.assessment_id, "assessment_id"),
        )
        object.__setattr__(
            self,
            "assessed_at",
            _require_timezone_aware(self.assessed_at, "assessed_at"),
        )
        object.__setattr__(
            self,
            "explicit_evaluator_ids",
            _require_unique_non_empty_refs(
                self.explicit_evaluator_ids, "explicit_evaluator_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class ExternalContractExpectationKey:
    """Resolver lookup key: canonical subject identity plus expected contract pin."""

    subject: ExternalContractCompatibilitySubject
    expected_contract: ExternalContractPin


@runtime_checkable
class ExternalContractCompatibilityExpectationResolver(Protocol):
    """Optional composition extension: resolve expectations by typed key (no Catalog)."""

    @property
    def resolver_id(self) -> str: ...

    def resolve(
        self, key: ExternalContractExpectationKey
    ) -> ExternalContractCompatibilityExpectation | None: ...


@dataclass(frozen=True, slots=True)
class ExternalContractEvidenceCollectionRequest:
    expectation: ExternalContractCompatibilityExpectation
    assessed_at: datetime
    assessment_window: ExternalContractAssessmentWindow | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessed_at",
            _require_timezone_aware(self.assessed_at, "assessed_at"),
        )

    @property
    def subject(self) -> ExternalContractCompatibilitySubject:
        return self.expectation.subject


@runtime_checkable
class ExternalContractEvidenceProvider(Protocol):
    """Collect external/provider/domain facts before assessment (I/O allowed here only)."""

    @property
    def evidence_provider_id(self) -> str: ...

    def collect(
        self,
        request: ExternalContractEvidenceCollectionRequest,
    ) -> tuple[ExternalContractCompatibilityEvidence, ...]: ...


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityAssessment:
    assessment_id: str
    expectation_id: str
    subject: ExternalContractCompatibilitySubject
    outcome: ExternalContractCompatibilityOutcome
    findings: tuple[ExternalContractCompatibilityFinding, ...]
    reason_code: ExternalContractCompatibilityReasonCode
    evidence_refs: tuple[str, ...]
    assessed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessment_id",
            _require_non_empty_text(self.assessment_id, "assessment_id"),
        )
        object.__setattr__(
            self,
            "expectation_id",
            _require_non_empty_text(self.expectation_id, "expectation_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _require_unique_non_empty_refs(self.evidence_refs, "evidence_refs"),
        )
        object.__setattr__(
            self,
            "assessed_at",
            _require_timezone_aware(self.assessed_at, "assessed_at"),
        )
