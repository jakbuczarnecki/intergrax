# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical worker capability discovery and acquisition decision contracts (AW-7A).

Produces capability need → discovery → bounded acquisition decision.
Does not execute capability acquisition, mint authority, or invoke CodeCraft.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryStrategy,
    WorkerRecoveryDecision,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
    validate_capability_profile_ref,
    validate_codecraft_profile_ref,
)
from intergrax.contracts.autonomous_work.references import (
    ProblemReference,
    validate_problem_reference,
)

ACQUISITION_DECISION_POLICY_VERSION: str = "aw-7a.v1"

_CAPABILITY_ACQUISITION_STRATEGIES: frozenset[RecoveryStrategy] = frozenset(
    {
        RecoveryStrategy.ACQUIRE_CAPABILITY,
        RecoveryStrategy.ADAPT_INTEGRATION,
    }
)

_AUTHORITY_SIGNAL_PREFIXES: frozenset[str] = frozenset(
    {
        "credential/",
        "authority/",
        "delegation/",
        "principal/",
        "policy/",
        "tenant/",
        "workspace-scope/",
    }
)


class CapabilityNeedKind(StrEnum):
    """Typed capability need categories — not vendor-specific."""

    DATA_TRANSFORMATION = "DATA_TRANSFORMATION"
    TOOL_OPERATION = "TOOL_OPERATION"
    EXTERNAL_INTEGRATION = "EXTERNAL_INTEGRATION"
    PROTOCOL_ADAPTATION = "PROTOCOL_ADAPTATION"
    WORKFLOW_ALTERNATIVE = "WORKFLOW_ALTERNATIVE"
    SCHEMA_ADAPTATION = "SCHEMA_ADAPTATION"
    GENERAL_CAPABILITY = "GENERAL_CAPABILITY"


class WorkerCapabilityCandidateKind(StrEnum):
    """Discovered or classified acquisition candidate kinds."""

    TOOL = "TOOL"
    SKILL = "SKILL"
    INTEGRATION = "INTEGRATION"
    APPROVED_ALTERNATE = "APPROVED_ALTERNATE"
    EXISTING_CONFIGURATION = "EXISTING_CONFIGURATION"
    CODECRAFT_EPHEMERAL = "CODECRAFT_EPHEMERAL"
    ADAPTIVE_INTEGRATION = "ADAPTIVE_INTEGRATION"
    DURABLE_PRODUCTION_CHANGE = "DURABLE_PRODUCTION_CHANGE"
    AUTHORITY_CHANGE = "AUTHORITY_CHANGE"


class WorkerAutonomyLevel(StrEnum):
    """Canonical A0–A4 capability autonomy / risk classification."""

    A0_KNOWN_CAPABILITY = "A0"
    A1_EPHEMERAL_SAFE = "A1"
    A2_SCOPED_ADAPTIVE = "A2"
    A3_PRODUCTION_CHANGE = "A3"
    A4_AUTHORITY_CHANGE = "A4"


class CapabilityDiscoveryDisposition(StrEnum):
    """Per discovery layer or aggregate discovery outcome."""

    MATCH_FOUND = "MATCH_FOUND"
    NO_MATCH = "NO_MATCH"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CONFLICT = "CONFLICT"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    AUTHORITY_CHANGE_REQUIRED = "AUTHORITY_CHANGE_REQUIRED"


class CapabilityAcquisitionDisposition(StrEnum):
    """Typed acquisition decision outcome."""

    USE_EXISTING = "USE_EXISTING"
    CONFIGURE_EXISTING = "CONFIGURE_EXISTING"
    EPHEMERAL_GENERATION_CANDIDATE = "EPHEMERAL_GENERATION_CANDIDATE"
    SCOPED_ADAPTATION_CANDIDATE = "SCOPED_ADAPTATION_CANDIDATE"
    PRODUCTION_CHANGE_REQUIRED = "PRODUCTION_CHANGE_REQUIRED"
    AUTHORITY_CHANGE_REQUIRED = "AUTHORITY_CHANGE_REQUIRED"
    ESCALATE = "ESCALATE"
    NO_SAFE_CAPABILITY = "NO_SAFE_CAPABILITY"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class CapabilityAcquisitionReasonCode(StrEnum):
    """Evidence-bearing acquisition decision reason codes."""

    EXISTING_TOOL_SELECTED = "EXISTING_TOOL_SELECTED"
    EXISTING_SKILL_SELECTED = "EXISTING_SKILL_SELECTED"
    EXISTING_INTEGRATION_SELECTED = "EXISTING_INTEGRATION_SELECTED"
    APPROVED_ALTERNATE_SELECTED = "APPROVED_ALTERNATE_SELECTED"
    EXISTING_CONFIGURATION_SELECTED = "EXISTING_CONFIGURATION_SELECTED"
    A1_CANDIDATE_ALLOWED = "A1_CANDIDATE_ALLOWED"
    A2_ADAPTATION_REQUIRED = "A2_ADAPTATION_REQUIRED"
    A3_PRODUCTION_CHANGE_REQUIRED = "A3_PRODUCTION_CHANGE_REQUIRED"
    A4_AUTHORITY_CHANGE_REQUIRED = "A4_AUTHORITY_CHANGE_REQUIRED"
    NO_SAFE_CANDIDATE = "NO_SAFE_CANDIDATE"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    DISCOVERY_UNAVAILABLE = "DISCOVERY_UNAVAILABLE"
    PROFILE_UNAVAILABLE = "PROFILE_UNAVAILABLE"
    STALE_PROFILE = "STALE_PROFILE"
    CONFLICT = "CONFLICT"
    RECOVERY_STRATEGY_REJECTED = "RECOVERY_STRATEGY_REJECTED"
    POLICY_DENIED_DEFENSE = "POLICY_DENIED_DEFENSE"
    CREDENTIAL_DEFENSE = "CREDENTIAL_DEFENSE"
    AUTHORITY_NEED_REJECTED = "AUTHORITY_NEED_REJECTED"


class WorkerCapabilityAuthorityCompatibility(StrEnum):
    """Read-only authority envelope assessment — never grants authority."""

    COMPATIBLE = "COMPATIBLE"
    AUTHORITY_CHANGE_REQUIRED = "AUTHORITY_CHANGE_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"


class CapabilityOperationCoverage(StrEnum):
    """Candidate compatibility against required operations."""

    EXACT = "EXACT"
    PARTIAL = "PARTIAL"


def validate_capability_operation(value: object) -> str:
    return require_non_empty_text(value, label="capability_operation")


def validate_capability_ref(value: object) -> str:
    return require_non_empty_text(value, label="capability_ref")


@dataclass(frozen=True, slots=True)
class WorkerCapabilityNeed:
    """Immutable typed capability need — distinct from authority need."""

    worker_instance_id: WorkerInstanceId
    obstacle_id: str
    need_kind: CapabilityNeedKind
    required_operations: tuple[str, ...]
    capability_profile_ref: CapabilityProfileRef
    requested_at: datetime
    recovery_decision_id: str
    evidence_refs: tuple[ProblemReference, ...] = ()
    recovery_episode_id: str | None = None
    required_data_domains: tuple[str, ...] = ()
    required_protocols: tuple[str, ...] = ()
    required_resource_refs: tuple[str, ...] = ()
    codecraft_profile_ref: CodecraftProfileRef | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "obstacle_id",
            require_non_empty_text(self.obstacle_id, label="obstacle_id"),
        )
        if type(self.need_kind) is not CapabilityNeedKind:
            raise TypeError("need_kind must be CapabilityNeedKind")
        object.__setattr__(
            self,
            "required_operations",
            _validate_operations(self.required_operations),
        )
        validate_capability_profile_ref(self.capability_profile_ref)
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )
        object.__setattr__(
            self,
            "recovery_decision_id",
            require_non_empty_text(
                self.recovery_decision_id,
                label="recovery_decision_id",
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        if self.recovery_episode_id is not None:
            object.__setattr__(
                self,
                "recovery_episode_id",
                require_non_empty_text(
                    self.recovery_episode_id,
                    label="recovery_episode_id",
                ),
            )
        object.__setattr__(
            self,
            "required_data_domains",
            _validate_text_tuple(self.required_data_domains, label="required_data_domains"),
        )
        object.__setattr__(
            self,
            "required_protocols",
            _validate_text_tuple(self.required_protocols, label="required_protocols"),
        )
        object.__setattr__(
            self,
            "required_resource_refs",
            _validate_text_tuple(self.required_resource_refs, label="required_resource_refs"),
        )
        if self.codecraft_profile_ref is not None:
            validate_codecraft_profile_ref(self.codecraft_profile_ref)


def need_implies_authority_expansion(need: WorkerCapabilityNeed) -> bool:
    """Detect authority/credential requirements masquerading as capability needs."""

    for resource_ref in need.required_resource_refs:
        normalized = resource_ref.strip().lower()
        for prefix in _AUTHORITY_SIGNAL_PREFIXES:
            if normalized.startswith(prefix):
                return True
    return False


def derive_worker_capability_need_id(need: WorkerCapabilityNeed) -> str:
    operations = "|".join(sorted(need.required_operations))
    return (
        f"{need.worker_instance_id}:"
        f"{need.obstacle_id}:"
        f"{need.recovery_decision_id}:"
        f"{need.need_kind.value}:"
        f"{operations}:"
        f"{need.capability_profile_ref.profile_id}@"
        f"{need.capability_profile_ref.version.value}"
    )


@dataclass(frozen=True, slots=True)
class WorkerCapabilityDiscoveryRequest:
    """Provider-neutral discovery request."""

    need: WorkerCapabilityNeed
    profile_ref: CapabilityProfileRef
    worker_instance_id: WorkerInstanceId

    def __post_init__(self) -> None:
        if type(self.need) is not WorkerCapabilityNeed:
            raise TypeError("need must be WorkerCapabilityNeed")
        validate_capability_profile_ref(self.profile_ref)
        validate_worker_instance_id(self.worker_instance_id)
        if self.worker_instance_id != self.need.worker_instance_id:
            raise ValueError("worker_instance_id must match need.worker_instance_id")


@dataclass(frozen=True, slots=True)
class WorkerCapabilityCandidate:
    """Canonical immutable capability candidate — no provider object references."""

    candidate_id: str
    candidate_kind: WorkerCapabilityCandidateKind
    capability_ref: str
    source_domain: str
    operations: tuple[str, ...]
    risk_class: WorkerAutonomyLevel
    evidence_refs: tuple[ProblemReference, ...]
    discovered_at: datetime
    version: str | None = None
    operation_coverage: CapabilityOperationCoverage = CapabilityOperationCoverage.EXACT
    configuration_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            require_non_empty_text(self.candidate_id, label="candidate_id"),
        )
        if type(self.candidate_kind) is not WorkerCapabilityCandidateKind:
            raise TypeError("candidate_kind must be WorkerCapabilityCandidateKind")
        object.__setattr__(
            self,
            "capability_ref",
            validate_capability_ref(self.capability_ref),
        )
        object.__setattr__(
            self,
            "source_domain",
            require_non_empty_text(self.source_domain, label="source_domain"),
        )
        object.__setattr__(
            self,
            "operations",
            _validate_operations(self.operations),
        )
        if type(self.risk_class) is not WorkerAutonomyLevel:
            raise TypeError("risk_class must be WorkerAutonomyLevel")
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        object.__setattr__(
            self,
            "discovered_at",
            require_aware_utc(self.discovered_at, label="discovered_at"),
        )
        if self.version is not None:
            object.__setattr__(
                self,
                "version",
                require_non_empty_text(self.version, label="version"),
            )
        if type(self.operation_coverage) is not CapabilityOperationCoverage:
            raise TypeError("operation_coverage must be CapabilityOperationCoverage")
        if self.configuration_ref is not None:
            object.__setattr__(
                self,
                "configuration_ref",
                require_non_empty_text(self.configuration_ref, label="configuration_ref"),
            )
        if self.risk_class is WorkerAutonomyLevel.A4_AUTHORITY_CHANGE:
            raise ValueError("A4 candidates are not executable acquisition candidates")


def derive_worker_capability_candidate_id(
    *,
    candidate_kind: WorkerCapabilityCandidateKind,
    capability_ref: str,
    version: str | None = None,
    configuration_ref: str | None = None,
) -> str:
    parts = [candidate_kind.value, capability_ref]
    if version is not None:
        parts.append(version)
    if configuration_ref is not None:
        parts.append(configuration_ref)
    return ":".join(parts)


@dataclass(frozen=True, slots=True)
class WorkerCapabilityDiscoveryLayerOutcome:
    """Typed discovery layer result — distinguishes NO_MATCH from UNAVAILABLE."""

    disposition: CapabilityDiscoveryDisposition
    candidates: tuple[WorkerCapabilityCandidate, ...] = ()

    def __post_init__(self) -> None:
        if type(self.disposition) is not CapabilityDiscoveryDisposition:
            raise TypeError("disposition must be CapabilityDiscoveryDisposition")
        object.__setattr__(
            self,
            "candidates",
            freeze_tuple(self.candidates, label="candidates"),
        )
        for candidate in self.candidates:
            if type(candidate) is not WorkerCapabilityCandidate:
                raise TypeError("candidates must contain WorkerCapabilityCandidate")


@dataclass(frozen=True, slots=True)
class ResolvedWorkerCapabilityPolicy:
    """Resolved capability acquisition policy — does not grant authority.

  ``allowed_operation_patterns`` semantics:
  - empty tuple: no operations allowed (fail-closed)
  - ``*``: unrestricted operation admission
  - otherwise: exact operation ID or ``fnmatch`` glob match
    """

    profile_ref: CapabilityProfileRef
    allowed_candidate_kinds: frozenset[WorkerCapabilityCandidateKind]
    allowed_autonomy_levels: frozenset[WorkerAutonomyLevel]
    allowed_operation_patterns: tuple[str, ...]
    generated_capability_allowed: bool
    adaptive_integration_allowed: bool
    durable_change_allowed: bool

    def __post_init__(self) -> None:
        validate_capability_profile_ref(self.profile_ref)
        if not isinstance(self.allowed_candidate_kinds, frozenset):
            raise TypeError("allowed_candidate_kinds must be frozenset")
        if not isinstance(self.allowed_autonomy_levels, frozenset):
            raise TypeError("allowed_autonomy_levels must be frozenset")
        object.__setattr__(
            self,
            "allowed_operation_patterns",
            _validate_text_tuple(
                self.allowed_operation_patterns,
                label="allowed_operation_patterns",
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkerCapabilityDiscoveryResult:
    """Aggregate discovery outcome before final selection."""

    need: WorkerCapabilityNeed
    candidates: tuple[WorkerCapabilityCandidate, ...]
    disposition: CapabilityDiscoveryDisposition
    profile_ref: CapabilityProfileRef
    discovered_at: datetime

    def __post_init__(self) -> None:
        if type(self.need) is not WorkerCapabilityNeed:
            raise TypeError("need must be WorkerCapabilityNeed")
        object.__setattr__(
            self,
            "candidates",
            freeze_tuple(self.candidates, label="candidates"),
        )
        if type(self.disposition) is not CapabilityDiscoveryDisposition:
            raise TypeError("disposition must be CapabilityDiscoveryDisposition")
        validate_capability_profile_ref(self.profile_ref)
        object.__setattr__(
            self,
            "discovered_at",
            require_aware_utc(self.discovered_at, label="discovered_at"),
        )


@dataclass(frozen=True, slots=True)
class WorkerCapabilityAcquisitionDecision:
    """Immutable bounded acquisition decision — recommendation only."""

    decision_id: str
    worker_instance_id: WorkerInstanceId
    obstacle_id: str
    recovery_decision_id: str
    need_id: str
    disposition: CapabilityAcquisitionDisposition
    capability_profile_ref: CapabilityProfileRef
    reason_code: CapabilityAcquisitionReasonCode
    evidence_refs: tuple[ProblemReference, ...]
    decided_at: datetime
    decision_policy_version: str = ACQUISITION_DECISION_POLICY_VERSION
    selected_candidate: WorkerCapabilityCandidate | None = None
    autonomy_level: WorkerAutonomyLevel | None = None
    codecraft_profile_ref: CodecraftProfileRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_id",
            require_non_empty_text(self.decision_id, label="decision_id"),
        )
        validate_worker_instance_id(self.worker_instance_id)
        object.__setattr__(
            self,
            "obstacle_id",
            require_non_empty_text(self.obstacle_id, label="obstacle_id"),
        )
        object.__setattr__(
            self,
            "recovery_decision_id",
            require_non_empty_text(
                self.recovery_decision_id,
                label="recovery_decision_id",
            ),
        )
        object.__setattr__(
            self,
            "need_id",
            require_non_empty_text(self.need_id, label="need_id"),
        )
        if type(self.disposition) is not CapabilityAcquisitionDisposition:
            raise TypeError("disposition must be CapabilityAcquisitionDisposition")
        validate_capability_profile_ref(self.capability_profile_ref)
        if type(self.reason_code) is not CapabilityAcquisitionReasonCode:
            raise TypeError("reason_code must be CapabilityAcquisitionReasonCode")
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        object.__setattr__(
            self,
            "decided_at",
            require_aware_utc(self.decided_at, label="decided_at"),
        )
        object.__setattr__(
            self,
            "decision_policy_version",
            require_non_empty_text(
                self.decision_policy_version,
                label="decision_policy_version",
            ),
        )
        if self.selected_candidate is not None:
            if type(self.selected_candidate) is not WorkerCapabilityCandidate:
                raise TypeError("selected_candidate must be WorkerCapabilityCandidate")
        if self.autonomy_level is not None:
            if type(self.autonomy_level) is not WorkerAutonomyLevel:
                raise TypeError("autonomy_level must be WorkerAutonomyLevel")
        if self.codecraft_profile_ref is not None:
            validate_codecraft_profile_ref(self.codecraft_profile_ref)
        _validate_acquisition_decision_invariants(self)


@dataclass(frozen=True, slots=True)
class WorkerCapabilityAcquisitionRequest:
    """Typed acquisition request correlated to recovery decision evidence."""

    need: WorkerCapabilityNeed
    recovery_decision: WorkerRecoveryDecision
    capability_profile_ref: CapabilityProfileRef
    codecraft_profile_ref: CodecraftProfileRef | None = None

    def __post_init__(self) -> None:
        if type(self.need) is not WorkerCapabilityNeed:
            raise TypeError("need must be WorkerCapabilityNeed")
        if type(self.recovery_decision) is not WorkerRecoveryDecision:
            raise TypeError("recovery_decision must be WorkerRecoveryDecision")
        validate_capability_profile_ref(self.capability_profile_ref)
        if self.codecraft_profile_ref is not None:
            validate_codecraft_profile_ref(self.codecraft_profile_ref)
        if self.need.recovery_decision_id != self.recovery_decision.decision_id:
            raise ValueError("need.recovery_decision_id must match recovery_decision.decision_id")
        if self.need.obstacle_id != self.recovery_decision.obstacle_id:
            raise ValueError("need.obstacle_id must match recovery_decision.obstacle_id")
        if not self.recovery_decision.obstacle_id.startswith(
            f"{self.need.worker_instance_id}:",
        ):
            raise ValueError(
                "need.worker_instance_id must correlate with recovery_decision obstacle_id",
            )


@dataclass(frozen=True, slots=True)
class WorkerCapabilityAcquisitionResult:
    """Acquisition service outcome bundle."""

    disposition: CapabilityAcquisitionDisposition
    decision: WorkerCapabilityAcquisitionDecision | None
    discovery: WorkerCapabilityDiscoveryResult | None = None


def derive_worker_capability_acquisition_decision_id(
    *,
    worker_instance_id: WorkerInstanceId,
    obstacle_id: str,
    recovery_decision_id: str,
    need_id: str,
    capability_profile_version: int,
    selected_candidate_id: str | None,
    decision_policy_version: str = ACQUISITION_DECISION_POLICY_VERSION,
) -> str:
    candidate_part = selected_candidate_id or "none"
    return (
        f"{worker_instance_id}:"
        f"{obstacle_id}:"
        f"{recovery_decision_id}:"
        f"{need_id}:"
        f"{capability_profile_version}:"
        f"{candidate_part}:"
        f"{decision_policy_version}"
    )


def is_capability_acquisition_recovery_strategy(strategy: RecoveryStrategy) -> bool:
    return strategy in _CAPABILITY_ACQUISITION_STRATEGIES


def a4_never_self_authorized(autonomy_level: WorkerAutonomyLevel | None) -> bool:
    """Hard invariant gate — A4 must never be auto-executable."""

    return autonomy_level is WorkerAutonomyLevel.A4_AUTHORITY_CHANGE


def operation_allowed(operation: str, allowed_patterns: tuple[str, ...]) -> bool:
    """Return whether ``operation`` is admitted by capability policy patterns.

    Empty ``allowed_patterns`` means no operations are allowed (fail-closed).
    Pattern ``*`` admits any operation. Otherwise exact match or ``fnmatch`` glob.
    """

    if not allowed_patterns:
        return False
    for pattern in allowed_patterns:
        if pattern == "*":
            return True
        if operation == pattern:
            return True
        if fnmatch.fnmatch(operation, pattern):
            return True
    return False


def operations_allowed_by_policy(
    required_operations: tuple[str, ...],
    allowed_patterns: tuple[str, ...],
) -> bool:
    """Return True when every required operation is admitted by policy patterns."""

    return all(operation_allowed(item, allowed_patterns) for item in required_operations)


def autonomy_level_allowed(
    autonomy_level: WorkerAutonomyLevel,
    allowed_levels: frozenset[WorkerAutonomyLevel],
) -> bool:
    """Return True when ``autonomy_level`` is admitted by resolved capability policy."""

    return autonomy_level in allowed_levels


def _validate_operations(value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    frozen = freeze_tuple(value, label="required_operations")
    if not frozen:
        raise ValueError("required_operations must be non-empty")
    return tuple(validate_capability_operation(item) for item in frozen)


def _validate_text_tuple(value: tuple[str, ...] | list[str], *, label: str) -> tuple[str, ...]:
    frozen = freeze_tuple(value, label=label)
    return tuple(require_non_empty_text(item, label=label) for item in frozen)


def _validate_acquisition_decision_invariants(
    decision: WorkerCapabilityAcquisitionDecision,
) -> None:
    disposition = decision.disposition
    candidate = decision.selected_candidate
    autonomy = decision.autonomy_level

    if disposition is CapabilityAcquisitionDisposition.USE_EXISTING:
        if candidate is None or autonomy is not WorkerAutonomyLevel.A0_KNOWN_CAPABILITY:
            raise ValueError("USE_EXISTING requires selected_candidate and autonomy A0")
        if candidate.candidate_kind is WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION:
            raise ValueError("USE_EXISTING cannot select EXISTING_CONFIGURATION candidate")

    if disposition is CapabilityAcquisitionDisposition.CONFIGURE_EXISTING:
        if candidate is None or autonomy is not WorkerAutonomyLevel.A0_KNOWN_CAPABILITY:
            raise ValueError("CONFIGURE_EXISTING requires selected_candidate and autonomy A0")
        if candidate.candidate_kind is not WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION:
            raise ValueError("CONFIGURE_EXISTING requires EXISTING_CONFIGURATION candidate")

    if disposition is CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE:
        if autonomy is not WorkerAutonomyLevel.A1_EPHEMERAL_SAFE:
            raise ValueError("EPHEMERAL_GENERATION_CANDIDATE requires autonomy A1")

    if disposition is CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE:
        if autonomy is not WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE:
            raise ValueError("SCOPED_ADAPTATION_CANDIDATE requires autonomy A2")

    if disposition is CapabilityAcquisitionDisposition.PRODUCTION_CHANGE_REQUIRED:
        if autonomy is not WorkerAutonomyLevel.A3_PRODUCTION_CHANGE:
            raise ValueError("PRODUCTION_CHANGE_REQUIRED requires autonomy A3")

    if disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED:
        if autonomy is not WorkerAutonomyLevel.A4_AUTHORITY_CHANGE:
            raise ValueError("AUTHORITY_CHANGE_REQUIRED requires autonomy A4")
        if candidate is not None:
            raise ValueError("A4 decisions must not carry executable selected_candidate")

    if autonomy is WorkerAutonomyLevel.A4_AUTHORITY_CHANGE:
        if candidate is not None:
            raise ValueError("A4 autonomy must not carry executable selected_candidate")
        if disposition not in {
            CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED,
            CapabilityAcquisitionDisposition.ESCALATE,
        }:
            raise ValueError("A4 autonomy is only valid for authority escalation dispositions")
