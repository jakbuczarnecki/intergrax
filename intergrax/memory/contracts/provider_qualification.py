# © Artur Czarnecki. All rights reserved.

"""Memory provider production qualification contracts (MEM-ENT-13)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

__all__ = [
    "MemoryProviderCapabilityKind",
    "MemoryProviderCheckResult",
    "MemoryProviderCheckSeverity",
    "MemoryProviderCapabilityQualification",
    "MemoryProviderDescriptor",
    "MemoryProviderQualificationCheck",
    "MemoryProviderQualificationContext",
    "MemoryProviderQualificationFailureReason",
    "MemoryProviderQualificationRequest",
    "MemoryProviderQualificationResult",
    "MemoryProviderQualificationStatus",
]


class MemoryProviderCapabilityKind(str, Enum):
    USER_PROFILE_STORE = "user_profile_store"
    SESSION_STORAGE = "session_storage"
    SESSION_TURN_INDEX_STORE = "session_turn_index_store"
    ENTITY_TEMPORAL_MEMORY_STORE = "entity_temporal_memory_store"
    PROCEDURE_MEMORY_STORE = "procedure_memory_store"
    LONG_HORIZON_MEMORY_STORE = "long_horizon_memory_store"
    CONDITIONAL_DOCUMENT_STORE = "conditional_document_store"


class MemoryProviderQualificationStatus(str, Enum):
    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"
    NOT_SUPPORTED = "not_supported"
    BLOCKED = "blocked"


class MemoryProviderQualificationFailureReason(str, Enum):
    CONTRACT_MISMATCH = "contract_mismatch"
    TENANT_ISOLATION_FAILURE = "tenant_isolation_failure"
    USER_ISOLATION_FAILURE = "user_isolation_failure"
    WORKSPACE_ISOLATION_FAILURE = "workspace_isolation_failure"
    REVISION_SEMANTICS_FAILURE = "revision_semantics_failure"
    DELETE_ISOLATION_FAILURE = "delete_isolation_failure"
    TEMPORAL_SEMANTICS_FAILURE = "temporal_semantics_failure"
    CAS_FAILURE = "cas_failure"
    IDEMPOTENCY_FAILURE = "idempotency_failure"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    INVALID_FAILURE_BEHAVIOR = "invalid_failure_behavior"
    PROJECTION_SCOPE_FAILURE = "projection_scope_failure"
    PLUGIN_LOAD_FAILURE = "plugin_load_failure"
    CLEANUP_FAILURE = "cleanup_failure"
    MATERIALIZATION_FAILURE = "materialization_failure"


class MemoryProviderCheckSeverity(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class MemoryProviderDescriptor:
    """Stable provider identity for qualification evidence."""

    provider_id: str
    capabilities: tuple[MemoryProviderCapabilityKind, ...]
    provider_version: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProviderQualificationContext:
    """Isolated scope identifiers for a single qualification run."""

    qualification_run_id: str
    tenant_qualification_id: str
    user_qualification_id: str
    workspace_qualification_id: str
    reference_time_iso: str


@dataclass(frozen=True, slots=True)
class MemoryProviderQualificationRequest:
    required_capabilities: tuple[MemoryProviderCapabilityKind, ...] = ()
    optional_capabilities: tuple[MemoryProviderCapabilityKind, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryProviderCheckResult:
    check_id: str
    capability: MemoryProviderCapabilityKind
    severity: MemoryProviderCheckSeverity
    passed: bool
    reason_code: MemoryProviderQualificationFailureReason | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProviderCapabilityQualification:
    capability: MemoryProviderCapabilityKind
    status: MemoryProviderQualificationStatus
    checks_executed: int
    checks_passed: int
    checks_failed: int
    check_results: tuple[MemoryProviderCheckResult, ...]
    reason_codes: tuple[MemoryProviderQualificationFailureReason, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryProviderQualificationResult:
    descriptor: MemoryProviderDescriptor
    qualification_run_id: str
    reference_time_iso: str
    status: MemoryProviderQualificationStatus
    capability_results: tuple[MemoryProviderCapabilityQualification, ...]
    reason_codes: tuple[MemoryProviderQualificationFailureReason, ...] = ()


class MemoryProviderQualificationCheck(Protocol):
    """Typed behavioral check executed against a materialized provider instance."""

    @property
    def check_id(self) -> str: ...

    @property
    def capability(self) -> MemoryProviderCapabilityKind: ...

    @property
    def severity(self) -> MemoryProviderCheckSeverity: ...

    async def run(
        self,
        instance: object,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult: ...
