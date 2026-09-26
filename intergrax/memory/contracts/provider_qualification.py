# © Artur Czarnecki. All rights reserved.

"""Memory provider production qualification contracts (MEM-ENT-13)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.contracts.long_horizon_memory import LongHorizonMemoryStore
from intergrax.memory.contracts.procedural_memory import ProcedureMemoryStore
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore

from intergrax.memory.contracts.user_profile_store import UserProfileStore

__all__ = [
    "MemoryProviderCapabilityKind",
    "MemoryProviderCheckResult",
    "MemoryProviderCheckSeverity",
    "MemoryProviderCapabilityQualification",
    "MemoryProviderDescriptor",
    "MemoryProviderQualificationCheckMetadata",
    "MemoryProviderQualificationContext",
    "MemoryProviderQualificationFailureReason",
    "MemoryProviderQualificationRequest",
    "MemoryProviderQualificationResult",
    "MemoryProviderQualificationStatus",
    "UserProfileStoreQualificationCheck",
    "EntityTemporalMemoryStoreQualificationCheck",
    "ProcedureMemoryStoreQualificationCheck",
    "LongHorizonMemoryStoreQualificationCheck",
    "SessionTurnIndexStoreQualificationCheck",
    "validate_memory_provider_descriptor",
    "validate_memory_provider_qualification_request",
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
    QUALIFICATION_COVERAGE_MISSING = "qualification_coverage_missing"
    SOURCE_FIDELITY_FAILURE = "source_fidelity_failure"
    DURABILITY_FAILURE = "durability_failure"


class MemoryProviderCheckSeverity(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class MemoryProviderDescriptor:
    """Stable provider identity for qualification evidence."""

    provider_id: str
    capabilities: tuple[MemoryProviderCapabilityKind, ...]
    provider_version: str | None = None
    backing_provider_id: str | None = None
    backing_provider_version: str | None = None


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


class MemoryProviderQualificationCheckMetadata(Protocol):
    """Shared check identity fields (per-capability execution protocols are typed)."""

    @property
    def check_id(self) -> str: ...

    @property
    def capability(self) -> MemoryProviderCapabilityKind: ...

    @property
    def severity(self) -> MemoryProviderCheckSeverity: ...


class UserProfileStoreQualificationCheck(MemoryProviderQualificationCheckMetadata, Protocol):
    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult: ...


class EntityTemporalMemoryStoreQualificationCheck(
    MemoryProviderQualificationCheckMetadata, Protocol
):
    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult: ...


class ProcedureMemoryStoreQualificationCheck(MemoryProviderQualificationCheckMetadata, Protocol):
    async def run(
        self,
        instance: ProcedureMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult: ...


class LongHorizonMemoryStoreQualificationCheck(MemoryProviderQualificationCheckMetadata, Protocol):
    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult: ...


class SessionTurnIndexStoreQualificationCheck(MemoryProviderQualificationCheckMetadata, Protocol):
    async def run(
        self,
        instance: SessionTurnIndexStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult: ...


def validate_memory_provider_descriptor(descriptor: MemoryProviderDescriptor) -> None:
    if not descriptor.provider_id.strip():
        raise ValueError("MemoryProviderDescriptor.provider_id must be non-empty")
    if len(descriptor.capabilities) != len(set(descriptor.capabilities)):
        raise ValueError("MemoryProviderDescriptor.capabilities must not contain duplicates")


def validate_memory_provider_qualification_request(
    request: MemoryProviderQualificationRequest,
) -> None:
    required = request.required_capabilities
    optional = request.optional_capabilities
    if len(required) != len(set(required)):
        raise ValueError("required_capabilities must not contain duplicates")
    if len(optional) != len(set(optional)):
        raise ValueError("optional_capabilities must not contain duplicates")
    overlap = set(required) & set(optional)
    if overlap:
        raise ValueError(
            "capabilities cannot appear in both required_capabilities and optional_capabilities"
        )
