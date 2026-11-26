# © Artur Czarnecki. All rights reserved.

"""Host-owned memory provider admission (MEM-FINAL-AUDIT-5A / 5A-R)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.persistence_topology import (
    PersistenceTopology,
    topology_satisfies,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentStore,
)
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    MemoryProviderDurability,
    MemoryStoreProviderMetadata,
    UserProfileStoreProviderClassification,
    classify_user_profile_store_provider,
    evaluate_production_persistent_user_profile_admission,
    evaluate_production_session_turn_index_store_admission,
    lookup_trusted_memory_provider_qualification_evidence,
    lookup_trusted_user_profile_durability_evidence,
    lookup_trusted_user_profile_qualification_evidence,
)
from intergrax.memory.contracts.provider_admission_evidence import (
    MemoryProviderAdmissionEvidenceContext,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidenceLookup,
    MemoryProviderDurabilityEvidenceRegistry,
    MemoryProviderDurabilityEvidenceResolveStatus,
    trusted_memory_provider_restart_reopen_admitted,
)
from intergrax.memory.contracts.provider_identity import (
    MemoryProviderIdentity,
    MemoryProviderIdentitySource,
    memory_provider_backing_identity_mismatch,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidenceLookup,
    MemoryProviderQualificationEvidenceRegistry,
    MemoryProviderQualificationEvidenceResolveStatus,
)
from intergrax.memory.user_profile_store import UserProfileStore

if TYPE_CHECKING:
    from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
    from intergrax.runtime.execution.session_host_composition import SessionStorage
    from intergrax.runtime.organization.organization_profile_store import OrganizationProfileStore


class _EmptyDurabilityEvidenceRegistry:
    def resolve(
        self,
        provider_id: str,
        capability: object,
        provider_version: str | None = None,
        backing_provider_id: str | None = None,
        backing_provider_version: str | None = None,
    ):
        _ = (
            provider_id,
            capability,
            provider_version,
            backing_provider_id,
            backing_provider_version,
        )
        return MemoryProviderDurabilityEvidenceLookup(
            resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.MISSING,
        )


class _EmptyQualificationEvidenceRegistry:
    def resolve(
        self,
        provider_id: str,
        capability: object,
        provider_version: str | None = None,
        backing_provider_id: str | None = None,
        backing_provider_version: str | None = None,
    ):
        from intergrax.memory.contracts.provider_qualification_evidence import (
            MemoryProviderQualificationEvidenceLookup,
            MemoryProviderQualificationEvidenceResolveStatus,
        )

        _ = (
            provider_id,
            capability,
            provider_version,
            backing_provider_id,
            backing_provider_version,
        )
        return MemoryProviderQualificationEvidenceLookup(
            resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
        )


def persistent_canonical_user_profile_memory_required(
    env: ApplicationEnvironmentProfile,
) -> bool:
    memory_profile = env.memory_profile
    return memory_profile.enable_user_memory or memory_profile.enable_long_term_memory


def tenant_scoped_memory_capability_enabled(env: ApplicationEnvironmentProfile) -> bool:
    """True when user or long-term memory requires a tenant in STRICT fail-closed posture.

    Organization memory scope is enforced by org/session admission (PT-FC2), not here.
    """
    memory_profile = env.memory_profile
    return memory_profile.enable_user_memory or memory_profile.enable_long_term_memory


class MemoryTenantScopeError(ValueError):
    """Raised when fail-closed posture composes tenant-scoped memory without a tenant."""

    def __init__(
        self,
        message: str = "memory capabilities require a tenant in fail-closed posture",
    ) -> None:
        super().__init__(message)


def validate_memory_tenant_scope_admission(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None,
) -> None:
    """Refuse unscoped user/LTM memory composition in STRICT LAB posture (EX-D2-06).

    PRODUCT hosts use org/session admission for store durability and org memory; organization
    memory is not a tenant-scope signal (PT-FC2).
    """
    if env.application_profile is not ApplicationProfile.LAB:
        return
    if env.execution_mode is not ExecutionMode.STRICT:
        return
    if not wiring_fails_closed(env):
        return
    if not tenant_scoped_memory_capability_enabled(env):
        return
    if tenant_id is not None and tenant_id.strip():
        return
    raise MemoryTenantScopeError()


def session_turn_index_vector_memory_required(
    env: ApplicationEnvironmentProfile,
) -> bool:
    return env.memory_profile.enable_session_vector_index


@dataclass(frozen=True, slots=True)
class MemoryProviderAdmissionDecision:
    admitted: bool
    reason_code: MemoryProviderAdmissionReasonCode | None


class MemoryProviderAdmissionPolicy(Protocol):
    """Replaceable admission strategy for non-product paths (LAB / harness)."""

    def evaluate_user_profile_store(
        self,
        classification: UserProfileStoreProviderClassification,
    ) -> MemoryProviderAdmissionDecision: ...


@dataclass(frozen=True, slots=True)
class DefaultProductionMemoryProviderAdmissionPolicy:
    """LAB harness: admission not enforced when ``admission_enforced`` is false."""

    admission_enforced: bool

    @classmethod
    def for_environment(
        cls,
        env: ApplicationEnvironmentProfile,
    ) -> DefaultProductionMemoryProviderAdmissionPolicy:
        return cls(
            admission_enforced=env.application_profile is ApplicationProfile.PRODUCT,
        )

    def evaluate_user_profile_store(
        self,
        classification: UserProfileStoreProviderClassification,
    ) -> MemoryProviderAdmissionDecision:
        if not self.admission_enforced:
            return MemoryProviderAdmissionDecision(admitted=True, reason_code=None)
        missing = MemoryProviderQualificationEvidenceLookup(
            resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
        )
        missing_durability = MemoryProviderDurabilityEvidenceLookup(
            resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.MISSING,
        )
        evaluation = evaluate_production_persistent_user_profile_admission(
            classification,
            None,
            missing,
            missing_durability,
        )
        return MemoryProviderAdmissionDecision(
            admitted=evaluation.admitted,
            reason_code=evaluation.reason_code,
        )


def validate_memory_platform_wiring_admission(
    env: ApplicationEnvironmentProfile,
    user_profile_store: UserProfileStore,
    *,
    user_profile_store_identity: MemoryProviderIdentity | None = None,
    policy: MemoryProviderAdmissionPolicy | None = None,
    qualification_evidence_registry: MemoryProviderQualificationEvidenceRegistry | None = None,
    durability_evidence_registry: MemoryProviderDurabilityEvidenceRegistry | None = None,
    admission_evidence: MemoryProviderAdmissionEvidenceContext | None = None,
) -> None:
    """Pure admission gate on final user profile store; no provider mutation."""
    if not persistent_canonical_user_profile_memory_required(env):
        return

    classification = classify_user_profile_store_provider(user_profile_store)
    if admission_evidence is not None:
        registry = admission_evidence.qualification_registry
        durability_registry = admission_evidence.durability_registry
    else:
        registry = qualification_evidence_registry or _EmptyQualificationEvidenceRegistry()
        durability_registry = durability_evidence_registry or _EmptyDurabilityEvidenceRegistry()

    if env.application_profile is ApplicationProfile.PRODUCT:
        if user_profile_store_identity is None:
            evidence_lookup = MemoryProviderQualificationEvidenceLookup(
                resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
            )
            durability_lookup = MemoryProviderDurabilityEvidenceLookup(
                resolve_status=MemoryProviderDurabilityEvidenceResolveStatus.MISSING,
            )
        else:
            evidence_lookup = lookup_trusted_user_profile_qualification_evidence(
                registry,
                user_profile_store_identity,
            )
            durability_lookup = lookup_trusted_user_profile_durability_evidence(
                durability_registry,
                user_profile_store_identity,
            )
        evaluation = evaluate_production_persistent_user_profile_admission(
            classification,
            user_profile_store_identity,
            evidence_lookup,
            durability_lookup,
        )
        if not evaluation.admitted:
            reason = evaluation.reason_code or MemoryProviderAdmissionReasonCode.PROVIDER_MISSING
            declared_id = classification.provider_id
            trusted_id = user_profile_store_identity.provider_id if user_profile_store_identity is not None else None
            raise MemoryProviderAdmissionError(
                capability=classification.capability,
                execution_mode=env.execution_mode.value,
                application_profile=env.application_profile.value,
                reason_code=reason,
                provider_id=evaluation.provider_id,
                durability=classification.durability,
                declared_qualification_status=classification.declared_qualification_status,
                trusted_qualification_status=evaluation.trusted_qualification_status,
                reference_only=classification.reference_only,
                qualification_run_id=evaluation.qualification_run_id,
                declared_provider_id=declared_id,
                trusted_provider_id=trusted_id,
                trusted_durability_status=evaluation.trusted_durability_status,
                durability_run_id=evaluation.durability_run_id,
            )
        return

    resolved_policy = policy or DefaultProductionMemoryProviderAdmissionPolicy.for_environment(
        env,
    )
    decision = resolved_policy.evaluate_user_profile_store(classification)
    if decision.admitted:
        return
    reason = decision.reason_code or MemoryProviderAdmissionReasonCode.PROVIDER_MISSING
    raise MemoryProviderAdmissionError(
        capability=classification.capability,
        execution_mode=env.execution_mode.value,
        application_profile=env.application_profile.value,
        reason_code=reason,
        provider_id=classification.provider_id,
        durability=classification.durability,
        declared_qualification_status=classification.declared_qualification_status,
        trusted_qualification_status=None,
        reference_only=classification.reference_only,
    )


def validate_session_turn_index_store_admission(
    env: ApplicationEnvironmentProfile,
    *,
    provider_identity: MemoryProviderIdentity | None,
    qualification_evidence_registry: MemoryProviderQualificationEvidenceRegistry | None = None,
    admission_evidence: MemoryProviderAdmissionEvidenceContext | None = None,
) -> None:
    """Fail closed on PRODUCT when episodic STI is enabled without trusted qualification."""
    if not session_turn_index_vector_memory_required(env):
        return
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return

    if admission_evidence is not None:
        registry = admission_evidence.qualification_registry
    else:
        registry = qualification_evidence_registry or _EmptyQualificationEvidenceRegistry()

    if provider_identity is None:
        evidence_lookup = MemoryProviderQualificationEvidenceLookup(
            resolve_status=MemoryProviderQualificationEvidenceResolveStatus.MISSING,
        )
    else:
        evidence_lookup = lookup_trusted_memory_provider_qualification_evidence(
            registry,
            provider_identity,
        )

    evaluation = evaluate_production_session_turn_index_store_admission(
        provider_identity,
        evidence_lookup,
    )
    if evaluation.admitted:
        return

    reason = evaluation.reason_code or MemoryProviderAdmissionReasonCode.PROVIDER_MISSING
    evidence = evidence_lookup.evidence
    raise MemoryProviderAdmissionError(
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        execution_mode=env.execution_mode.value,
        application_profile=env.application_profile.value,
        reason_code=reason,
        provider_id=evaluation.provider_id,
        durability=MemoryProviderDurability.UNKNOWN,
        declared_qualification_status=MemoryProviderQualificationStatus.NOT_QUALIFIED,
        trusted_qualification_status=evaluation.trusted_qualification_status,
        reference_only=False,
        qualification_run_id=evaluation.qualification_run_id,
        trusted_provider_id=(provider_identity.provider_id if provider_identity is not None else None),
        trusted_backing_provider_id=evaluation.backing_provider_id,
        evidence_backing_provider_id=(evidence.backing_provider_id if evidence is not None else None),
    )


_BUILTIN_SQLITE_SESSION_STORAGE_ID = "sqlite.session_storage"
_BUILTIN_SQLITE_ORGANIZATION_PROFILE_STORE_ID = "sqlite.organization_profile"
_BUILTIN_DOCUMENT_STORE_SESSION_STORAGE_ID = "document_store.session_storage"


class MemoryOrgSessionAdmissionReason(StrEnum):
    """Reason codes of the organization-store and session-storage admission (T-IGX-043-B)."""

    ORG_STORE_UNSUPPORTED_BACKEND = "ORG_STORE_UNSUPPORTED_BACKEND"
    ORG_STORE_MISSING = "ORG_STORE_MISSING"
    ORG_STORE_NOT_DURABLE = "ORG_STORE_NOT_DURABLE"
    SESSION_STORAGE_NOT_DURABLE = "SESSION_STORAGE_NOT_DURABLE"
    EPISODIC_RECALL_NOT_BOUND = "EPISODIC_RECALL_NOT_BOUND"


class MemoryOrgSessionAdmissionError(ValueError):
    """Raised when a fail-closed posture composes an ephemeral or missing org/session store."""

    def __init__(self, reason: MemoryOrgSessionAdmissionReason, message: str) -> None:
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason


def organization_profile_store_required(env: ApplicationEnvironmentProfile) -> bool:
    """Organization memory needs an organization profile store."""
    return env.memory_profile.enable_org_memory


def _durability_evidence_identity_matches(
    identity: MemoryProviderIdentity,
    evidence_lookup: MemoryProviderDurabilityEvidenceLookup,
) -> bool:
    evidence = evidence_lookup.evidence
    if evidence is None:
        return False
    if evidence.capability is not identity.capability:
        return False
    if evidence.provider_id != identity.provider_id:
        return False
    return not memory_provider_backing_identity_mismatch(
        identity,
        backing_provider_id=evidence.backing_provider_id,
        backing_provider_version=evidence.backing_provider_version,
    )


def _provider_has_restart_reopen_durability(
    identity: MemoryProviderIdentity | None,
    durability_registry: MemoryProviderDurabilityEvidenceRegistry,
) -> bool:
    if identity is None:
        return False
    lookup = lookup_trusted_user_profile_durability_evidence(
        durability_registry,
        identity,
    )
    if not trusted_memory_provider_restart_reopen_admitted(lookup):
        return False
    return _durability_evidence_identity_matches(identity, lookup)


def _resolve_organization_profile_store_provider_identity(
    store: OrganizationProfileStore,
) -> MemoryProviderIdentity | None:
    from intergrax.runtime.organization.stores.in_memory_organization_profile_store import (
        InMemoryOrganizationProfileStore,
    )
    from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
        SQLiteOrganizationProfileStore,
    )

    if isinstance(store, InMemoryOrganizationProfileStore):
        return None
    if isinstance(store, SQLiteOrganizationProfileStore):
        return MemoryProviderIdentity(
            provider_id=_BUILTIN_SQLITE_ORGANIZATION_PROFILE_STORE_ID,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            source=MemoryProviderIdentitySource.BUILT_IN,
        )
    if isinstance(store, MemoryStoreProviderMetadata):
        return MemoryProviderIdentity(
            provider_id=store.memory_provider_id,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            source=MemoryProviderIdentitySource.DIRECT_INJECTION,
            provider_version=store.memory_provider_version,
        )
    return None


def _resolve_session_storage_provider_identity(
    storage: SessionStorage,
    *,
    session_storage_plugin_id: str | None,
    document_store_backing_provider_id: str | None,
) -> MemoryProviderIdentity | None:
    from intergrax.runtime.execution.session_host_composition import SQLiteSessionStorage
    from intergrax.runtime.nexus.session.document_store_session_storage import (
        DocumentStoreSessionStorage,
    )
    from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage

    if isinstance(storage, InMemorySessionStorage):
        return None
    if isinstance(storage, SQLiteSessionStorage):
        return MemoryProviderIdentity(
            provider_id=_BUILTIN_SQLITE_SESSION_STORAGE_ID,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            source=MemoryProviderIdentitySource.BUILT_IN,
        )
    if isinstance(storage, DocumentStoreSessionStorage):
        return MemoryProviderIdentity(
            provider_id=_BUILTIN_DOCUMENT_STORE_SESSION_STORAGE_ID,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            source=MemoryProviderIdentitySource.BUILT_IN,
            backing_provider_id=document_store_backing_provider_id,
        )
    if session_storage_plugin_id is not None:
        return MemoryProviderIdentity(
            provider_id=session_storage_plugin_id,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            source=MemoryProviderIdentitySource.PLUGIN,
        )
    if isinstance(storage, MemoryStoreProviderMetadata):
        return MemoryProviderIdentity(
            provider_id=storage.memory_provider_id,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            source=MemoryProviderIdentitySource.DIRECT_INJECTION,
            provider_version=storage.memory_provider_version,
        )
    return None


def _session_storage_backing_document_store(
    storage: SessionStorage,
) -> DocumentStore | None:
    from intergrax.runtime.nexus.session.document_store_session_storage import (
        DocumentStoreSessionStorage,
    )

    if isinstance(storage, DocumentStoreSessionStorage):
        return storage.backing_document_store
    return None


def _validate_document_store_shared_posture(
    env: ApplicationEnvironmentProfile,
    *,
    document_store: DocumentStore | None,
    mechanism: str,
) -> None:
    required = env.meta.required_persistence_topology
    if required is not PersistenceTopology.SHARED_MULTI_HOST:
        return
    if document_store is None:
        raise MemoryOrgSessionAdmissionError(
            MemoryOrgSessionAdmissionReason.SESSION_STORAGE_NOT_DURABLE,
            f"{mechanism} requires shared multi-host persistence topology but has no document-store backing",
        )
    provided = getattr(document_store, "persistence_topology", None)
    if not isinstance(provided, PersistenceTopology) or not topology_satisfies(
        required,
        provided,
    ):
        raise MemoryOrgSessionAdmissionError(
            MemoryOrgSessionAdmissionReason.SESSION_STORAGE_NOT_DURABLE,
            f"{mechanism} does not provide required shared multi-host persistence topology",
        )
    if not isinstance(document_store, ConditionalDocumentStore):
        raise MemoryOrgSessionAdmissionError(
            MemoryOrgSessionAdmissionReason.SESSION_STORAGE_NOT_DURABLE,
            f"{mechanism} requires conditional compare-and-swap writes but backing store "
            "does not implement ConditionalDocumentStore",
        )
    replace_if_match = getattr(document_store, "replace_if_match", None)
    if not callable(replace_if_match):
        raise MemoryOrgSessionAdmissionError(
            MemoryOrgSessionAdmissionReason.SESSION_STORAGE_NOT_DURABLE,
            f"{mechanism} requires replace_if_match but backing store does not expose it",
        )


def _organization_store_is_durable(
    store: OrganizationProfileStore,
    *,
    durability_registry: MemoryProviderDurabilityEvidenceRegistry,
) -> bool:
    identity = _resolve_organization_profile_store_provider_identity(store)
    return _provider_has_restart_reopen_durability(identity, durability_registry)


def _session_storage_is_durable(
    storage: SessionStorage,
    *,
    durability_registry: MemoryProviderDurabilityEvidenceRegistry,
    session_storage_plugin_id: str | None,
    document_store_backing_provider_id: str | None,
) -> bool:
    """Admit session storage only on trusted restart-reopen durability evidence (not declared flags)."""
    identity = _resolve_session_storage_provider_identity(
        storage,
        session_storage_plugin_id=session_storage_plugin_id,
        document_store_backing_provider_id=document_store_backing_provider_id,
    )
    return _provider_has_restart_reopen_durability(identity, durability_registry)


def validate_organization_and_session_admission(
    env: ApplicationEnvironmentProfile,
    *,
    session_storage: SessionStorage,
    organization_profile_store: OrganizationProfileStore | None,
    durability_evidence_registry: MemoryProviderDurabilityEvidenceRegistry | None = None,
    session_storage_plugin_id: str | None = None,
    document_store_backing_provider_id: str | None = None,
) -> None:
    """Pure admission gate on the final session storage and organization store.

    Applies in PRODUCT or STRICT postures (DP-24). Session storage is checked only when
    persistent user or long-term memory is required (decision Q2 = A); the organization store
    only when organization memory is enabled. No provider mutation.
    """
    if not wiring_fails_closed(env):
        return
    durability_registry = durability_evidence_registry or _EmptyDurabilityEvidenceRegistry()
    if organization_profile_store_required(env):
        if organization_profile_store is None:
            raise MemoryOrgSessionAdmissionError(
                MemoryOrgSessionAdmissionReason.ORG_STORE_MISSING,
                "organization memory is enabled but no organization profile store is composed",
            )
        if not _organization_store_is_durable(
            organization_profile_store,
            durability_registry=durability_registry,
        ):
            raise MemoryOrgSessionAdmissionError(
                MemoryOrgSessionAdmissionReason.ORG_STORE_NOT_DURABLE,
                f"organization profile store {type(organization_profile_store).__name__} "
                "lacks trusted restart-reopen durability evidence",
            )
    if persistent_canonical_user_profile_memory_required(env):
        if not _session_storage_is_durable(
            session_storage,
            durability_registry=durability_registry,
            session_storage_plugin_id=session_storage_plugin_id,
            document_store_backing_provider_id=document_store_backing_provider_id,
        ):
            raise MemoryOrgSessionAdmissionError(
                MemoryOrgSessionAdmissionReason.SESSION_STORAGE_NOT_DURABLE,
                f"session storage {type(session_storage).__name__} lacks trusted "
                "restart-reopen durability evidence while persistent memory is required",
            )
        backing = _session_storage_backing_document_store(session_storage)
        _validate_document_store_shared_posture(
            env,
            document_store=backing,
            mechanism="session storage",
        )


def validate_episodic_recall_composition_admission(
    env: ApplicationEnvironmentProfile,
    *,
    session_turn_index_store: SessionTurnIndexStore | None,
) -> None:
    """Fail closed when episodic recall is enabled but no STI implementation is composed."""
    if not env.memory_profile.enable_session_vector_index:
        return
    if not wiring_fails_closed(env):
        return
    if session_turn_index_store is not None:
        return
    raise MemoryOrgSessionAdmissionError(
        MemoryOrgSessionAdmissionReason.EPISODIC_RECALL_NOT_BOUND,
        "session vector index (episodic recall) is enabled but no implementation is bound",
    )
