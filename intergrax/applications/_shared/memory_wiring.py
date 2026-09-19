# © Artur Czarnecki. All rights reserved.

"""Tier-3 memory platform wiring (Phase MEM-1.3, MEM-2.2, MEM-PERS.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Optional

from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_MEMORY_STORES
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications._shared.memory_vector_wiring import (
    build_session_turn_index_store,
    build_user_profile_manager,
)
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import (
    MongoDBIntegrationBundle,
    create_mongodb_integration,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.runtime.nexus.session.document_store_session_storage import DocumentStoreSessionStorage
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.memory.resolver import (
    MemoryStoreMaterializationContext,
    MemoryStorePluginResolutionError,
    materialize_session_storage,
    materialize_user_profile_store,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager
from intergrax.runtime.organization.organization_profile_store import OrganizationProfileStore
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.memory.contracts.memory_control import MemoryControlPlane
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.applications._shared.entity_graph_wiring import (
    resolve_entity_temporal_memory_capability,
)
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryIndexer,
    EntityTemporalMemoryCapability,
)
from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.applications._shared.entity_graph_wiring import (
    resolve_entity_temporal_memory_store,
)
from intergrax.applications._shared.memory_observability_wiring import (
    resolve_memory_diagnostic_emitter,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
from intergrax.applications._shared.memory_provider_admission import (
    validate_memory_platform_wiring_admission,
    validate_session_turn_index_store_admission,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
    BUILTIN_IN_MEMORY_USER_PROFILE_ID,
    BUILTIN_SQLITE_USER_PROFILE_ID,
    MemoryProviderIdentity,
    builtin_user_profile_store_identity,
    plugin_user_profile_store_identity,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidenceRegistry,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidenceRegistry,
)
from intergrax.applications._shared.specialized_memory_wiring import (
    SpecializedMemoryCapabilities,
    resolve_specialized_memory_capabilities,
)
from intergrax.memory.contracts.long_horizon_memory import CanonicalMemorySourceAuthority
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
)


@dataclass(frozen=True)
class MemoryPlatformWiring:
    """Resolved session + profile stores for a Tier-3 host."""

    session_storage: SessionStorage
    user_profile_store: UserProfileStore
    organization_profile_store: OrganizationProfileStore | None
    user_profile_store_identity: MemoryProviderIdentity | None = None
    user_profile_manager: UserProfileManager | None = None
    sqlite_bundle: SQLiteIntegrationBundle | None = None
    mongodb_bundle: MongoDBIntegrationBundle | None = None
    entity_temporal_memory_capability: EntityTemporalMemoryCapability | None = None
    entity_memory_indexer: EntityMemoryIndexer | None = None
    specialized_memory: SpecializedMemoryCapabilities = SpecializedMemoryCapabilities()
    memory_store_plugin_load_report: DomainPluginLoadReport = DomainPluginLoadReport.empty(
        EP_MEMORY_STORES
    )


def memory_plugin_bootstrap_errors(report: DomainPluginLoadReport) -> tuple[str, ...]:
    errors: list[str] = []
    for item in report.failed:
        errors.append(f"memory plugin load failed: {item.spec.name}: {item.error}")
    for item in report.rejected:
        if item.fail_closed:
            errors.append(
                "memory plugin admission rejected: "
                f"{item.spec.name}: {item.reason_code.value}",
            )
    if not errors:
        errors.append("memory plugin bootstrap admission is not acceptable")
    return tuple(errors)


def assert_strict_memory_bootstrap_acceptable(
    env: ApplicationEnvironmentProfile,
    memory_wiring: MemoryPlatformWiring,
) -> None:
    if env.execution_mode is not ExecutionMode.STRICT:
        return
    report = memory_wiring.memory_store_plugin_load_report
    if report.critical_bootstrap_acceptable:
        return
    raise MemoryStorePluginResolutionError(
        "; ".join(memory_plugin_bootstrap_errors(report)),
    )


def _sqlite_enabled(profile: IntegrationProfile) -> bool:
    binding = profile.relational_store
    if binding is None:
        return False
    return binding.resolved_slug() == "sqlite"


def _mongodb_enabled(profile: IntegrationProfile) -> bool:
    binding = profile.document_store
    if binding is None:
        return False
    return binding.resolved_slug() == "mongodb"


def _sqlite_integration_overrides(profile: IntegrationProfile) -> dict[str, object]:
    options = profile.options or {}
    raw = options.get("sqlite")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _mongodb_integration_overrides(profile: IntegrationProfile) -> dict[str, object]:
    options = profile.options or {}
    raw = options.get("mongodb")
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _document_store_backing_provider_id(profile: IntegrationProfile) -> str | None:
    binding = profile.document_store
    if binding is None:
        return None
    return binding.resolved_slug()


def _resolve_baseline_memory_platform_wiring(
    env: ApplicationEnvironmentProfile,
    profile: IntegrationProfile,
    *,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority | None = None,
    long_horizon_source_authority: CanonicalMemorySourceAuthority | None = None,
) -> MemoryPlatformWiring:
    """Resolve integration-backed memory stores without external plugin overlay."""
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
        memory_diagnostic_emitter=emitter,
    )
    entity_store = resolve_entity_temporal_memory_store(env)
    entity_temporal_memory_capability = resolve_entity_temporal_memory_capability(
        env,
        security_governance=governance,
        memory_diagnostic_emitter=emitter,
        store=entity_store,
    )
    entity_memory_indexer = None
    if entity_store is not None:
        from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer

        entity_memory_indexer = DefaultEntityMemoryIndexer(
            entity_store,
            security_governance=governance,
            diagnostic_emitter=emitter,
        )
    specialized_memory = resolve_specialized_memory_capabilities(
        env,
        security_governance=governance,
        memory_observability_sink=memory_observability_sink,
        memory_diagnostic_emitter=emitter,
        governance_source_authority=governance_source_authority,
        long_horizon_source_authority=long_horizon_source_authority,
    )
    if _sqlite_enabled(profile):
        bundle = create_sqlite_integration(**_sqlite_integration_overrides(profile))
        return MemoryPlatformWiring(
            session_storage=bundle.session_storage,
            user_profile_store=bundle.user_profile_store,
            user_profile_store_identity=builtin_user_profile_store_identity(
                BUILTIN_SQLITE_USER_PROFILE_ID,
            ),
            organization_profile_store=bundle.organization_profile_store,
            sqlite_bundle=bundle,
            mongodb_bundle=None,
            entity_temporal_memory_capability=entity_temporal_memory_capability,
            entity_memory_indexer=entity_memory_indexer,
            specialized_memory=specialized_memory,
        )

    if _mongodb_enabled(profile):
        mongo_bundle = create_mongodb_integration(**_mongodb_integration_overrides(profile))
        document_store: DocumentStore = mongo_bundle.document_store.as_document_store()
        document_store_backing = _document_store_backing_provider_id(profile)
        org_store = None
        if env.memory_profile.enable_org_memory:
            from intergrax.runtime.organization.stores.in_memory_organization_profile_store import (
                InMemoryOrganizationProfileStore,
            )

            org_store = InMemoryOrganizationProfileStore()
        return MemoryPlatformWiring(
            session_storage=DocumentStoreSessionStorage(document_store),
            user_profile_store=DocumentStoreUserProfileStore(document_store),
            user_profile_store_identity=builtin_user_profile_store_identity(
                BUILTIN_DOCUMENT_STORE_USER_PROFILE_ID,
                backing_provider_id=document_store_backing,
            ),
            organization_profile_store=org_store,
            sqlite_bundle=None,
            mongodb_bundle=mongo_bundle,
            entity_temporal_memory_capability=entity_temporal_memory_capability,
            entity_memory_indexer=entity_memory_indexer,
            specialized_memory=specialized_memory,
        )

    return MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        user_profile_store_identity=builtin_user_profile_store_identity(
            BUILTIN_IN_MEMORY_USER_PROFILE_ID,
        ),
        organization_profile_store=None,
        sqlite_bundle=None,
        mongodb_bundle=None,
        entity_temporal_memory_capability=entity_temporal_memory_capability,
        entity_memory_indexer=entity_memory_indexer,
        specialized_memory=specialized_memory,
    )


def _apply_external_memory_store_overlay(
    wiring: MemoryPlatformWiring,
    env: ApplicationEnvironmentProfile,
    profile: IntegrationProfile,
    *,
    tenant_id: str | None,
    discover_entry_points: bool,
    explicit_memory_plugins: Sequence[type] = (),
) -> MemoryPlatformWiring:
    memory_profile = env.memory_profile
    user_plugin_id = memory_profile.user_profile_store_plugin_id
    session_plugin_id = memory_profile.session_storage_plugin_id
    if user_plugin_id is None and session_plugin_id is None:
        return wiring

    if not discover_entry_points and not explicit_memory_plugins:
        raise MemoryStorePluginResolutionError(
            "External memory store plugin selection requires entry-point discovery "
            "or explicit_memory_plugins candidates"
        )

    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=discover_entry_points,
        explicit_plugins=explicit_memory_plugins,
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)

    materialization_ctx = MemoryStoreMaterializationContext(
        tenant_id=tenant_id,
        integration_profile=profile,
    )
    updated = wiring
    if user_plugin_id is not None:
        user_profile_store = materialize_user_profile_store(
            user_plugin_id,
            materialization_ctx,
            catalog=catalog,
        )
        updated = replace(
            updated,
            user_profile_store=user_profile_store,
            user_profile_store_identity=plugin_user_profile_store_identity(user_plugin_id),
        )
    if session_plugin_id is not None:
        session_storage = materialize_session_storage(
            session_plugin_id,
            materialization_ctx,
            catalog=catalog,
        )
        updated = replace(updated, session_storage=session_storage)
    return replace(updated, memory_store_plugin_load_report=catalog.load_report)


def resolve_memory_platform_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
    tenant_id: str | None = None,
    discover_entry_points: bool | None = None,
    explicit_memory_plugins: Sequence[type] = (),
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    qualification_evidence_registry: MemoryProviderQualificationEvidenceRegistry | None = None,
    durability_evidence_registry: MemoryProviderDurabilityEvidenceRegistry | None = None,
    governance_source_authority: CanonicalMemoryGovernanceSourceAuthority | None = None,
    long_horizon_source_authority: CanonicalMemorySourceAuthority | None = None,
) -> MemoryPlatformWiring:
    """
    Resolve durable memory backends from the integration profile.

    Priority:
    1. SQLite relational_store — session + user LTM + org profile (lab default).
    2. MongoDB document_store — user LTM artifacts when SQLite is not enabled (MEM-PERS.2).
    3. In-memory fallbacks for session and user LTM.
    4. Explicit external Memory store plugin ids overlay their owned slots only.
    """
    profile = integration_profile or env.integration_profile
    wiring = _resolve_baseline_memory_platform_wiring(
        env,
        profile,
        security_governance=security_governance,
        memory_observability_sink=memory_observability_sink,
        memory_diagnostic_emitter=memory_diagnostic_emitter,
        governance_source_authority=governance_source_authority,
        long_horizon_source_authority=long_horizon_source_authority,
    )
    discover = discover_plugins_enabled() if discover_entry_points is None else discover_entry_points
    wiring = _apply_external_memory_store_overlay(
        wiring,
        env,
        profile,
        tenant_id=tenant_id,
        discover_entry_points=discover,
        explicit_memory_plugins=explicit_memory_plugins,
    )
    validate_memory_platform_wiring_admission(
        env,
        wiring.user_profile_store,
        user_profile_store_identity=wiring.user_profile_store_identity,
        qualification_evidence_registry=qualification_evidence_registry,
        durability_evidence_registry=durability_evidence_registry,
    )
    return wiring


def build_session_manager_from_environment(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
    integration_profile: IntegrationProfile | None = None,
    memory_wiring: MemoryPlatformWiring | None = None,
    rag_stack: RagStack | None = None,
    memory_control_plane: MemoryControlPlane | None = None,
    qualification_evidence_registry: MemoryProviderQualificationEvidenceRegistry | None = None,
    durability_evidence_registry: MemoryProviderDurabilityEvidenceRegistry | None = None,
    session_turn_index_store: SessionTurnIndexStore | None = None,
    session_turn_index_store_identity: MemoryProviderIdentity | None = None,
) -> SessionManager:
    """Construct ``SessionManager`` with profile managers driven by ``MemoryProfile``."""
    wiring = memory_wiring or resolve_memory_platform_wiring(
        env,
        integration_profile=integration_profile,
        tenant_id=tenant_id,
        qualification_evidence_registry=qualification_evidence_registry,
        durability_evidence_registry=durability_evidence_registry,
    )
    validate_memory_platform_wiring_admission(
        env,
        wiring.user_profile_store,
        user_profile_store_identity=wiring.user_profile_store_identity,
        qualification_evidence_registry=qualification_evidence_registry,
        durability_evidence_registry=durability_evidence_registry,
    )
    memory_profile = env.memory_profile

    user_manager = wiring.user_profile_manager
    if user_manager is None and (
        memory_profile.enable_user_memory or memory_profile.enable_long_term_memory
    ):
        user_manager = build_user_profile_manager(
            wiring.user_profile_store,
            env,
            tenant_id=tenant_id,
            rag_stack=rag_stack,
            entity_memory_indexer=wiring.entity_memory_indexer,
            entity_temporal_memory_capability=wiring.entity_temporal_memory_capability,
        )

    org_manager: Optional[OrganizationProfileManager] = None
    if memory_profile.enable_org_memory and wiring.organization_profile_store is not None:
        from intergrax.memory.org_memory_scope import ORG_MEMORY_SCOPES

        _ = ORG_MEMORY_SCOPES  # org memory 2.5 scope catalog (AUDIT-IDEAL-15.1)
        org_manager = OrganizationProfileManager(wiring.organization_profile_store)

    resolved_integration = integration_profile or env.integration_profile
    if session_turn_index_store is not None:
        validate_session_turn_index_store_admission(
            env,
            provider_identity=session_turn_index_store_identity,
            qualification_evidence_registry=qualification_evidence_registry,
        )
        session_turn_index = session_turn_index_store
    else:
        session_turn_index = build_session_turn_index_store(
            env,
            tenant_id=tenant_id,
            rag_stack=rag_stack,
            integration_profile=resolved_integration,
            qualification_evidence_registry=qualification_evidence_registry,
        )

    resolved_memory_control_plane = memory_control_plane
    if resolved_memory_control_plane is None and user_manager is not None:
        resolved_memory_control_plane = build_default_memory_control_plane(
            user_profile_manager=user_manager,
        )

    return SessionManager(
        wiring.session_storage,
        user_profile_manager=user_manager,
        organization_profile_manager=org_manager,
        session_turn_index_store=session_turn_index,
        session_turn_index_enabled=memory_profile.enable_session_vector_index,
        session_index_top_k=memory_profile.session_index_top_k,
        session_index_score_threshold=memory_profile.session_index_score_threshold,
        include_cross_session_episodic=memory_profile.include_cross_session_episodic,
        memory_consolidation_mode=memory_profile.consolidation_mode,
        memory_control_plane=resolved_memory_control_plane,
    )
