# © Artur Czarnecki. All rights reserved.

"""Canonical host effective profile materialization + admission for environment execution."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.profile_resolution import (
    materialize_effective_profile_revision,
    resolve_profile,
)
from intergrax.applications._shared.profile_resolution.activation_service import (
    EffectiveProfileActivationDependencies,
    EffectiveProfileActivationService,
)
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
    EffectiveProfileRevisionAdmission,
)
from intergrax.applications._shared.profile_resolution.wiring import (
    EffectiveProfilePersistenceWiring,
    resolve_effective_profile_persistence_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevisionScope,
    ProfileLayerInput,
    ProfileResolution,
)
from intergrax.applications.contracts.profile_resolution.activation import (
    ActivateEffectiveProfileRevisionRequest,
    ActiveEffectiveProfileRevisionBinding,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.applications._shared.profile_resolution.profile_resolution_child_context_inheritance_adapter import (
    build_profile_resolution_child_context_inheritance_adapter,
)
from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
)
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)


@dataclass(frozen=True, slots=True)
class HostEffectiveProfileExecutionWiring:
    """Resolved effective environment, persistence, and admission for one host."""

    effective_environment: ApplicationEnvironmentProfile
    profile_resolution: ProfileResolution
    persistence: EffectiveProfilePersistenceWiring
    revision_admission: EffectiveProfileRevisionAdmissionPort
    child_context_inheritance: ChildExecutionContextInheritancePort


def wire_host_effective_profile_execution(
    environment: ApplicationEnvironmentProfile,
    *,
    application_id: str,
    tenant_id: str,
    document_store: DocumentStore | None = None,
    kv_store: DistributedKVStore | None = None,
    profile_layers: tuple[ProfileLayerInput, ...] = (),
) -> HostEffectiveProfileExecutionWiring:
    """Resolve profile, persist revision truth, activate, and build canonical admission."""
    profile_resolution = resolve_profile(environment, layers=profile_layers)
    effective_environment = profile_resolution.effective_profile
    production_mode = effective_environment.execution_mode.value == "strict"
    persistence = resolve_effective_profile_persistence_wiring(
        production_mode=production_mode,
        kv_store=kv_store,
        document_store=document_store,
    )
    revision_scope = EffectiveProfileRevisionScope(
        application_id=application_id,
        tenant_id=tenant_id,
    )
    activation_baseline_binding: ActiveEffectiveProfileRevisionBinding | None = (
        persistence.active_store.get_active(revision_scope)
    )
    activation_baseline_revision_id = (
        activation_baseline_binding.revision_id
        if activation_baseline_binding is not None
        else None
    )
    effective_profile_revision = materialize_effective_profile_revision(
        profile_resolution,
        scope=revision_scope,
        store=persistence.revision_store,
    )
    activation_service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=persistence.revision_store,
            active_store=persistence.active_store,
        ),
    )
    activation_service.activate(
        ActivateEffectiveProfileRevisionRequest(
            scope=revision_scope,
            candidate_revision_id=effective_profile_revision.revision_id,
            expected_active_revision_id=activation_baseline_revision_id,
        ),
    )
    admission = EffectiveProfileRevisionAdmission(
        EffectiveProfileExecutionPinningDependencies(
            revision_store=persistence.revision_store,
            pinning_store=persistence.pinning_store,
            active_store=persistence.active_store,
            scope=revision_scope,
        ),
    )
    child_context_inheritance = build_profile_resolution_child_context_inheritance_adapter(
        tenant_id=tenant_id,
        pinning_store=persistence.pinning_store,
    )
    return HostEffectiveProfileExecutionWiring(
        effective_environment=effective_environment,
        profile_resolution=profile_resolution,
        persistence=persistence,
        revision_admission=admission,
        child_context_inheritance=child_context_inheritance,
    )


__all__ = ["HostEffectiveProfileExecutionWiring", "wire_host_effective_profile_execution"]
