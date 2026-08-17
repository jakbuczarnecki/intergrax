# © Artur Czarnecki. All rights reserved.

"""Build canonical ``RegistryProjectionInputBundle`` for reference production lifecycle.

Explicit deploy input — not startup manifest projection. Uses process-local
``InMemoryRuntimeAgentFactoryResolver`` with caller-supplied factories; OCI/VENV
artifact factory loading remains deferred (``PRODUCTION_RUNTIME_FACTORY_ADAPTER_DEFERRED``).
"""

from __future__ import annotations

from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    InMemoryRuntimeAgentFactoryResolver,
    RuntimeAgentFactoryResolutionError,
)
from intergrax.applications._shared.wiring import (
    BuilderMap,
    factory_reference_for_roster_entry,
    _index_manifest_bindings,
    _resolve_manifest_binding_for_entry,
)
from intergrax.applications.contracts.agent_ref import qualname_for_callable
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

_REFERENCE_DIGEST = "sha256:" + ("a" * 64)
_REFERENCE_ROSTER_DIGEST = "sha256:" + ("e" * 64)
_REFERENCE_LOCK_ID = "reference-lock-1"
_REFERENCE_LOCK_DIGEST = "sha256:" + ("b" * 64)
_REFERENCE_GRAPH_DIGEST = "sha256:" + ("c" * 64)
_REFERENCE_ARTIFACT = "sha256:" + ("d" * 64)
_REFERENCE_RELEASE = "reference-release-1"
_REFERENCE_ARTIFACT_LOCATOR = "reference://process-local/venv-bundle"


def reference_artifact_locator_for_revision(runtime_revision_id: str) -> str:
    """Deterministic process-local artifact locator for reference lifecycle activation."""
    return f"{_REFERENCE_ARTIFACT_LOCATOR}/{runtime_revision_id}"


def _binding_stem(binding: AgentBinding) -> str:
    if binding.contract_id is not None:
        return binding.contract_id
    import_path = binding.import_path or ""
    stem = import_path.rsplit(".", 1)[-1]
    if stem.endswith("Agent") and len(stem) > 5:
        return stem[:-5].lower()
    return stem.lower()


def _roster_entry_from_binding(
    binding: AgentBinding,
    *,
    package_digest: str = _REFERENCE_DIGEST,
) -> EffectiveRosterEntry:
    contract_id = binding.contract_id
    import_path = binding.import_path
    if contract_id is not None:
        logical_id = contract_id
    elif import_path is not None:
        logical_id = import_path.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    else:
        logical_id = "agent"
    manifest_ref = f"manifest:agents/{logical_id}"
    return EffectiveRosterEntry(
        logical_agent_id=logical_id,
        installation_slot_id=f"slot-{logical_id}",
        package_digest=package_digest,
        distribution_package_id=f"pkg-{logical_id}",
        effective_enablement=binding.enabled,
        effective_default_agent=binding.default,
        manifest_origin_ref=manifest_ref,
    )


def _factory_for_binding(binding: AgentBinding, builders: BuilderMap | None) -> object:
    if binding.factory is not None:
        return binding.factory
    if builders is not None and binding.agent_type is not None:
        factory = builders.get(binding.agent_type)
        if factory is not None:
            return factory
    raise ValueError(f"missing factory for binding {binding.display_name()}")


def _resolver_for_bindings(
    manifest: ApplicationManifest,
    entries: tuple[EffectiveRosterEntry, ...],
    *,
    package_digest: str = _REFERENCE_DIGEST,
    builders: BuilderMap | None = None,
) -> InMemoryRuntimeAgentFactoryResolver:
    manifest_bindings = _index_manifest_bindings(manifest)
    resolver = InMemoryRuntimeAgentFactoryResolver()
    for entry in entries:
        if not entry.effective_enablement:
            continue
        binding = _resolve_manifest_binding_for_entry(entry, manifest_bindings)
        if binding is None:
            raise ValueError(f"missing manifest binding for roster entry {entry.logical_agent_id!r}")
        factory = _factory_for_binding(binding, builders)
        try:
            factory_ref = factory_reference_for_roster_entry(entry, manifest_bindings)
        except RuntimeAgentFactoryResolutionError:
            if binding.builder_key is not None:
                factory_ref = AgentBindingFactoryReference(builder_key=binding.builder_key)
            elif binding.factory_path is not None:
                factory_ref = AgentBindingFactoryReference(factory_path=binding.factory_path)
            elif builders is not None and binding.agent_type in builders:
                factory_ref = AgentBindingFactoryReference(
                    builder_key=binding.agent_type.__name__.lower().replace("agent", "")
                )
            else:
                raise
        resolver.register(
            package_digest=package_digest,
            factory_reference=factory_ref,
            factory=factory,
        )
    return resolver


def _manifest_with_builder_factories(
    manifest: ApplicationManifest,
    builders: BuilderMap | None,
) -> ApplicationManifest:
    if builders is None:
        return manifest
    agents: list[AgentBinding] = []
    for binding in manifest.agents:
        if binding.factory is None and binding.agent_type is not None:
            factory = builders.get(binding.agent_type)
            if factory is not None:
                agents.append(
                    binding.model_copy(
                        update={
                            "factory": factory,
                            "factory_path": qualname_for_callable(factory),
                        }
                    )
                )
                continue
        agents.append(binding)
    return manifest.model_copy(update={"agents": agents})


def _manifest_with_contract_ids(manifest: ApplicationManifest) -> ApplicationManifest:
    agents: list[AgentBinding] = []
    for binding in manifest.agents:
        if binding.contract_id is not None:
            agents.append(binding)
            continue
        class_contract = getattr(binding.agent_type, "contract_id", None)
        if isinstance(class_contract, str) and class_contract.strip():
            agents.append(binding.model_copy(update={"contract_id": class_contract.strip()}))
            continue
        import_path = binding.import_path
        if import_path is None:
            agents.append(binding)
            continue
        stem = import_path.rsplit(".", 1)[-1]
        if stem.endswith("Agent") and len(stem) > 5:
            stem = stem[:-5].lower()
        agents.append(binding.model_copy(update={"contract_id": stem}))
    return manifest.model_copy(update={"agents": agents})


def build_reference_registry_projection_input_bundle(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    enabled_bindings: tuple[AgentBinding, ...] | None = None,
    enabled_contract_stems: frozenset[str] | None = None,
    builders: BuilderMap | None = None,
    runtime_revision_id: str = "reference-runtime-revision",
    application_release_id: str = _REFERENCE_RELEASE,
    package_digest: str = _REFERENCE_DIGEST,
    materialization_artifact_digest: str = _REFERENCE_ARTIFACT,
    settings: object | None = None,
) -> RegistryProjectionInputBundle:
    """Build frozen AP-10 projection inputs for explicit reference deploy/activate."""
    manifest = _manifest_with_contract_ids(manifest)
    manifest = _manifest_with_builder_factories(manifest, builders)
    bindings = enabled_bindings or tuple(manifest.enabled_agents())
    if enabled_contract_stems is not None:
        bindings = tuple(
            binding
            for binding in bindings
            if (binding.contract_id or _binding_stem(binding)) in enabled_contract_stems
        )
    entries = tuple(
        _roster_entry_from_binding(binding, package_digest=package_digest) for binding in bindings
    )
    roster = EffectiveRoster(
        application_id=manifest.app_id,
        application_environment_id=environment.profile_id,
        manifest_release_id=application_release_id,
        entries=entries,
    ).with_revision_id()
    revision = RuntimeRevision(
        runtime_revision_id=runtime_revision_id,
        application_id=manifest.app_id,
        application_environment_id=environment.profile_id,
        application_release_id=application_release_id,
        platform_version="0.1.0",
        effective_roster_revision_id=roster.effective_roster_revision_id or _REFERENCE_ROSTER_DIGEST,
        installed_agent_package_digests=(package_digest,),
        materialized_runtime_lock_id=_REFERENCE_LOCK_ID,
        materialized_runtime_lock_digest=_REFERENCE_LOCK_DIGEST,
        runtime_graph_digest=_REFERENCE_GRAPH_DIGEST,
        materialization_artifact_digest=materialization_artifact_digest,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.CANDIDATE,
        activated_at=None,
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    if settings is not None:
        from intergrax.applications._shared.environment_wiring import wire_application_environment

        ctx = wire_application_environment(
            manifest,
            environment,
            settings=settings,
        ).build_context
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ctx,
        factory_resolver=_resolver_for_bindings(
            manifest,
            entries,
            package_digest=package_digest,
            builders=builders,
        ),
        builders=builders,
        materialization_artifact_digest=materialization_artifact_digest,
    )


def build_reference_activation_request(
    projection_input: RegistryProjectionInputBundle,
    *,
    expected_serving_pointer_revision: int = 0,
    expected_prior_traffic_revision_id: str | None = None,
) -> ActivateRuntimeRevisionRequest:
    """Canonical activation request aligned with one reference projection input bundle."""
    revision_id = projection_input.runtime_revision.runtime_revision_id
    artifact_digest = projection_input.materialization_artifact_digest
    if artifact_digest is None:
        raise ValueError("projection input requires materialization_artifact_digest")
    return ActivateRuntimeRevisionRequest(
        runtime_revision_id=revision_id,
        artifact_locator=reference_artifact_locator_for_revision(revision_id),
        expected_artifact_digest=artifact_digest,
        expected_serving_pointer_revision=expected_serving_pointer_revision,
        expected_prior_traffic_revision_id=expected_prior_traffic_revision_id,
    )


__all__ = [
    "build_reference_activation_request",
    "build_reference_registry_projection_input_bundle",
    "reference_artifact_locator_for_revision",
]
