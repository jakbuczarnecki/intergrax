# © Artur Czarnecki. All rights reserved.

"""Revision-bound registry reprojection with host-sanctioned AC3 factory (R2-F02)."""

from __future__ import annotations

from intergrax.applications._shared.host_overriding_runtime_agent_factory_resolver import (
    HostOverridingRuntimeAgentFactoryResolver,
    HostRuntimeAgentFactoryOverride,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolver,
    _factory_key,
)
from intergrax.applications._shared.wiring import (
    binding_from_roster_entry,
    factory_reference_for_roster_entry,
    _index_manifest_bindings,
)
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import ApplicationManifest
from governed_contractor_application.host.agent_builders import (
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)

_EXTERNAL_CONTRACTOR_CONTRACT_ID = "external_contractor_adapter"


def _external_contractor_factory_keys(
    bundle: RegistryProjectionInputBundle,
) -> frozenset[tuple[str, str | None, str | None]]:
    manifest_bindings = _index_manifest_bindings(bundle.manifest)
    keys: set[tuple[str, str | None, str | None]] = set()
    for entry in bundle.effective_roster.entries:
        if not entry.effective_enablement:
            continue
        binding = binding_from_roster_entry(entry, manifest_bindings)
        if binding.contract_id != _EXTERNAL_CONTRACTOR_CONTRACT_ID:
            continue
        factory_reference = factory_reference_for_roster_entry(entry, manifest_bindings)
        keys.add(_factory_key(entry.package_digest, factory_reference))
    if not keys:
        raise ValueError(
            "strict sanctioned external work requires revision-bound "
            f"{_EXTERNAL_CONTRACTOR_CONTRACT_ID!r} roster entry",
        )
    return frozenset(keys)


def build_strict_sanctioned_external_work_runtime_agent_factory_resolver(
    bundle: RegistryProjectionInputBundle,
    delegate: RuntimeAgentFactoryResolver,
    *,
    host_runtime: GovernedContractorHostRuntimeComposition,
    manifest: ApplicationManifest,
) -> HostOverridingRuntimeAgentFactoryResolver:
    if not host_runtime.require_sanctioned_external_work_mutations:
        raise ValueError(
            "strict sanctioned external work resolver requires "
            "require_sanctioned_external_work_mutations",
        )
    if host_runtime.sanctioned_external_work_operation_port is None:
        raise ValueError(
            "strict sanctioned external work resolver requires "
            "sanctioned_external_work_operation_port",
        )
    builders = build_governed_contractor_agent_builders(host_runtime)
    from external_contractor_adapter.external_contractor_adapter_agent import (
        ExternalContractorAdapterAgent,
    )

    host_factory: CanonicalAgentFactory = builders[ExternalContractorAdapterAgent]
    override = HostRuntimeAgentFactoryOverride(
        factory_keys=_external_contractor_factory_keys(bundle),
        host_factory=host_factory,
    )
    return HostOverridingRuntimeAgentFactoryResolver(delegate, override=override)


def reproject_registry_with_strict_sanctioned_external_work(
    bundle: RegistryProjectionInputBundle,
    delegate_resolver: RuntimeAgentFactoryResolver,
    *,
    host_runtime: GovernedContractorHostRuntimeComposition,
    manifest: ApplicationManifest,
) -> MaterializedRegistryProjection:
    """Materialize one authoritative registry using host-injected AC3 factory only."""
    decorated = build_strict_sanctioned_external_work_runtime_agent_factory_resolver(
        bundle,
        delegate_resolver,
        host_runtime=host_runtime,
        manifest=manifest,
    )
    reprojection_bundle = RegistryProjectionInputBundle(
        runtime_revision=bundle.runtime_revision,
        effective_roster=bundle.effective_roster,
        manifest=bundle.manifest,
        build_context=bundle.build_context,
        composition=bundle.composition,
        factory_resolver=decorated,
        builders=None,
        materialization_artifact_digest=bundle.materialization_artifact_digest,
        roster_contract_authority=bundle.roster_contract_authority,
    )
    return build_registry_projection(reprojection_bundle)


__all__ = [
    "build_strict_sanctioned_external_work_runtime_agent_factory_resolver",
    "reproject_registry_with_strict_sanctioned_external_work",
]
