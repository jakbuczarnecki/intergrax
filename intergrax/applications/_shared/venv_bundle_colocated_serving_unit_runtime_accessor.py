# © Artur Czarnecki. All rights reserved.

"""Composition-side accessor for VENV_BUNDLE co-located serving units (ADR §4.5B)."""

from __future__ import annotations

from pathlib import Path

from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolver,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    build_production_runtime_agent_factory_resolver,
)


class VenvBundleColocatedServingUnitRuntimeAccessor:
    """Maps attested colocated unit handles to revision-bound factory resolvers."""

    def __init__(self, adapter: VenvBundleColocatedRuntimeDeploymentAdapter) -> None:
        self._adapter = adapter
        self._resolver_by_unit: dict[str, RuntimeAgentFactoryResolver] = {}
        self._binding_warmup_by_unit: dict[str, str] = {}

    def runtime_agent_factory_resolver_for_unit(
        self,
        *,
        serving_unit_ref: str,
        revision: RuntimeRevision,
        artifact_root: Path,
    ) -> RuntimeAgentFactoryResolver:
        binding = self._adapter.colocated_unit_runtime_binding(serving_unit_ref)
        if binding.runtime_revision_id != revision.runtime_revision_id:
            raise ValueError("revision identity does not match colocated unit binding")
        if binding.materialization_artifact_digest != revision.materialization_artifact_digest:
            raise ValueError("revision digest does not match colocated unit binding")
        if binding.artifact_root.resolve() != artifact_root.resolve():
            raise ValueError("artifact_root does not match colocated unit binding")

        cached = self._resolver_by_unit.get(serving_unit_ref)
        cached_warmup = self._binding_warmup_by_unit.get(serving_unit_ref)
        if (
            cached is not None
            and cached_warmup == binding.physical_runtime_warmup_evidence
        ):
            return cached

        digest = revision.materialization_artifact_digest
        if digest is None:
            raise ValueError("runtime revision requires materialization_artifact_digest")
        resolver = build_production_runtime_agent_factory_resolver(
            runtime_revision=revision,
            artifact_root=binding.artifact_root,
            expected_artifact_digest=digest,
        )
        self._resolver_by_unit[serving_unit_ref] = resolver
        self._binding_warmup_by_unit[serving_unit_ref] = (
            binding.physical_runtime_warmup_evidence
        )
        return resolver

    def admit_execution(self, serving_unit_ref: str) -> None:
        self._adapter.admit_unit_execution(serving_unit_ref)

    def release_execution(self, serving_unit_ref: str) -> None:
        self._adapter.release_unit_execution(serving_unit_ref)


__all__ = ["VenvBundleColocatedServingUnitRuntimeAccessor"]
