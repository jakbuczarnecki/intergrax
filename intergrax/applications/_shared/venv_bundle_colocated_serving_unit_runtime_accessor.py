# © Artur Czarnecki. All rights reserved.

"""Composition-side accessor for VENV_BUNDLE co-located serving units (ADR §4.5B)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolver,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    build_production_runtime_agent_factory_resolver,
)

if TYPE_CHECKING:
    from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
        VenvBundleColocatedRuntimeDeploymentAdapter,
    )


class VenvBundleColocatedServingUnitRuntimeAccessor:
    """Maps attested colocated unit handles to revision-bound factory resolvers."""

    def __init__(self, adapter: VenvBundleColocatedRuntimeDeploymentAdapter) -> None:
        self._adapter = adapter
        self._resolver_by_unit: dict[str, RuntimeAgentFactoryResolver] = {}

    def runtime_agent_factory_resolver_for_unit(
        self,
        *,
        serving_unit_ref: str,
        revision: object,
        artifact_root: object,
    ) -> RuntimeAgentFactoryResolver:
        cached = self._resolver_by_unit.get(serving_unit_ref)
        if cached is not None:
            return cached
        if not isinstance(artifact_root, Path):
            raise TypeError("artifact_root must be a pathlib.Path for VENV_BUNDLE handoff")
        if not isinstance(revision, RuntimeRevision):
            raise TypeError("revision must be a RuntimeRevision for VENV_BUNDLE handoff")
        digest = revision.materialization_artifact_digest
        resolver = build_production_runtime_agent_factory_resolver(
            runtime_revision=revision,
            artifact_root=artifact_root,
            expected_artifact_digest=digest,
        )
        self._resolver_by_unit[serving_unit_ref] = resolver
        return resolver

    def admit_execution(self, serving_unit_ref: str) -> None:
        self._adapter.admit_unit_execution(serving_unit_ref)

    def release_execution(self, serving_unit_ref: str) -> None:
        self._adapter.release_unit_execution(serving_unit_ref)


__all__ = ["VenvBundleColocatedServingUnitRuntimeAccessor"]
