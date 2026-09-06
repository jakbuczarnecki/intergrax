# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped resolved skill pack pinning (P1.10)."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId, peek_active_execution_id
from intergrax.skills.contribution_provenance import (
    SkillContributionProvenance,
    build_skill_contribution_provenance,
)
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.factory import enabled_skill_ids_for_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import (
    ResolvedSkillComposition,
    ResolvedSkillPack,
    SkillResolutionError,
    SkillResolver,
)

SKILL_EXECUTION_BINDING_HANDLE = "skill_execution_binding"
SKILL_EXECUTION_PINNING_STORE_HANDLE = "skill_execution_pinning_store"


@dataclass(frozen=True, slots=True)
class SkillExecutionBinding:
    """Immutable execution-bound resolved skill composition."""

    tenant_id: str
    execution_id: ExecutionId
    configured_skill_ids: tuple[str, ...]
    resolved_pack: ResolvedSkillPack
    contribution_provenance: tuple[SkillContributionProvenance, ...]


@runtime_checkable
class SkillExecutionPinningStore(Protocol):
    """Execution-scoped skill pack binding — semantic data only."""

    def pin(self, binding: SkillExecutionBinding) -> None: ...

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> SkillExecutionBinding | None: ...


class InMemorySkillExecutionPinningStore:
    """Process-local execution skill pinning for hosts and tests."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], SkillExecutionBinding] = {}
        self._lock = threading.Lock()

    def pin(self, binding: SkillExecutionBinding) -> None:
        key = (binding.tenant_id, str(binding.execution_id))
        with self._lock:
            existing = self._bindings.get(key)
            if existing is not None and existing != binding:
                raise SkillResolutionError(
                    f"execution already pinned with different skill pack: {binding.execution_id}",
                )
            self._bindings[key] = binding

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> SkillExecutionBinding | None:
        with self._lock:
            return self._bindings.get((tenant_id, str(execution_id)))


def resolve_skill_composition_from_profile(
    skill_profile: SkillProfile,
    *,
    skill_registry: SkillRegistry,
) -> ResolvedSkillComposition:
    """Initial materialization for configured host skill selection."""
    skill_ids = enabled_skill_ids_for_profile(skill_profile)
    if not skill_ids:
        return SkillResolver(skill_registry).resolve_composition(())
    return SkillResolver(skill_registry).resolve_composition(skill_ids)


def resolve_skill_pack_from_profile(
    skill_profile: SkillProfile,
    *,
    skill_registry: SkillRegistry,
) -> ResolvedSkillPack:
    """Initial materialization for configured host skill selection."""
    return resolve_skill_composition_from_profile(
        skill_profile,
        skill_registry=skill_registry,
    ).pack


def bind_resolved_skill_pack(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    skill_profile: SkillProfile,
    skill_registry: SkillRegistry,
    pinning_store: SkillExecutionPinningStore,
    resolved_composition: ResolvedSkillComposition | None = None,
) -> SkillExecutionBinding:
    """Pin one immutable resolved skill pack for an execution."""
    normalized_tenant = tenant_id.strip()
    if not normalized_tenant:
        raise SkillResolutionError("tenant_id must be non-empty")
    configured_skill_ids = tuple(enabled_skill_ids_for_profile(skill_profile))
    composition = resolved_composition or resolve_skill_composition_from_profile(
        skill_profile,
        skill_registry=skill_registry,
    )
    binding = SkillExecutionBinding(
        tenant_id=normalized_tenant,
        execution_id=execution_id,
        configured_skill_ids=configured_skill_ids,
        resolved_pack=composition.pack,
        contribution_provenance=build_skill_contribution_provenance(
            composition.pack,
            composition.manifest_by_skill_id(),
        ),
    )
    pinning_store.pin(binding)
    return binding


def require_bound_skill_pack(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    pinning_store: SkillExecutionPinningStore,
) -> SkillExecutionBinding:
    binding = pinning_store.get(tenant_id=tenant_id, execution_id=execution_id)
    if binding is None:
        raise SkillResolutionError(
            f"no bound skill pack for execution {execution_id}",
        )
    return binding


def resolve_bound_skill_pack(
    *,
    tenant_id: str,
    skill_profile: SkillProfile,
    skill_registry: SkillRegistry,
    pinning_store: SkillExecutionPinningStore | None,
    execution_id: ExecutionId | None = None,
    explicit_binding: SkillExecutionBinding | None = None,
) -> ResolvedSkillPack:
    """
    Canonical consumer entry — resolve once per execution, then reuse bound pack.

    When ``explicit_binding`` is provided it wins over store lookup.
    """
    if explicit_binding is not None:
        return explicit_binding.resolved_pack

    active_execution_id = execution_id or peek_active_execution_id()
    if pinning_store is not None and active_execution_id is not None and tenant_id.strip():
        existing = pinning_store.get(tenant_id=tenant_id, execution_id=active_execution_id)
        if existing is not None:
            return existing.resolved_pack
        binding = bind_resolved_skill_pack(
            tenant_id=tenant_id,
            execution_id=active_execution_id,
            skill_profile=skill_profile,
            skill_registry=skill_registry,
            pinning_store=pinning_store,
        )
        return binding.resolved_pack

    return resolve_skill_pack_from_profile(skill_profile, skill_registry=skill_registry)


def binding_from_pack(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    skill_profile: SkillProfile,
    pack: ResolvedSkillPack,
    manifests: Mapping[str, SkillManifest],
) -> SkillExecutionBinding:
    """Construct binding evidence without registry reads (tests / explicit admission)."""
    return SkillExecutionBinding(
        tenant_id=tenant_id.strip(),
        execution_id=execution_id,
        configured_skill_ids=tuple(enabled_skill_ids_for_profile(skill_profile)),
        resolved_pack=pack,
        contribution_provenance=build_skill_contribution_provenance(pack, manifests),
    )


def binding_from_composition(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    skill_profile: SkillProfile,
    composition: ResolvedSkillComposition,
) -> SkillExecutionBinding:
    """Construct binding from one coherent resolution observation."""
    return binding_from_pack(
        tenant_id=tenant_id,
        execution_id=execution_id,
        skill_profile=skill_profile,
        pack=composition.pack,
        manifests=composition.manifest_by_skill_id(),
    )
