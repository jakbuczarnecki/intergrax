# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped resolved skill pack pinning (P1.10 / SKILL-X Batch C)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId, peek_active_execution_id
from intergrax.contracts.validation import compute_sha256_content_digest
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
from intergrax.skills.snapshot_digest import compute_resolved_skill_pack_digest

SKILL_EXECUTION_BINDING_HANDLE = "skill_execution_binding"
SKILL_EXECUTION_PINNING_STORE_HANDLE = "skill_execution_pinning_store"

BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1 = (
    "skill_execution_binding_integrity.manifest_observation.v1"
)
BINDING_INTEGRITY_RUNTIME_CONTENT_V2 = (
    "skill_execution_binding_integrity.runtime_content.v2"
)


class SkillExecutionPinningConflictError(SkillResolutionError):
    """Conflicting bind for the same tenant/execution."""


class SkillExecutionPinningStorageError(SkillResolutionError):
    """Durable pinning storage read/write or codec failure."""


@dataclass(frozen=True, slots=True)
class SkillExecutionBinding:
    """Immutable execution-bound resolved skill composition."""

    tenant_id: str
    execution_id: ExecutionId
    configured_skill_ids: tuple[str, ...]
    resolved_pack: ResolvedSkillPack
    contribution_provenance: tuple[SkillContributionProvenance, ...]
    observed_manifests: tuple[SkillManifest, ...] = ()
    runtime_content_digests: tuple[tuple[str, str], ...] = ()
    binding_integrity_version: str = BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1
    content_bound_skill_ids: tuple[str, ...] = ()


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

    @property
    def is_durable(self) -> bool:
        return False

    def pin(self, binding: SkillExecutionBinding) -> None:
        verify_skill_execution_binding_integrity(binding)
        key = (binding.tenant_id, str(binding.execution_id))
        with self._lock:
            existing = self._bindings.get(key)
            if existing is not None and not bindings_equivalent_for_pinning(existing, binding):
                raise SkillExecutionPinningConflictError(
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


def bindings_equivalent_for_pinning(
    left: SkillExecutionBinding,
    right: SkillExecutionBinding,
) -> bool:
    """Idempotent pin equivalence — ignores object identity, not historical semantics."""
    return left == right


def manifest_content_fingerprint(manifest: SkillManifest) -> str:
    """Content identity for manifest observations when runtime digest is absent."""
    return compute_sha256_content_digest(manifest.model_dump_json().encode("utf-8"))


def collect_runtime_content_digests(
    skill_registry: SkillRegistry,
    pack: ResolvedSkillPack,
) -> tuple[tuple[str, str], ...]:
    digests: list[tuple[str, str]] = []
    for ref in pack.resolved_skills:
        metadata = skill_registry.binding_metadata(ref.skill_id)
        if metadata is None:
            continue
        digests.append((ref.skill_id, metadata.content_digest))
    return tuple(sorted(digests))


def _binding_integrity_profile_for_digests(
    runtime_content_digests: tuple[tuple[str, str], ...],
) -> tuple[str, tuple[str, ...]]:
    if not runtime_content_digests:
        return (BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1, ())
    bound_ids = tuple(sorted({skill_id for skill_id, _ in runtime_content_digests}))
    return (BINDING_INTEGRITY_RUNTIME_CONTENT_V2, bound_ids)


def _assert_binding_matches_storage_scope(
    binding: SkillExecutionBinding,
    *,
    tenant_id: str,
    execution_id: ExecutionId,
) -> None:
    normalized_tenant = tenant_id.strip()
    if binding.tenant_id != normalized_tenant:
        raise SkillResolutionError("skill execution binding tenant_id mismatch")
    if binding.execution_id != execution_id:
        raise SkillResolutionError("skill execution binding execution_id mismatch")


def verify_skill_execution_binding_integrity(binding: SkillExecutionBinding) -> None:
    """Fail closed on digest, provenance, manifest, or identity mismatch."""
    normalized_tenant = binding.tenant_id.strip()
    if not normalized_tenant:
        raise SkillResolutionError("tenant_id must be non-empty")
    if str(binding.execution_id).strip() == "":
        raise SkillResolutionError("execution_id must be non-empty")

    expected_digest = compute_resolved_skill_pack_digest(binding.resolved_pack.resolved_skills)
    if binding.resolved_pack.snapshot_digest != expected_digest:
        raise SkillResolutionError("resolved pack snapshot_digest mismatch")

    pack_skill_ids = {ref.skill_id for ref in binding.resolved_pack.resolved_skills}
    is_empty_pack = not pack_skill_ids

    if is_empty_pack:
        if binding.observed_manifests:
            raise SkillResolutionError("empty resolved pack cannot retain observed manifests")
        if binding.runtime_content_digests:
            raise SkillResolutionError("empty resolved pack cannot retain runtime content digests")
        if binding.contribution_provenance:
            raise SkillResolutionError("empty resolved pack cannot retain contribution provenance")
        if binding.content_bound_skill_ids:
            raise SkillResolutionError("empty resolved pack cannot retain content-bound skill ids")
        if binding.binding_integrity_version != BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1:
            raise SkillResolutionError(
                "empty resolved pack requires manifest-observation binding integrity",
            )
        return

    if not binding.observed_manifests:
        raise SkillResolutionError(
            "nonempty resolved pack requires observed manifests for recovery",
        )

    manifest_by_skill_id = {
        manifest.skill_id: manifest for manifest in binding.observed_manifests
    }
    if len(manifest_by_skill_id) != len(binding.observed_manifests):
        raise SkillResolutionError("duplicate observed manifest skill_id in binding")

    expected_provenance = build_skill_contribution_provenance(
        binding.resolved_pack,
        manifest_by_skill_id,
    )
    if binding.contribution_provenance != expected_provenance:
        raise SkillResolutionError("contribution provenance does not match resolved pack")

    for ref in binding.resolved_pack.resolved_skills:
        manifest = manifest_by_skill_id.get(ref.skill_id)
        if manifest is None:
            raise SkillResolutionError(
                f"missing observed manifest for resolved skill: {ref.qualified_id}",
            )
        if manifest.version != ref.version:
            raise SkillResolutionError(
                f"manifest version mismatch for {ref.qualified_id}",
            )

    integrity_version = binding.binding_integrity_version
    if integrity_version not in (
        BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1,
        BINDING_INTEGRITY_RUNTIME_CONTENT_V2,
    ):
        raise SkillResolutionError("unsupported skill execution binding integrity version")

    digest_skill_ids: list[str] = []
    for skill_id, digest in binding.runtime_content_digests:
        if not digest.strip():
            raise SkillResolutionError(f"empty runtime content digest for skill: {skill_id}")
        if skill_id in digest_skill_ids:
            raise SkillResolutionError(f"duplicate runtime content digest for skill: {skill_id}")
        digest_skill_ids.append(skill_id)
        if skill_id not in pack_skill_ids:
            raise SkillResolutionError(
                f"runtime content digest for unknown resolved skill: {skill_id}",
            )
        manifest = manifest_by_skill_id[skill_id]
        if digest == manifest_content_fingerprint(manifest):
            raise SkillResolutionError(
                f"runtime content digest cannot equal manifest serialization fingerprint: {skill_id}",
            )

    bound_from_digests = tuple(sorted(digest_skill_ids))
    if binding.content_bound_skill_ids != bound_from_digests:
        raise SkillResolutionError("content_bound_skill_ids mismatch runtime content digests")

    if integrity_version == BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1:
        if binding.runtime_content_digests or binding.content_bound_skill_ids:
            raise SkillResolutionError(
                "manifest-observation binding cannot claim runtime content digests",
            )
        return

    if not bound_from_digests:
        raise SkillResolutionError(
            "runtime-content binding integrity requires content-bound skill digests",
        )
    for skill_id in binding.content_bound_skill_ids:
        if skill_id not in pack_skill_ids:
            raise SkillResolutionError(
                f"content-bound skill not present in resolved pack: {skill_id}",
            )


def _build_skill_execution_binding(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    skill_profile: SkillProfile,
    composition: ResolvedSkillComposition,
    runtime_content_digests: tuple[tuple[str, str], ...] = (),
) -> SkillExecutionBinding:
    integrity_version, content_bound_skill_ids = _binding_integrity_profile_for_digests(
        runtime_content_digests,
    )
    binding = SkillExecutionBinding(
        tenant_id=tenant_id.strip(),
        execution_id=execution_id,
        configured_skill_ids=tuple(enabled_skill_ids_for_profile(skill_profile)),
        resolved_pack=composition.pack,
        contribution_provenance=build_skill_contribution_provenance(
            composition.pack,
            composition.manifest_by_skill_id(),
        ),
        observed_manifests=composition.observed_manifests,
        runtime_content_digests=runtime_content_digests,
        binding_integrity_version=integrity_version,
        content_bound_skill_ids=content_bound_skill_ids,
    )
    verify_skill_execution_binding_integrity(binding)
    return binding


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
    if str(execution_id).strip() == "":
        raise SkillResolutionError("execution_id must be non-empty")
    composition = resolved_composition or resolve_skill_composition_from_profile(
        skill_profile,
        skill_registry=skill_registry,
    )
    runtime_digests = collect_runtime_content_digests(skill_registry, composition.pack)
    binding = _build_skill_execution_binding(
        tenant_id=normalized_tenant,
        execution_id=execution_id,
        skill_profile=skill_profile,
        composition=composition,
        runtime_content_digests=runtime_digests,
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
    _assert_binding_matches_storage_scope(
        binding,
        tenant_id=tenant_id,
        execution_id=execution_id,
    )
    verify_skill_execution_binding_integrity(binding)
    return binding


def recover_skill_execution_binding(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    pinning_store: SkillExecutionPinningStore,
) -> SkillExecutionBinding:
    """Load historical binding only — never performs fresh skill resolution."""
    return require_bound_skill_pack(
        tenant_id=tenant_id,
        execution_id=execution_id,
        pinning_store=pinning_store,
    )


def rehydrate_resolved_composition(binding: SkillExecutionBinding) -> ResolvedSkillComposition:
    """Reconstruct coherent composition observation from immutable binding evidence."""
    verify_skill_execution_binding_integrity(binding)
    return ResolvedSkillComposition(
        pack=binding.resolved_pack,
        observed_manifests=binding.observed_manifests,
    )


def recover_bound_skill_pack(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    pinning_store: SkillExecutionPinningStore,
) -> ResolvedSkillPack:
    return recover_skill_execution_binding(
        tenant_id=tenant_id,
        execution_id=execution_id,
        pinning_store=pinning_store,
    ).resolved_pack


def resolve_bound_skill_pack(
    *,
    tenant_id: str,
    skill_profile: SkillProfile,
    skill_registry: SkillRegistry,
    pinning_store: SkillExecutionPinningStore | None,
    execution_id: ExecutionId | None = None,
    explicit_binding: SkillExecutionBinding | None = None,
    skill_execution_recovery_required: bool = False,
) -> ResolvedSkillPack:
    """
    Canonical consumer entry — resolve once per execution, then reuse bound pack.

    When ``explicit_binding`` is provided it wins over store lookup.
    When ``skill_execution_recovery_required`` is true, missing durable binding
    fails closed instead of performing a fresh bind.
    """
    active_execution_id = execution_id or peek_active_execution_id()
    if explicit_binding is not None:
        if active_execution_id is not None:
            _assert_binding_matches_storage_scope(
                explicit_binding,
                tenant_id=tenant_id,
                execution_id=active_execution_id,
            )
        verify_skill_execution_binding_integrity(explicit_binding)
        return explicit_binding.resolved_pack

    if pinning_store is not None and active_execution_id is not None and tenant_id.strip():
        if skill_execution_recovery_required:
            return recover_bound_skill_pack(
                tenant_id=tenant_id,
                execution_id=active_execution_id,
                pinning_store=pinning_store,
            )
        existing = pinning_store.get(tenant_id=tenant_id, execution_id=active_execution_id)
        if existing is not None:
            _assert_binding_matches_storage_scope(
                existing,
                tenant_id=tenant_id,
                execution_id=active_execution_id,
            )
            verify_skill_execution_binding_integrity(existing)
            return existing.resolved_pack
        binding = bind_resolved_skill_pack(
            tenant_id=tenant_id,
            execution_id=active_execution_id,
            skill_profile=skill_profile,
            skill_registry=skill_registry,
            pinning_store=pinning_store,
        )
        return binding.resolved_pack

    if skill_execution_recovery_required:
        raise SkillResolutionError(
            "skill execution recovery required but pinning store or execution identity missing",
        )

    return resolve_skill_pack_from_profile(skill_profile, skill_registry=skill_registry)


def binding_from_composition(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    skill_profile: SkillProfile,
    composition: ResolvedSkillComposition,
    runtime_content_digests: tuple[tuple[str, str], ...] = (),
) -> SkillExecutionBinding:
    """Construct binding from one coherent resolution observation."""
    return _build_skill_execution_binding(
        tenant_id=tenant_id,
        execution_id=execution_id,
        skill_profile=skill_profile,
        composition=composition,
        runtime_content_digests=runtime_content_digests,
    )
