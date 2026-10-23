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
from intergrax.skills.contracts.skill_runtime_binding_metadata import (
    SkillRuntimeBindingMetadata,
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
    observed_runtime_binding_metadata: tuple[SkillRuntimeBindingMetadata, ...] = ()
    binding_integrity_version: str = BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1
    content_bound_skill_ids: tuple[str, ...] = ()
    registry_content_witness_digests: tuple[tuple[str, str], ...] = ()
    # Catalog witness at pin time — not replay authenticity without live registry cross-check.


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


def collect_registry_content_witness_digests(
    skill_registry: SkillRegistry,
    pack: ResolvedSkillPack,
) -> tuple[tuple[str, str], ...]:
    """
    Catalog-anchored content-digest snapshot frozen at execution bind (BC-F02).

    Independent of the durable binding's ``runtime_content_digests`` field at
    verification time: consumers with a live ``SkillRegistry`` cross-check this
    witness against current catalog ``binding_metadata`` for the pinned version.
    """
    snapshots = collect_observed_runtime_binding_metadata(skill_registry, pack)
    if not snapshots:
        return ()
    if len(snapshots) != len(pack.resolved_skills):
        return ()
    return tuple(sorted((meta.logical_skill_id, meta.content_digest) for meta in snapshots))


def collect_runtime_content_digests(
    skill_registry: SkillRegistry,
    pack: ResolvedSkillPack,
) -> tuple[tuple[str, str], ...]:
    return collect_registry_content_witness_digests(skill_registry, pack)


def collect_observed_runtime_binding_metadata(
    skill_registry: SkillRegistry,
    pack: ResolvedSkillPack,
) -> tuple[SkillRuntimeBindingMetadata, ...]:
    """Trusted registry package identity snapshots for durable content-bound bindings."""
    snapshots: list[SkillRuntimeBindingMetadata] = []
    for ref in pack.resolved_skills:
        metadata = skill_registry.binding_metadata(ref.skill_id)
        if metadata is None:
            return ()
        if metadata.logical_skill_id != ref.skill_id or metadata.version_label != ref.version:
            raise SkillResolutionError(
                f"registry binding metadata identity mismatch for {ref.qualified_id}",
            )
        snapshots.append(metadata)
    return tuple(sorted(snapshots, key=lambda item: item.logical_skill_id))


def _binding_integrity_profile_for_digests(
    runtime_content_digests: tuple[tuple[str, str], ...],
    *,
    resolved_skill_ids: frozenset[str],
) -> tuple[str, tuple[str, ...]]:
    if not runtime_content_digests:
        return (BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1, ())
    bound_ids = frozenset(skill_id for skill_id, _ in runtime_content_digests)
    if bound_ids != resolved_skill_ids:
        raise SkillResolutionError(
            "runtime-content binding requires digest coverage for every resolved skill",
        )
    return (
        BINDING_INTEGRITY_RUNTIME_CONTENT_V2,
        tuple(sorted(bound_ids)),
    )


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


def _verify_registry_content_witness_against_catalog(
    binding: SkillExecutionBinding,
    skill_registry: SkillRegistry,
) -> None:
    """Cross-check pinned catalog witness against live registry when the release is still observable."""
    witness_by_skill_id = dict(binding.registry_content_witness_digests)
    for ref in binding.resolved_pack.resolved_skills:
        catalog_metadata = skill_registry.binding_metadata(ref.skill_id)
        if catalog_metadata is None:
            raise SkillResolutionError(
                f"catalog binding metadata unavailable for content witness: {ref.qualified_id}",
            )
        if catalog_metadata.version_label != ref.version:
            continue
        witnessed_digest = witness_by_skill_id.get(ref.skill_id)
        if witnessed_digest is None:
            raise SkillResolutionError(
                f"missing registry content witness for resolved skill: {ref.qualified_id}",
            )
        if catalog_metadata.content_digest != witnessed_digest:
            raise SkillResolutionError(
                "registry content digest mismatch with pinned catalog witness for "
                f"{ref.qualified_id}",
            )


def verify_skill_execution_binding_integrity(
    binding: SkillExecutionBinding,
    *,
    skill_registry: SkillRegistry | None = None,
) -> None:
    """Fail closed on digest, provenance, manifest, identity, or witness mismatch."""
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
        if binding.registry_content_witness_digests:
            raise SkillResolutionError(
                "empty resolved pack cannot retain registry content witness digests",
            )
        if binding.observed_runtime_binding_metadata:
            raise SkillResolutionError(
                "empty resolved pack cannot retain runtime binding metadata snapshots",
            )
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

    metadata_by_skill_id = {
        item.logical_skill_id: item for item in binding.observed_runtime_binding_metadata
    }
    if len(metadata_by_skill_id) != len(binding.observed_runtime_binding_metadata):
        raise SkillResolutionError("duplicate runtime binding metadata skill_id in binding")

    digest_by_skill_id = dict(binding.runtime_content_digests)

    if integrity_version == BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1:
        if binding.runtime_content_digests or binding.content_bound_skill_ids:
            raise SkillResolutionError(
                "manifest-observation binding cannot claim runtime content digests",
            )
        if binding.registry_content_witness_digests:
            raise SkillResolutionError(
                "manifest-observation binding cannot retain registry content witness digests",
            )
        if binding.observed_runtime_binding_metadata:
            raise SkillResolutionError(
                "manifest-observation binding cannot retain runtime binding metadata snapshots",
            )
        return

    if binding.registry_content_witness_digests != binding.runtime_content_digests:
        raise SkillResolutionError(
            "registry content witness digests must match runtime content digests",
        )
    if not binding.registry_content_witness_digests:
        raise SkillResolutionError(
            "runtime-content binding requires registry content witness digests",
        )

    if not bound_from_digests:
        raise SkillResolutionError(
            "runtime-content binding integrity requires content-bound skill digests",
        )
    if bound_from_digests != tuple(sorted(pack_skill_ids)):
        raise SkillResolutionError(
            "runtime-content binding requires complete content-bound coverage for all resolved skills",
        )
    for skill_id in binding.content_bound_skill_ids:
        if skill_id not in pack_skill_ids:
            raise SkillResolutionError(
                f"content-bound skill not present in resolved pack: {skill_id}",
            )
        ref = next(ref for ref in binding.resolved_pack.resolved_skills if ref.skill_id == skill_id)
        metadata = metadata_by_skill_id.get(skill_id)
        if metadata is None:
            raise SkillResolutionError(
                f"missing trusted runtime binding metadata snapshot for skill: {skill_id}",
            )
        if metadata.logical_skill_id != ref.skill_id or metadata.version_label != ref.version:
            raise SkillResolutionError(
                f"runtime binding metadata version mismatch for {ref.qualified_id}",
            )
        trusted_digest = metadata.content_digest
        recorded_digest = digest_by_skill_id.get(skill_id)
        if recorded_digest != trusted_digest:
            raise SkillResolutionError(
                f"runtime content digest mismatch with trusted package identity: {skill_id}",
            )
    if set(metadata_by_skill_id) != pack_skill_ids:
        raise SkillResolutionError(
            "runtime binding metadata snapshots must cover every resolved skill",
        )

    if skill_registry is not None:
        _verify_registry_content_witness_against_catalog(binding, skill_registry)


def _build_skill_execution_binding(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    skill_profile: SkillProfile,
    composition: ResolvedSkillComposition,
    runtime_content_digests: tuple[tuple[str, str], ...] = (),
    observed_runtime_binding_metadata: tuple[SkillRuntimeBindingMetadata, ...] = (),
    skill_registry: SkillRegistry | None = None,
) -> SkillExecutionBinding:
    pack_skill_ids = frozenset(ref.skill_id for ref in composition.pack.resolved_skills)
    integrity_version, content_bound_skill_ids = _binding_integrity_profile_for_digests(
        runtime_content_digests,
        resolved_skill_ids=pack_skill_ids,
    )
    if integrity_version == BINDING_INTEGRITY_RUNTIME_CONTENT_V2:
        if not observed_runtime_binding_metadata:
            raise SkillResolutionError(
                "runtime-content binding requires trusted runtime binding metadata snapshots",
            )
    elif observed_runtime_binding_metadata:
        raise SkillResolutionError(
            "manifest-observation binding cannot retain runtime binding metadata snapshots",
        )
    registry_content_witness_digests = ()
    if integrity_version == BINDING_INTEGRITY_RUNTIME_CONTENT_V2:
        if skill_registry is not None:
            registry_content_witness_digests = collect_registry_content_witness_digests(
                skill_registry,
                composition.pack,
            )
        else:
            registry_content_witness_digests = tuple(sorted(runtime_content_digests))
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
        observed_runtime_binding_metadata=observed_runtime_binding_metadata,
        binding_integrity_version=integrity_version,
        content_bound_skill_ids=content_bound_skill_ids,
        registry_content_witness_digests=registry_content_witness_digests,
    )
    verify_skill_execution_binding_integrity(binding, skill_registry=skill_registry)
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
    metadata_snapshots = collect_observed_runtime_binding_metadata(
        skill_registry,
        composition.pack,
    )
    runtime_digests = collect_runtime_content_digests(skill_registry, composition.pack)
    binding = _build_skill_execution_binding(
        tenant_id=normalized_tenant,
        execution_id=execution_id,
        skill_profile=skill_profile,
        composition=composition,
        runtime_content_digests=runtime_digests,
        observed_runtime_binding_metadata=metadata_snapshots,
        skill_registry=skill_registry,
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
        verify_skill_execution_binding_integrity(
            explicit_binding,
            skill_registry=skill_registry,
        )
        return explicit_binding.resolved_pack

    if pinning_store is not None and active_execution_id is not None and tenant_id.strip():
        if skill_execution_recovery_required:
            try:
                return recover_bound_skill_pack(
                    tenant_id=tenant_id,
                    execution_id=active_execution_id,
                    pinning_store=pinning_store,
                )
            except SkillExecutionPinningStorageError:
                raise
        try:
            existing = pinning_store.get(tenant_id=tenant_id, execution_id=active_execution_id)
        except SkillExecutionPinningStorageError:
            raise
        if existing is not None:
            _assert_binding_matches_storage_scope(
                existing,
                tenant_id=tenant_id,
                execution_id=active_execution_id,
            )
            verify_skill_execution_binding_integrity(
                existing,
                skill_registry=skill_registry,
            )
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
    observed_runtime_binding_metadata: tuple[SkillRuntimeBindingMetadata, ...] = (),
) -> SkillExecutionBinding:
    """Construct binding from one coherent resolution observation."""
    return _build_skill_execution_binding(
        tenant_id=tenant_id,
        execution_id=execution_id,
        skill_profile=skill_profile,
        composition=composition,
        runtime_content_digests=runtime_content_digests,
        observed_runtime_binding_metadata=observed_runtime_binding_metadata,
    )
