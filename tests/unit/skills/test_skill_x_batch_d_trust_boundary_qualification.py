# © Artur Czarnecki. All rights reserved.

"""SKILL-X Batch D — BC-F02 trust boundary and adversarial replay qualification."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.skills.contracts.skill_runtime_binding_metadata import SkillRuntimeBindingMetadata
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.execution_binding import (
    BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1,
    BINDING_INTEGRITY_RUNTIME_CONTENT_V2,
    InMemorySkillExecutionPinningStore,
    bind_resolved_skill_pack,
    collect_registry_content_witness_digests,
    recover_skill_execution_binding,
    require_bound_skill_pack,
    resolve_bound_skill_pack,
    verify_skill_execution_binding_integrity,
)
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver

pytestmark = pytest.mark.unit


def _manifest(
    skill_id: str,
    version: str = "1.0.0",
    *,
    tool_ids: tuple[str, ...] = ("tool.x",),
) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        description=skill_id,
        version=version,
        tool_ids=tool_ids,
    )


def _binding_metadata(skill_id: str, version: str, digest: str) -> SkillRuntimeBindingMetadata:
    return SkillRuntimeBindingMetadata(
        catalog_source_id="catalog.test",
        logical_skill_id=skill_id,
        package_reference=f"pkg://{skill_id}",
        version_label=version,
        content_digest=digest,
    )


def test_witness_captured_from_catalog_not_post_hoc_runtime_tuple() -> None:
    registry = SkillRegistry()
    digest = "sha256:catalog-anchor"
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", digest),
    )
    pack = SkillResolver(registry).resolve_composition(["a.pack"]).pack
    witness = collect_registry_content_witness_digests(registry, pack)
    assert witness == (("a.pack", digest),)


def test_catalog_digest_substitution_same_version_rejected_on_canonical_resolve() -> None:
    pinning = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.old",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.register_or_replace(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.evil",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:substituted"),
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="catalog witness"):
            resolve_bound_skill_pack(
                tenant_id="tenant-a",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=pinning,
            )
    finally:
        reset_active_execution_identity(token)


def test_missing_catalog_binding_metadata_blocks_canonical_resolve() -> None:
    pinning = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.unregister("a.pack")
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="catalog binding metadata unavailable"):
            resolve_bound_skill_pack(
                tenant_id="tenant-a",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=pinning,
            )
    finally:
        reset_active_execution_identity(token)


def test_verified_historical_replay_when_catalog_matches_pin() -> None:
    pinning = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        pack = resolve_bound_skill_pack(
            tenant_id="tenant-a",
            skill_profile=SkillProfile(enabled=["a.pack"]),
            skill_registry=registry,
            pinning_store=pinning,
        )
        assert pack.snapshot_digest == composition.pack.snapshot_digest
    finally:
        reset_active_execution_identity(token)


def test_manifest_observation_v1_preserves_no_witness_claims() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=InMemorySkillExecutionPinningStore(),
        resolved_composition=composition,
    )
    assert binding.binding_integrity_version == BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1
    assert binding.registry_content_witness_digests == ()
    assert binding.runtime_content_digests == ()


def test_storage_only_recovery_without_registry_is_structural_not_authenticity() -> None:
    """BC-F02: recover_* without live catalog cannot assert independent replay authenticity."""
    pinning = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.register_or_replace(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.evil",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:substituted"),
    )
    historical = recover_skill_execution_binding(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning,
    )
    assert historical.binding_integrity_version == BINDING_INTEGRITY_RUNTIME_CONTENT_V2
    verify_skill_execution_binding_integrity(historical)
    require_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning,
    )
