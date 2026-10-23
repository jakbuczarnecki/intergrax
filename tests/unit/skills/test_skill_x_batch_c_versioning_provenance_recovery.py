# © Artur Czarnecki. All rights reserved.

"""SKILL-X Batch C — versioning, provenance, durable pinning and recovery."""

from __future__ import annotations

import dataclasses
import json
import threading
from unittest.mock import MagicMock

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore, DocumentRecord
from intergrax.skills.contribution_provenance import SkillContributionKind, contributors_for
from intergrax.skills.contracts.skill_runtime_binding_metadata import SkillRuntimeBindingMetadata
from intergrax.skills.core.contracts import SkillManifest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.skills.execution_binding import (
    BINDING_INTEGRITY_RUNTIME_CONTENT_V2,
    InMemorySkillExecutionPinningStore,
    SkillExecutionBinding,
    SkillExecutionPinningConflictError,
    SkillExecutionPinningStorageError,
    bind_resolved_skill_pack,
    binding_from_composition,
    collect_runtime_content_digests,
    manifest_content_fingerprint,
    recover_bound_skill_pack,
    recover_skill_execution_binding,
    rehydrate_resolved_composition,
    require_bound_skill_pack,
    resolve_bound_skill_pack,
    verify_skill_execution_binding_integrity,
)
from intergrax.skills.execution_pinning_durability import (
    DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG,
    validate_skill_execution_pinning_durability_for_composition,
)
from intergrax.skills.execution_pinning_persistence import (
    DocumentStoreSkillExecutionPinningStore,
    KvSkillExecutionPinningStore,
    decode_skill_execution_binding,
    encode_skill_execution_binding,
    wire_skill_execution_pinning_store,
)
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver
from intergrax.skills.snapshot_digest import compute_resolved_skill_pack_digest
from intergrax.skills.skill_execution_pinning_wiring import resolve_skill_execution_pinning_wiring

pytestmark = pytest.mark.unit


def _manifest(
    skill_id: str,
    version: str = "1.0.0",
    *,
    requires_skills: tuple[str, ...] = (),
    tool_ids: tuple[str, ...] = ("tool.x",),
) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        description=skill_id,
        version=version,
        requires_skills=requires_skills,
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


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self.fail_get = False

    def get(self, tenant_id: str, key: str) -> bytes | None:
        if self.fail_get:
            raise RuntimeError("simulated kv read failure")
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        del ttl_seconds
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class InMemoryConditionalDocumentStore(ConditionalDocumentStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], DocumentRecord] = {}

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._records.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._records[(document.partition_key, document.row_key)] = document

    def put_if_absent(self, document: DocumentRecord) -> bool:
        key = (document.partition_key, document.row_key)
        if key in self._records:
            return False
        self._records[key] = document
        return True

    def delete(self, partition_key: str, row_key: str) -> None:
        self._records.pop((partition_key, row_key), None)

    def query_partition(self, partition_key: str, *, page_size: int | None = None):
        del page_size
        raise NotImplementedError


def test_snapshot_digest_deterministic() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    pack = SkillResolver(registry).resolve_composition(["a.pack"]).pack
    assert pack.snapshot_digest == compute_resolved_skill_pack_digest(pack.resolved_skills)


def test_same_version_label_different_content_distinct_fingerprints() -> None:
    left = _manifest("a.pack", "1.0.0", tool_ids=("tool.a",))
    right = _manifest("a.pack", "1.0.0", tool_ids=("tool.b",))
    assert manifest_content_fingerprint(left) != manifest_content_fingerprint(right)


def test_durable_pin_idempotent_and_recovery_without_resolution() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    profile = SkillProfile(enabled=["a.pack"])
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    registry.register_or_replace(_manifest("a.pack", "9.9.9", tool_ids=("tool.z",)))
    recovered = recover_skill_execution_binding(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store,
    )
    assert recovered.resolved_pack.snapshot_digest == binding.resolved_pack.snapshot_digest
    assert recovered.resolved_pack.resolved_skills[0].version == "1.0.0"
    rehydrated = rehydrate_resolved_composition(recovered)
    assert rehydrated.pack == binding.resolved_pack


def test_recovery_missing_binding_fails_closed() -> None:
    store = InMemorySkillExecutionPinningStore()
    with pytest.raises(SkillResolutionError, match="no bound skill pack"):
        recover_bound_skill_pack(
            tenant_id="tenant-a",
            execution_id=mint_execution_id(),
            pinning_store=store,
        )


def test_cross_tenant_isolation_for_same_execution_id() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    assert store.get(tenant_id="tenant-b", execution_id=execution_id) is None


def test_conflicting_concurrent_bind_rejected() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.one",)))
    first = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=first,
    )
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.two",)))
    second = SkillResolver(registry).resolve_composition(["a.pack"])
    with pytest.raises(SkillExecutionPinningConflictError, match="already pinned"):
        bind_resolved_skill_pack(
            tenant_id="tenant-a",
            execution_id=execution_id,
            skill_profile=SkillProfile(enabled=["a.pack"]),
            skill_registry=registry,
            pinning_store=store,
            resolved_composition=second,
        )


def test_lost_acknowledgement_idempotent_retry() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    profile = SkillProfile(enabled=["a.pack"])
    first = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    second = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    assert first == second


def test_storage_read_error_not_treated_as_missing() -> None:
    backing = InMemoryKVStore()
    backing.fail_get = True
    store = KvSkillExecutionPinningStore(backing)
    with pytest.raises(SkillExecutionPinningStorageError, match="read failed"):
        store.get(tenant_id="tenant-a", execution_id=mint_execution_id())


def test_corrupted_storage_record_rejected() -> None:
    with pytest.raises(SkillExecutionPinningStorageError, match="invalid skill execution binding"):
        decode_skill_execution_binding(b"not-json")


def test_unsupported_schema_version_rejected() -> None:
    payload = json.dumps({"schema_version": 99, "binding": {}}).encode("utf-8")
    with pytest.raises(SkillExecutionPinningStorageError, match="unsupported"):
        decode_skill_execution_binding(payload)


def test_partial_storage_failure_after_failed_cas() -> None:
    class EmptyAfterFalseCAS(InMemoryKVStore):
        def compare_and_set(self, tenant_id, key, expected, new_value, *, ttl_seconds=None):
            del ttl_seconds
            return False

    store = KvSkillExecutionPinningStore(EmptyAfterFalseCAS())
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=InMemorySkillExecutionPinningStore(),
        resolved_composition=composition,
    )
    with pytest.raises(SkillExecutionPinningStorageError, match="without readable record"):
        store.pin(binding)


def test_document_store_pinning_parity() -> None:
    doc = InMemoryConditionalDocumentStore()
    store = DocumentStoreSkillExecutionPinningStore(doc)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    recovered = require_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store,
    )
    assert recovered.resolved_pack.snapshot_digest == composition.pack.snapshot_digest


def test_runtime_content_digest_captured_at_bind() -> None:
    registry = SkillRegistry()
    digest = "sha256:abc123"
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", digest),
    )
    pack = SkillResolver(registry).resolve_composition(["a.pack"]).pack
    assert collect_runtime_content_digests(registry, pack) == (("a.pack", digest),)


def test_registry_replacement_during_active_execution_preserves_history() -> None:
    pinning = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.old",)))
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
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.new",)))
    historical = recover_skill_execution_binding(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning,
    )
    assert contributors_for(
        historical.contribution_provenance,
        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
        contribution_id="tool.old",
    ) == ("a.pack@1.0.0",)


def test_production_mode_requires_durable_pinning() -> None:
    with pytest.raises(SkillResolutionError, match=DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG):
        validate_skill_execution_pinning_durability_for_composition(
            production_mode=True,
            pinning_store=InMemorySkillExecutionPinningStore(),
        )


def test_production_wiring_uses_kv_when_available() -> None:
    backing = InMemoryKVStore()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        kv_store=backing,
    )
    assert wiring.pinning_store.is_durable is True


def test_lab_wiring_allows_in_memory() -> None:
    wiring = resolve_skill_execution_pinning_wiring(production_mode=False)
    assert wiring.pinning_store.is_durable is False


def test_concurrent_identical_bind_single_winner() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    profile = SkillProfile(enabled=["a.pack"])
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            bind_resolved_skill_pack(
                tenant_id="tenant-a",
                execution_id=execution_id,
                skill_profile=profile,
                skill_registry=registry,
                pinning_store=store,
                resolved_composition=composition,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not any(isinstance(exc, SkillExecutionPinningConflictError) for exc in errors)
    assert recover_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store,
    ).snapshot_digest == composition.pack.snapshot_digest


def test_forged_trace_payload_does_not_grant_execution_permission() -> None:
    """Runtime events remain non-authoritative — recovery requires pinning store evidence."""
    store = InMemorySkillExecutionPinningStore()
    forged_pack = MagicMock()
    forged_pack.skill_ids = ("forged.skill",)
    with pytest.raises(SkillResolutionError, match="no bound skill pack"):
        recover_bound_skill_pack(
            tenant_id="tenant-a",
            execution_id=mint_execution_id(),
            pinning_store=store,
        )


def test_wire_skill_execution_pinning_store_rejects_dual_backends() -> None:
    backing = InMemoryKVStore()
    doc = InMemoryConditionalDocumentStore()
    with pytest.raises(ValueError, match="not both"):
        wire_skill_execution_pinning_store(kv_store=backing, document_store=doc)


def test_empty_composition_bind_recover_and_replay() -> None:
    store = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    composition = SkillResolver(registry).resolve_composition(())
    execution_id = mint_execution_id()
    profile = SkillProfile()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=store,
        resolved_composition=composition,
    )
    assert binding.resolved_pack.resolved_skills == ()
    assert binding.observed_manifests == ()
    assert binding.runtime_content_digests == ()
    recovered = recover_skill_execution_binding(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store,
    )
    assert rehydrate_resolved_composition(recovered).pack == composition.pack


def test_nonempty_composition_rejects_missing_manifests() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
    )
    tampered = dataclasses.replace(binding, observed_manifests=())
    with pytest.raises(SkillResolutionError, match="requires observed manifests"):
        verify_skill_execution_binding_integrity(tampered)


def test_runtime_content_digest_duplicate_and_mismatch_rejected() -> None:
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:pkg-a"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
        runtime_content_digests=(("a.pack", "sha256:pkg-a"),),
    )
    tampered = dataclasses.replace(
        binding,
        runtime_content_digests=(
            ("a.pack", "sha256:pkg-a"),
            ("a.pack", "sha256:pkg-b"),
        ),
    )
    with pytest.raises(SkillResolutionError, match="duplicate runtime content digest"):
        verify_skill_execution_binding_integrity(tampered)


def test_manifest_fingerprint_cannot_pose_as_runtime_content_digest() -> None:
    registry = SkillRegistry()
    manifest = _manifest("a.pack")
    registry.register(manifest)
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
    )
    forged = dataclasses.replace(
        binding,
        runtime_content_digests=(("a.pack", manifest_content_fingerprint(manifest)),),
        binding_integrity_version=BINDING_INTEGRITY_RUNTIME_CONTENT_V2,
        content_bound_skill_ids=("a.pack",),
    )
    with pytest.raises(SkillResolutionError, match="manifest serialization fingerprint"):
        verify_skill_execution_binding_integrity(forged)


def test_tampered_binding_snapshot_digest_rejected() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    tampered_pack = binding.resolved_pack
    object.__setattr__(tampered_pack, "snapshot_digest", "forged")
    tampered = SkillExecutionBinding(
        tenant_id=binding.tenant_id,
        execution_id=binding.execution_id,
        configured_skill_ids=binding.configured_skill_ids,
        resolved_pack=tampered_pack,
        contribution_provenance=binding.contribution_provenance,
        observed_manifests=binding.observed_manifests,
        runtime_content_digests=binding.runtime_content_digests,
        binding_integrity_version=binding.binding_integrity_version,
        content_bound_skill_ids=binding.content_bound_skill_ids,
    )
    with pytest.raises(SkillResolutionError, match="snapshot_digest mismatch"):
        verify_skill_execution_binding_integrity(tampered)


def test_forged_explicit_binding_tenant_mismatch_rejected() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
    )
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="tenant_id mismatch"):
            resolve_bound_skill_pack(
                tenant_id="tenant-b",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=InMemorySkillExecutionPinningStore(),
                explicit_binding=binding,
            )
    finally:
        reset_active_execution_identity(token)


def test_resumed_execution_missing_pin_fails_closed() -> None:
    store = InMemorySkillExecutionPinningStore()
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="no bound skill pack"):
            resolve_bound_skill_pack(
                tenant_id="tenant-a",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=store,
                skill_execution_recovery_required=True,
            )
    finally:
        reset_active_execution_identity(token)


def test_runtime_content_binding_integrity_version_on_verified_metadata() -> None:
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:verified"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = binding_from_composition(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        composition=composition,
        runtime_content_digests=(("a.pack", "sha256:verified"),),
    )
    assert binding.binding_integrity_version == BINDING_INTEGRITY_RUNTIME_CONTENT_V2
    verify_skill_execution_binding_integrity(binding)


def test_encode_round_trip_preserves_binding() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    pinning = InMemorySkillExecutionPinningStore()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    round_trip = decode_skill_execution_binding(encode_skill_execution_binding(binding))
    assert round_trip == binding
