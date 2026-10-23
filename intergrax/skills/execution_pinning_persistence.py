# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable execution-scoped skill pack pinning (SKILL-X Batch C)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.skills.contribution_provenance import (
    SkillContributionKind,
    SkillContributionProvenance,
)
from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.version_binding import (
    ResolvedSkillRef,
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.contracts.skill_runtime_binding_metadata import (
    SkillRuntimeBindingMetadata,
)
from intergrax.skills.execution_binding import (
    BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1,
    SkillExecutionBinding,
    SkillExecutionPinningConflictError,
    SkillExecutionPinningStorageError,
    bindings_equivalent_for_pinning,
)
from intergrax.skills.resolver import ResolvedSkillPack, SkillResolutionError

_PINNING_KV_PREFIX = "skill_execution_pinning"
_PINNING_DOCUMENT_PARTITION_PREFIX = "intergrax.skill_execution_pinning.v1"
_PINNING_SCHEMA_VERSION = 1


def _pinning_kv_key(execution_id: str) -> str:
    return f"{_PINNING_KV_PREFIX}:{execution_id}"


def _pinning_document_partition(tenant_id: str) -> str:
    return f"{_PINNING_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"


def _ref_to_dict(ref: ResolvedSkillRef) -> dict[str, str]:
    return {
        "skill_id": ref.skill_id,
        "version": ref.version,
        "qualified_id": ref.qualified_id,
        "resolution_mode": ref.resolution_mode.value,
        "role": ref.role.value,
    }


def _ref_from_dict(raw: dict[str, Any]) -> ResolvedSkillRef:
    return ResolvedSkillRef(
        skill_id=str(raw["skill_id"]),
        version=str(raw["version"]),
        qualified_id=str(raw["qualified_id"]),
        resolution_mode=SkillVersionResolutionMode(str(raw["resolution_mode"])),
        role=ResolvedSkillRole(str(raw["role"])),
    )


def _pack_to_dict(pack: ResolvedSkillPack) -> dict[str, Any]:
    return {
        "resolved_skills": [_ref_to_dict(ref) for ref in pack.resolved_skills],
        "tool_ids": sorted(pack.tool_ids),
        "prompt_instruction_ids": sorted(pack.prompt_instruction_ids),
        "policy_fragment_ids": sorted(pack.policy_fragment_ids),
        "risk_tier": pack.risk_tier.value,
        "snapshot_digest": pack.snapshot_digest,
    }


def _pack_from_dict(raw: dict[str, Any]) -> ResolvedSkillPack:
    refs = tuple(_ref_from_dict(item) for item in raw["resolved_skills"])
    return ResolvedSkillPack(
        resolved_skills=refs,
        tool_ids=frozenset(str(item) for item in raw["tool_ids"]),
        prompt_instruction_ids=frozenset(str(item) for item in raw["prompt_instruction_ids"]),
        policy_fragment_ids=frozenset(str(item) for item in raw["policy_fragment_ids"]),
        risk_tier=SkillRiskTier(str(raw["risk_tier"])),
        snapshot_digest=str(raw["snapshot_digest"]),
    )


def _provenance_to_dict(entry: SkillContributionProvenance) -> dict[str, str]:
    return {
        "skill_id": entry.skill_id,
        "version": entry.version,
        "qualified_id": entry.qualified_id,
        "role": entry.role.value,
        "resolution_mode": entry.resolution_mode.value,
        "contribution_kind": entry.contribution_kind.value,
        "contribution_id": entry.contribution_id,
    }


def _runtime_binding_metadata_to_dict(entry: SkillRuntimeBindingMetadata) -> dict[str, str]:
    return {
        "schema_version": entry.schema_version,
        "catalog_source_id": entry.catalog_source_id,
        "logical_skill_id": entry.logical_skill_id,
        "package_reference": entry.package_reference,
        "version_label": entry.version_label,
        "content_digest": entry.content_digest,
    }


def _runtime_binding_metadata_from_dict(raw: dict[str, Any]) -> SkillRuntimeBindingMetadata:
    return SkillRuntimeBindingMetadata.model_validate(raw)


def _provenance_from_dict(raw: dict[str, Any]) -> SkillContributionProvenance:
    return SkillContributionProvenance(
        skill_id=str(raw["skill_id"]),
        version=str(raw["version"]),
        qualified_id=str(raw["qualified_id"]),
        role=ResolvedSkillRole(str(raw["role"])),
        resolution_mode=SkillVersionResolutionMode(str(raw["resolution_mode"])),
        contribution_kind=SkillContributionKind(str(raw["contribution_kind"])),
        contribution_id=str(raw["contribution_id"]),
    )


def _binding_storage_payload(binding: SkillExecutionBinding) -> dict[str, Any]:
    return {
        "tenant_id": binding.tenant_id,
        "execution_id": str(binding.execution_id),
        "configured_skill_ids": list(binding.configured_skill_ids),
        "resolved_pack": _pack_to_dict(binding.resolved_pack),
        "contribution_provenance": [
            _provenance_to_dict(item) for item in binding.contribution_provenance
        ],
        "observed_manifests": [
            json.loads(manifest.model_dump_json()) for manifest in binding.observed_manifests
        ],
        "runtime_content_digests": [
            {"skill_id": skill_id, "content_digest": digest}
            for skill_id, digest in binding.runtime_content_digests
        ],
        "observed_runtime_binding_metadata": [
            _runtime_binding_metadata_to_dict(item)
            for item in binding.observed_runtime_binding_metadata
        ],
        "binding_integrity_version": binding.binding_integrity_version,
        "content_bound_skill_ids": list(binding.content_bound_skill_ids),
    }


def encode_skill_execution_binding(binding: SkillExecutionBinding) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": _PINNING_SCHEMA_VERSION,
        "binding": _binding_storage_payload(binding),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_skill_execution_binding(raw: bytes) -> SkillExecutionBinding:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillExecutionPinningStorageError(
            "invalid skill execution binding encoding",
        ) from exc
    if not isinstance(payload, dict):
        raise SkillExecutionPinningStorageError("invalid skill execution binding payload")
    schema_version = payload.get("schema_version")
    if schema_version != _PINNING_SCHEMA_VERSION:
        raise SkillExecutionPinningStorageError(
            "unsupported skill execution binding schema version",
        )
    binding_raw = payload.get("binding")
    if not isinstance(binding_raw, dict):
        raise SkillExecutionPinningStorageError("invalid skill execution binding snapshot")
    try:
        tenant_id = str(binding_raw["tenant_id"]).strip()
        execution_id = validate_execution_id(str(binding_raw["execution_id"]))
        configured_skill_ids = tuple(str(item) for item in binding_raw["configured_skill_ids"])
        resolved_pack = _pack_from_dict(binding_raw["resolved_pack"])
        provenance = tuple(
            _provenance_from_dict(item) for item in binding_raw["contribution_provenance"]
        )
        manifests = tuple(
            SkillManifest.model_validate(item) for item in binding_raw["observed_manifests"]
        )
        digest_items = binding_raw.get("runtime_content_digests", [])
        runtime_content_digests = tuple(
            sorted(
                (str(item["skill_id"]), str(item["content_digest"]))
                for item in digest_items
            ),
        )
        binding_integrity_version = str(
            binding_raw.get(
                "binding_integrity_version",
                BINDING_INTEGRITY_MANIFEST_OBSERVATION_V1,
            ),
        )
        metadata_raw = binding_raw.get("observed_runtime_binding_metadata", [])
        observed_runtime_binding_metadata = tuple(
            _runtime_binding_metadata_from_dict(item) for item in metadata_raw
        )
        content_bound_raw = binding_raw.get("content_bound_skill_ids", [])
        content_bound_skill_ids = tuple(str(item) for item in content_bound_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise SkillExecutionPinningStorageError(
            "corrupt skill execution binding snapshot",
        ) from exc
    binding = SkillExecutionBinding(
        tenant_id=tenant_id,
        execution_id=execution_id,
        configured_skill_ids=configured_skill_ids,
        resolved_pack=resolved_pack,
        contribution_provenance=provenance,
        observed_manifests=manifests,
        runtime_content_digests=runtime_content_digests,
        observed_runtime_binding_metadata=observed_runtime_binding_metadata,
        binding_integrity_version=binding_integrity_version,
        content_bound_skill_ids=content_bound_skill_ids,
    )
    from intergrax.skills.execution_binding import verify_skill_execution_binding_integrity

    verify_skill_execution_binding_integrity(binding)
    return binding


def _assert_decoded_binding_scope(
    binding: SkillExecutionBinding,
    *,
    tenant_id: str,
    execution_id: ExecutionId,
) -> None:
    normalized_tenant = tenant_id.strip()
    if binding.tenant_id != normalized_tenant:
        raise SkillExecutionPinningStorageError(
            "skill execution binding tenant_id mismatch in storage record",
        )
    if binding.execution_id != execution_id:
        raise SkillExecutionPinningStorageError(
            "skill execution binding execution_id mismatch in storage record",
        )


class KvSkillExecutionPinningStore:
    """DistributedKVStore-backed immutable execution skill pinning."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @property
    def is_durable(self) -> bool:
        return True

    def _idempotent_pin_after_uncertain_write(
        self,
        binding: SkillExecutionBinding,
        *,
        key: str,
    ) -> bool:
        try:
            existing_raw = self._kv_store.get(tenant_id=binding.tenant_id, key=key)
        except Exception:
            return False
        if existing_raw is None:
            return False
        existing = decode_skill_execution_binding(existing_raw)
        if not bindings_equivalent_for_pinning(existing, binding):
            raise SkillExecutionPinningConflictError(
                f"execution already pinned with different skill pack: {binding.execution_id}",
            )
        return True

    def pin(self, binding: SkillExecutionBinding) -> None:
        encoded = encode_skill_execution_binding(binding)
        key = _pinning_kv_key(str(binding.execution_id))
        try:
            created = self._kv_store.compare_and_set(
                tenant_id=binding.tenant_id,
                key=key,
                expected=None,
                new_value=encoded,
            )
        except Exception as exc:
            if self._idempotent_pin_after_uncertain_write(binding, key=key):
                return
            raise SkillExecutionPinningStorageError(
                "skill execution pinning compare-and-set failed",
            ) from exc
        if created:
            return
        try:
            existing_raw = self._kv_store.get(tenant_id=binding.tenant_id, key=key)
        except Exception as exc:
            raise SkillExecutionPinningStorageError(
                "skill execution pinning storage read failed after compare-and-set",
            ) from exc
        if existing_raw is None:
            raise SkillExecutionPinningStorageError(
                "skill execution pinning compare-and-set failed without readable record",
            )
        existing = decode_skill_execution_binding(existing_raw)
        if not bindings_equivalent_for_pinning(existing, binding):
            raise SkillExecutionPinningConflictError(
                f"execution already pinned with different skill pack: {binding.execution_id}",
            )

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> SkillExecutionBinding | None:
        try:
            raw = self._kv_store.get(
                tenant_id=tenant_id,
                key=_pinning_kv_key(str(execution_id)),
            )
        except Exception as exc:
            raise SkillExecutionPinningStorageError(
                "skill execution pinning storage read failed",
            ) from exc
        if raw is None:
            return None
        binding = decode_skill_execution_binding(raw)
        _assert_decoded_binding_scope(
            binding,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )
        return binding


class DocumentStoreSkillExecutionPinningStore:
    """ConditionalDocumentStore-backed immutable execution skill pinning."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "skill execution pinning requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def _idempotent_pin_after_uncertain_document_write(
        self,
        binding: SkillExecutionBinding,
    ) -> bool:
        partition_key = _pinning_document_partition(binding.tenant_id)
        row_key = str(binding.execution_id)
        try:
            existing = self._document_store.get(partition_key, row_key)
        except Exception:
            return False
        if existing is None:
            return False
        stored = decode_skill_execution_binding(_binding_record_to_bytes(existing))
        if not bindings_equivalent_for_pinning(stored, binding):
            raise SkillExecutionPinningConflictError(
                f"execution already pinned with different skill pack: {binding.execution_id}",
            )
        return True

    def pin(self, binding: SkillExecutionBinding) -> None:
        partition_key = _pinning_document_partition(binding.tenant_id)
        row_key = str(binding.execution_id)
        document = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "binding": encode_skill_execution_binding(binding).decode("utf-8"),
            },
        )
        try:
            created = self._document_store.put_if_absent(document)
        except Exception as exc:
            if self._idempotent_pin_after_uncertain_document_write(binding):
                return
            raise SkillExecutionPinningStorageError(
                "skill execution pinning document create failed",
            ) from exc
        if created:
            return
        try:
            existing = self._document_store.get(partition_key, row_key)
        except Exception as exc:
            raise SkillExecutionPinningStorageError(
                "skill execution pinning storage read failed after document create",
            ) from exc
        if existing is None:
            raise SkillExecutionPinningStorageError(
                "skill execution pinning document create failed without readable record",
            )
        stored = decode_skill_execution_binding(_binding_record_to_bytes(existing))
        if not bindings_equivalent_for_pinning(stored, binding):
            raise SkillExecutionPinningConflictError(
                f"execution already pinned with different skill pack: {binding.execution_id}",
            )

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> SkillExecutionBinding | None:
        try:
            record = self._document_store.get(
                _pinning_document_partition(tenant_id),
                str(execution_id),
            )
        except Exception as exc:
            raise SkillExecutionPinningStorageError(
                "skill execution pinning storage read failed",
            ) from exc
        if record is None:
            return None
        binding = decode_skill_execution_binding(_binding_record_to_bytes(record))
        _assert_decoded_binding_scope(
            binding,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )
        return binding


def _binding_record_to_bytes(record: DocumentRecord) -> bytes:
    binding = record.data.get("binding")
    if not isinstance(binding, str):
        raise SkillExecutionPinningStorageError(
            "invalid skill execution binding document record",
        )
    return binding.encode("utf-8")


def wire_skill_execution_pinning_store(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> KvSkillExecutionPinningStore | DocumentStoreSkillExecutionPinningStore:
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_skill_execution_pinning_store accepts kv_store or document_store, not both",
        )
    if kv_store is not None:
        return KvSkillExecutionPinningStore(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "skill execution pinning requires ConditionalDocumentStore",
            )
        return DocumentStoreSkillExecutionPinningStore(document_store)
    raise ValueError(
        "wire_skill_execution_pinning_store requires kv_store or document_store",
    )


__all__ = [
    "DocumentStoreSkillExecutionPinningStore",
    "KvSkillExecutionPinningStore",
    "decode_skill_execution_binding",
    "encode_skill_execution_binding",
    "wire_skill_execution_pinning_store",
]
