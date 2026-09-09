"""Scenario-owned durable vector index metadata record for Qdrant targets."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity

INDEX_METADATA_LOGICAL_POINT_ID = "vpi:__index_identity_metadata__"
INDEX_METADATA_MARKER_PAYLOAD_KEY = "vpi_index_metadata_marker"
INDEX_METADATA_MARKER_VALUE = "v1"
DATA_PACK_CONTENT_IDENTITY_PAYLOAD_KEY = "data_pack_content_identity"
VECTOR_TARGET_LOGICAL_NAME_PAYLOAD_KEY = "vector_target_logical_name"
VECTOR_TARGET_TENANT_ID_PAYLOAD_KEY = "vector_target_tenant_id"


@dataclass(frozen=True, slots=True)
class VectorIndexPersistedMetadata:
    target: VectorIndexIdentity
    content_identity: str
    provider: str
    model: str
    revision: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.content_identity.strip():
            raise ValueError("content_identity must be non-empty")
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.revision.strip():
            raise ValueError("revision must be non-empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be > 0")
        if not self.target.logical_name.strip():
            raise ValueError("target.logical_name must be non-empty")
        if not self.target.tenant_id.strip():
            raise ValueError("target.tenant_id must be non-empty")

    def to_provider_payload(
        self,
        *,
        embedding_provider_key: str,
        embedding_model_key: str,
        embedding_revision_key: str,
        embedding_dimension_key: str,
    ) -> dict[str, str | int]:
        return {
            INDEX_METADATA_MARKER_PAYLOAD_KEY: INDEX_METADATA_MARKER_VALUE,
            DATA_PACK_CONTENT_IDENTITY_PAYLOAD_KEY: self.content_identity,
            VECTOR_TARGET_LOGICAL_NAME_PAYLOAD_KEY: self.target.logical_name,
            VECTOR_TARGET_TENANT_ID_PAYLOAD_KEY: self.target.tenant_id,
            embedding_provider_key: self.provider,
            embedding_model_key: self.model,
            embedding_revision_key: self.revision,
            embedding_dimension_key: self.dimension,
        }

    @classmethod
    def from_provider_payload(
        cls,
        raw_payload: dict[str, str | int],
        *,
        embedding_provider_key: str,
        embedding_model_key: str,
        embedding_revision_key: str,
        embedding_dimension_key: str,
    ) -> VectorIndexPersistedMetadata:
        marker = raw_payload.get(INDEX_METADATA_MARKER_PAYLOAD_KEY)
        if marker != INDEX_METADATA_MARKER_VALUE:
            raise ValueError("index metadata marker mismatch")
        content_identity = _require_str(raw_payload, DATA_PACK_CONTENT_IDENTITY_PAYLOAD_KEY)
        logical_name = _require_str(raw_payload, VECTOR_TARGET_LOGICAL_NAME_PAYLOAD_KEY)
        tenant_id = _require_str(raw_payload, VECTOR_TARGET_TENANT_ID_PAYLOAD_KEY)
        provider = _require_str(raw_payload, embedding_provider_key)
        model = _require_str(raw_payload, embedding_model_key)
        revision = _require_str(raw_payload, embedding_revision_key)
        dimension = _require_int(raw_payload, embedding_dimension_key)
        return cls(
            target=VectorIndexIdentity(logical_name=logical_name, tenant_id=tenant_id),
            content_identity=content_identity,
            provider=provider,
            model=model,
            revision=revision,
            dimension=dimension,
        )


def _require_str(raw_payload: dict[str, str | int], key: str) -> str:
    value = raw_payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"index metadata missing {key}")
    return value


def _require_int(raw_payload: dict[str, str | int], key: str) -> int:
    value = raw_payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"index metadata missing {key}")
    return value


__all__ = [
    "DATA_PACK_CONTENT_IDENTITY_PAYLOAD_KEY",
    "INDEX_METADATA_LOGICAL_POINT_ID",
    "INDEX_METADATA_MARKER_PAYLOAD_KEY",
    "INDEX_METADATA_MARKER_VALUE",
    "VECTOR_TARGET_LOGICAL_NAME_PAYLOAD_KEY",
    "VECTOR_TARGET_TENANT_ID_PAYLOAD_KEY",
    "VectorIndexPersistedMetadata",
]
