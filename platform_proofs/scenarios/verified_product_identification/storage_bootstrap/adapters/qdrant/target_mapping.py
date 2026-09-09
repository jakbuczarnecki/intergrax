"""Fail-closed mapping from logical vector targets to physical Qdrant collections."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    DEFAULT_LOGICAL_COLLECTION_NAME,
    QdrantBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.errors import (
    QdrantBootstrapConfigurationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorTargetId,
)

_APPROVED_LOGICAL_TARGETS: dict[str, str] = {
    "vpi-product-embeddings": DEFAULT_LOGICAL_COLLECTION_NAME,
}


def physical_collection_name(logical_collection_name: str, tenant_id: str) -> str:
    return f"{logical_collection_name}__tenant__{tenant_id}"


@dataclass(frozen=True, slots=True)
class PhysicalVectorTarget:
    index_identity: VectorIndexIdentity
    collection_name: str
    uses_named_dense_vector: bool
    dense_vector_channel_name: str


def resolve_physical_target(
    logical_target: VectorTargetId,
    configuration: QdrantBootstrapConfiguration,
) -> PhysicalVectorTarget:
    logical_key = str(logical_target).strip()
    mapped_collection = _APPROVED_LOGICAL_TARGETS.get(logical_key)
    if mapped_collection is None:
        raise QdrantBootstrapConfigurationError(
            f"unmapped vector target: {logical_key}"
        )
    if mapped_collection != configuration.logical_collection_name:
        raise QdrantBootstrapConfigurationError(
            f"vector target {logical_key} requires collection {mapped_collection!r}, "
            f"but configuration specifies {configuration.logical_collection_name!r}"
        )
    tenant_id = configuration.integration.tenant_id
    return PhysicalVectorTarget(
        index_identity=VectorIndexIdentity(
            logical_name=mapped_collection,
            tenant_id=tenant_id,
        ),
        collection_name=physical_collection_name(mapped_collection, tenant_id),
        uses_named_dense_vector=configuration.uses_named_dense_vector,
        dense_vector_channel_name=configuration.dense_vector_channel_name,
    )


def reject_unsafe_logical_target(logical_target: str) -> None:
    candidate = logical_target.strip()
    if candidate in _APPROVED_LOGICAL_TARGETS:
        return
    if ";" in candidate or "--" in candidate or "'" in candidate:
        raise QdrantBootstrapConfigurationError(
            f"unsafe vector target rejected: {logical_target!r}"
        )
    raise QdrantBootstrapConfigurationError(
        f"unmapped vector target: {logical_target!r}"
    )
