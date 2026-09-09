"""Qdrant-specific composition and lifecycle for vector round-trip qualification."""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexAdministration,
    VectorIndexIdentity,
    VectorSearchCapability,
)
from intergrax.integrations.contracts.vector_store import VectorStore, VectorStoreScope
from intergrax.integrations.providers.vector_store.qdrant.config import (
    ENV_QDRANT_METRIC,
    QdrantIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_vector_index_administration,
    open_qdrant_vector_store,
)

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_bootstrap_adapter import (
    PlatformSearchIndexBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    qdrant_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.contracts import (
    QdrantIndexSnapshot,
    RankedVectorHit,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)


class QdrantResourcePreconditionError(RuntimeError):
    """Raised when Qdrant cannot be prepared for qualification."""


class QdrantMetricConfigurationError(RuntimeError):
    """Raised when configured Qdrant metric is incompatible with qualification."""


@dataclass(frozen=True, slots=True)
class QdrantQualificationRuntime:
    collection_name: str
    adapter: PlatformSearchIndexBootstrapAdapter
    index_admin: VectorIndexAdministration
    vector_store: VectorStore
    scope: VectorStoreScope
    config: QdrantIntegrationConfig


def _configured_metric() -> str:
    return os.environ.get(ENV_QDRANT_METRIC, "cosine").strip() or "cosine"


def assert_cosine_metric_configuration() -> str:
    metric = _configured_metric()
    if metric != "cosine":
        msg = (
            f"INTERGRAX_QDRANT_METRIC={metric!r} is incompatible with qualification; "
            "expected cosine"
        )
        raise QdrantMetricConfigurationError(msg)
    return metric


def _try_probe_qdrant() -> bool:
    collection_name = f"vpi_probe_{uuid.uuid4().hex[:8]}"
    runtime = open_qdrant_qualification_runtime(collection_name)
    try:
        report = runtime.adapter.probe_readiness()
        return report.status.value == "PASS"
    finally:
        delete_qualification_collection(runtime.collection_name, config=runtime.config)
        runtime.adapter.close()


def _start_repository_qdrant_service() -> None:
    compose_file = "infra/integration/docker-compose.yml"
    commands = (
        ["docker", "start", "intergrax-qdrant"],
        ["docker", "compose", "-f", compose_file, "up", "-d", "qdrant"],
    )
    for command in commands:
        try:
            subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError:
            continue
        if _try_probe_qdrant():
            return
    if not _try_probe_qdrant():
        msg = "Qdrant is not reachable after docker start attempts"
        raise QdrantResourcePreconditionError(msg)


def ensure_qdrant_available() -> None:
    assert_cosine_metric_configuration()
    if _try_probe_qdrant():
        return
    if qdrant_environment_available():
        msg = "Qdrant is configured but not reachable"
        raise QdrantResourcePreconditionError(msg)
    _start_repository_qdrant_service()


def make_qualification_collection_name() -> str:
    return f"vpi_5c4e5a_roundtrip_{uuid.uuid4().hex[:8]}"


def open_qdrant_qualification_runtime(collection_name: str) -> QdrantQualificationRuntime:
    metric = assert_cosine_metric_configuration()
    if metric != "cosine":
        raise QdrantMetricConfigurationError("qualification requires cosine metric")
    config = QdrantIntegrationConfig.from_env(
        collection_name=collection_name,
        enable_sparse_vectors=True,
        metric="cosine",
    )
    index_admin = open_qdrant_vector_index_administration(config)
    vector_store = open_qdrant_vector_store(config)
    adapter = PlatformSearchIndexBootstrapAdapter(
        _index_admin=index_admin,
        _vector_store=vector_store,
        _index_identity=VectorIndexIdentity(
            logical_name=collection_name,
            tenant_id=config.tenant_id,
        ),
        _dense_channel_name="dense",
        _sparse_channel_name="sparse",
        _sparse_required=config.enable_sparse_vectors,
    )
    scope = VectorStoreScope(tenant_id=config.tenant_id)
    return QdrantQualificationRuntime(
        collection_name=collection_name,
        adapter=adapter,
        index_admin=index_admin,
        vector_store=vector_store,
        scope=scope,
        config=config,
    )


def delete_qualification_collection(
    collection_name: str,
    *,
    config: QdrantIntegrationConfig,
) -> bool:
    from qdrant_client import QdrantClient

    resolved_url = config.resolved_url()
    if resolved_url:
        client = QdrantClient(url=resolved_url, api_key=config.api_key or None)
    else:
        client = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key or None,
        )
    collections = {item.name for item in client.get_collections().collections}
    if collection_name not in collections:
        return True
    client.delete_collection(collection_name)
    remaining = {item.name for item in client.get_collections().collections}
    return collection_name not in remaining


def describe_qdrant_index(
    runtime: QdrantQualificationRuntime,
) -> QdrantIndexSnapshot:
    description = runtime.index_admin.describe_index(
        VectorIndexIdentity(
            logical_name=runtime.collection_name,
            tenant_id=runtime.config.tenant_id,
        )
    )
    dense_available = VectorSearchCapability.DENSE in description.present_capabilities
    return QdrantIndexSnapshot(
        collection_name=runtime.collection_name,
        metric=_configured_metric(),
        dimension=description.dense_dimension or 0,
        point_count=description.point_count,
        dense_search_available=dense_available,
        temporary_isolated_collection=True,
        cleanup_passed=False,
    )


def _source_ref_from_hit_metadata(
    metadata: Mapping[str, str | int | float | bool | None],
) -> SourceRecordRef:
    offer_id_raw = metadata.get("offer_id")
    if offer_id_raw is None:
        raise VpiBootstrapProviderError("qdrant hit metadata missing offer_id")
    catalog_id_raw = metadata.get("catalog_id")
    if catalog_id_raw is None:
        raise VpiBootstrapProviderError("qdrant hit metadata missing catalog_id")
    source_revision_raw = metadata.get("source_revision")
    source_revision = str(source_revision_raw) if source_revision_raw is not None else None
    return SourceRecordRef(
        offer_id=ProductOfferId(str(offer_id_raw)),
        catalog_id=str(catalog_id_raw),
        source_revision=source_revision,
    )


def query_qdrant_ranked_hits(
    runtime: QdrantQualificationRuntime,
    query_vector: NDArray[np.float64] | Sequence[float],
    *,
    top_k: int,
) -> tuple[tuple[RankedVectorHit, ...], dict[str, SourceRecordRef]]:
    vector = np.asarray(query_vector, dtype=np.float32)
    hits = runtime.vector_store.query(
        vector,
        scope=runtime.scope,
        top_k=top_k,
    )
    ranked: list[RankedVectorHit] = []
    source_refs: dict[str, SourceRecordRef] = {}
    for rank, hit in enumerate(hits, start=1):
        logical_point_id = hit.vector_id
        ranked.append(
            RankedVectorHit(
                logical_point_id=logical_point_id,
                rank=rank,
                cosine_score=float(hit.similarity_score),
            )
        )
        source_refs[logical_point_id] = _source_ref_from_hit_metadata(hit.metadata)
    return tuple(ranked), source_refs
