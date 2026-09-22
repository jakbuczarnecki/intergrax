# © Artur Czarnecki. All rights reserved.

"""Provider-neutral vector index configuration projection and revision digest (GR-12-A4-R2-R0)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.canonical_payload_hash import stable_payload_hash
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexDescription,
    VectorIndexSpec,
    VectorSearchCapability,
)
from intergrax.rag.vectorstore.config.vector_config import Metric

VECTOR_INDEX_ABSENT_REVISION: str = "ABSENT"


class VectorIndexConfigurationProjectionError(ValueError):
    """Cannot project persisted description into configuration authority."""


@dataclass(frozen=True, slots=True)
class VectorIndexConfigurationProjection:
    """Single logical schema for current (description) and target (spec) revisions."""

    logical_name: str
    tenant_id: str
    dense_dimension: int
    dense_metric: Metric
    dense_channel_name: str
    required_capabilities: frozenset[str]
    sparse_lexical_channel_name: str | None = None


def _sorted_capability_values(
    capabilities: frozenset[VectorSearchCapability] | frozenset[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for item in capabilities:
        if isinstance(item, VectorSearchCapability):
            values.append(item.value)
        else:
            values.append(str(item))
    return tuple(sorted(values))


def _projection_payload(projection: VectorIndexConfigurationProjection) -> dict[str, object]:
    return {
        "logical_name": projection.logical_name,
        "tenant_id": projection.tenant_id,
        "dense_dimension": projection.dense_dimension,
        "dense_metric": projection.dense_metric,
        "dense_channel_name": projection.dense_channel_name,
        "required_capabilities": _sorted_capability_values(
            projection.required_capabilities
        ),
        "sparse_lexical_channel_name": projection.sparse_lexical_channel_name,
    }


def configuration_digest(projection: VectorIndexConfigurationProjection) -> str:
    return stable_payload_hash(_projection_payload(projection))


def configuration_revision_token(projection: VectorIndexConfigurationProjection) -> str:
    return configuration_digest(projection)


def project_vector_index_spec(spec: VectorIndexSpec) -> VectorIndexConfigurationProjection:
    sparse_name = (
        spec.sparse_lexical.channel_name if spec.sparse_lexical is not None else None
    )
    return VectorIndexConfigurationProjection(
        logical_name=spec.identity.logical_name,
        tenant_id=spec.identity.tenant_id,
        dense_dimension=spec.dense.dimension,
        dense_metric=spec.dense.metric,
        dense_channel_name=spec.dense.channel_name,
        required_capabilities=frozenset(
            _sorted_capability_values(spec.required_capabilities)
        ),
        sparse_lexical_channel_name=sparse_name,
    )


def project_vector_index_description(
    description: VectorIndexDescription,
) -> VectorIndexConfigurationProjection | None:
    if not description.exists:
        return None
    if description.dense_dimension is None:
        raise VectorIndexConfigurationProjectionError(
            "existing vector index description missing dense_dimension"
        )
    if description.dense_metric is None:
        raise VectorIndexConfigurationProjectionError(
            "existing vector index description missing dense_metric"
        )
    if not (description.dense_channel_name or "").strip():
        raise VectorIndexConfigurationProjectionError(
            "existing vector index description missing dense_channel_name"
        )
    return VectorIndexConfigurationProjection(
        logical_name=description.identity.logical_name,
        tenant_id=description.identity.tenant_id,
        dense_dimension=description.dense_dimension,
        dense_metric=description.dense_metric,
        dense_channel_name=description.dense_channel_name,
        required_capabilities=frozenset(
            _sorted_capability_values(description.present_capabilities)
        ),
        sparse_lexical_channel_name=description.sparse_lexical_channel_name,
    )


def current_revision_from_description(description: VectorIndexDescription) -> str:
    if not description.exists:
        return VECTOR_INDEX_ABSENT_REVISION
    projection = project_vector_index_description(description)
    assert projection is not None
    return configuration_revision_token(projection)


def target_revision_from_spec(spec: VectorIndexSpec) -> str:
    return configuration_revision_token(project_vector_index_spec(spec))
