# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2-R1 vector configuration projection and digest proofs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.applications._shared.vector_index_configuration_projection import (
    VECTOR_INDEX_ABSENT_REVISION,
    configuration_digest,
    current_revision_from_description,
    project_vector_index_description,
    project_vector_index_spec,
    target_revision_from_spec,
)
from intergrax.integrations.contracts.vector_index_administration import (
    DenseVectorChannelSpec,
    SparseLexicalChannelSpec,
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexSpec,
    VectorSearchCapability,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _identity(
    *,
    logical_name: str = "catalog",
    tenant_id: str = "tenant-a",
) -> VectorIndexIdentity:
    return VectorIndexIdentity(logical_name=logical_name, tenant_id=tenant_id)


def _spec(
    *,
    dimension: int = 1024,
    metric: str = "cosine",
    dense_channel: str = "dense",
    sparse_channel: str | None = "sparse",
    logical_name: str = "catalog",
    tenant_id: str = "tenant-a",
) -> VectorIndexSpec:
    sparse_spec = (
        SparseLexicalChannelSpec(channel_name=sparse_channel) if sparse_channel else None
    )
    caps = {VectorSearchCapability.DENSE}
    if sparse_spec is not None:
        caps.add(VectorSearchCapability.SPARSE_LEXICAL)
    return VectorIndexSpec(
        identity=_identity(logical_name=logical_name, tenant_id=tenant_id),
        dense=DenseVectorChannelSpec(
            channel_name=dense_channel,
            dimension=dimension,
            metric=metric,
        ),
        required_capabilities=frozenset(caps),
        sparse_lexical=sparse_spec,
    )


def _description(
    *,
    exists: bool = True,
    dimension: int = 1024,
    metric: str = "cosine",
    dense_channel: str = "dense",
    sparse_channel: str | None = "sparse",
    point_count: int = 0,
    reachable: bool = True,
    logical_name: str = "catalog",
    tenant_id: str = "tenant-a",
) -> VectorIndexDescription:
    caps = {VectorSearchCapability.DENSE}
    if sparse_channel:
        caps.add(VectorSearchCapability.SPARSE_LEXICAL)
    return VectorIndexDescription(
        identity=_identity(logical_name=logical_name, tenant_id=tenant_id),
        exists=exists,
        reachable=reachable,
        point_count=point_count,
        dense_dimension=dimension if exists else None,
        dense_metric=metric if exists else None,
        present_capabilities=frozenset(caps) if exists else frozenset(),
        dense_channel_name=dense_channel if exists else None,
        sparse_lexical_channel_name=sparse_channel if exists else None,
    )


def test_vec_digest_capability_order_irrelevant() -> None:
    spec_a = _spec()
    spec_b = replace(
        spec_a,
        required_capabilities=frozenset(
            {
                VectorSearchCapability.SPARSE_LEXICAL,
                VectorSearchCapability.DENSE,
            }
        ),
    )
    assert target_revision_from_spec(spec_a) == target_revision_from_spec(spec_b)


def test_vec_digest_point_count_irrelevant() -> None:
    low = _description(point_count=0)
    high = _description(point_count=999_999)
    assert current_revision_from_description(low) == current_revision_from_description(high)


def test_vec_digest_reachable_irrelevant() -> None:
    up = _description(reachable=True)
    down = _description(reachable=False)
    assert current_revision_from_description(up) == current_revision_from_description(down)


def test_vec_digest_metric_change() -> None:
    a = _spec(metric="cosine")
    b = _spec(metric="dot")
    assert target_revision_from_spec(a) != target_revision_from_spec(b)


def test_vec_digest_dimension_change() -> None:
    assert target_revision_from_spec(_spec(dimension=512)) != target_revision_from_spec(
        _spec(dimension=1024)
    )


def test_vec_digest_channel_name_change() -> None:
    assert target_revision_from_spec(_spec(dense_channel="dense-a")) != target_revision_from_spec(
        _spec(dense_channel="dense-b")
    )


def test_vec_digest_tenant_change() -> None:
    assert target_revision_from_spec(_spec(tenant_id="t1")) != target_revision_from_spec(
        _spec(tenant_id="t2")
    )


def test_vec_digest_logical_name_change() -> None:
    assert target_revision_from_spec(_spec(logical_name="a")) != target_revision_from_spec(
        _spec(logical_name="b")
    )


def test_vec_absent_revision_token() -> None:
    missing = _description(exists=False)
    assert current_revision_from_description(missing) == VECTOR_INDEX_ABSENT_REVISION


def test_vec_spec_and_description_symmetry() -> None:
    spec = _spec()
    description = _description()
    spec_projection = project_vector_index_spec(spec)
    desc_projection = project_vector_index_description(description)
    assert desc_projection is not None
    assert configuration_digest(spec_projection) == configuration_digest(desc_projection)
