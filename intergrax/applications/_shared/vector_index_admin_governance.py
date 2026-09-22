# © Artur Czarnecki. All rights reserved.

"""CLA-04 request construction for governed vector index prepare."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity

MUTATION_TYPE_VECTOR_INDEX_PREPARE = "vector_index.prepare"
VECTOR_INDEX_RESOURCE_TYPE = "vector_index"


def vector_index_resource_id(identity: VectorIndexIdentity) -> str:
    return f"{identity.tenant_id}/{identity.logical_name}"


def vector_index_resource_scope(tenant_id: str) -> str:
    return f"vector_index.tenant/{tenant_id}"


def build_vector_index_prepare_mutation_request(
    *,
    mutation_id: str,
    principal: RequestIdentity,
    identity: VectorIndexIdentity,
    current_revision: str,
    target_revision: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_VECTOR_INDEX_PREPARE,
        principal=principal,
        resource_scope=vector_index_resource_scope(identity.tenant_id),
        resource_type=VECTOR_INDEX_RESOURCE_TYPE,
        resource_id=vector_index_resource_id(identity),
        current_revision=current_revision,
        target_revision=target_revision,
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )
