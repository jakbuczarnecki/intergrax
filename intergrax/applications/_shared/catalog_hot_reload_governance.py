# © Artur Czarnecki. All rights reserved.

"""CLA-04 request construction for integration catalog hot reload."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.contracts.integration_catalog_hot_reload import CatalogHotReloadPreset
from intergrax.contracts.integration_catalog_revision import (
    CANONICAL_INTEGRATION_CATALOG_ID,
    CatalogRevision,
)

MUTATION_TYPE_INTEGRATION_CATALOG_HOT_RELOAD = "integration_catalog.hot_reload"
INTEGRATION_CATALOG_RESOURCE_TYPE = "integration_catalog"
INTEGRATION_CATALOG_HOST_GLOBAL_SCOPE = "integration_catalog.host_global"


def integration_catalog_resource_scope() -> str:
    return INTEGRATION_CATALOG_HOST_GLOBAL_SCOPE


def build_integration_catalog_hot_reload_mutation_request(
    *,
    mutation_id: str,
    principal: RequestIdentity,
    preset: CatalogHotReloadPreset,
    current_revision: CatalogRevision,
    target_revision: CatalogRevision,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_INTEGRATION_CATALOG_HOT_RELOAD,
        principal=principal,
        resource_scope=integration_catalog_resource_scope(),
        resource_type=INTEGRATION_CATALOG_RESOURCE_TYPE,
        resource_id=CANONICAL_INTEGRATION_CATALOG_ID,
        current_revision=current_revision.as_mutation_revision_token(),
        target_revision=target_revision.as_mutation_revision_token(),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )
