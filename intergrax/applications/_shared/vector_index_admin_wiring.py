# © Artur Czarnecki. All rights reserved.

"""Vector index operator service wiring (GR-12-A4-R2-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.vector_index_admin_service import VectorIndexAdminService
from intergrax.integrations.contracts.vector_index_administration import VectorIndexAdministration
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


@dataclass(frozen=True, slots=True)
class VectorIndexAdminWiring:
    service: VectorIndexAdminService


def build_vector_index_admin_service(
    *,
    vector_index_administration: VectorIndexAdministration,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
) -> VectorIndexAdminService:
    """Construct governed vector operator service without executing prepare."""
    return VectorIndexAdminService(
        vector_index_administration=vector_index_administration,
        mutation_authorization_boundary=mutation_authorization_boundary,
    )


def resolve_vector_index_admin_wiring(
    *,
    vector_index_administration: VectorIndexAdministration,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
) -> VectorIndexAdminWiring:
    return VectorIndexAdminWiring(
        service=build_vector_index_admin_service(
            vector_index_administration=vector_index_administration,
            mutation_authorization_boundary=mutation_authorization_boundary,
        ),
    )
