# © Artur Czarnecki. All rights reserved.

"""Governed operator service for integration catalog hot reload (GR-12-A4-R1)."""

from __future__ import annotations

from intergrax.applications._shared.catalog_hot_reload_governance import (
    build_integration_catalog_hot_reload_mutation_request,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.integration_catalog_hot_reload import (
    CatalogHotReloadOperatorRequest,
    CatalogHotReloadResult,
)
from intergrax.integrations.registry.catalog_mutation import (
    CatalogReplaceOutcome,
    build_catalog_entries_for_preset,
    current_catalog_revision,
    replace_catalog_if_revision,
)
from intergrax.integrations.registry.catalog_revision import project_target_revision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

BLOCKER_MISSING_BOUNDARY = "CATALOG_HOT_RELOAD_BLOCKED_MISSING_BOUNDARY"
BLOCKER_PRECONDITION_REVISION = "CATALOG_HOT_RELOAD_BLOCKED_PRECONDITION_REVISION"
BLOCKER_POLICY = "CATALOG_HOT_RELOAD_BLOCKED_BY_POLICY"
BLOCKER_POST_AUTH_STALE = "CATALOG_HOT_RELOAD_BLOCKED_POST_AUTHORIZATION_STALE_REVISION"


class CatalogHotReloadService:
    """Composition-owned governed catalog hot reload — no default ALLOW."""

    def __init__(
        self,
        *,
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None,
        operator_principal: RequestIdentity,
    ) -> None:
        self._boundary = mutation_authorization_boundary
        self._principal = operator_principal

    @property
    def mutation_authorization_boundary(
        self,
    ) -> ControlPlaneMutationAuthorizationBoundary | None:
        return self._boundary

    def reload(self, request: CatalogHotReloadOperatorRequest) -> CatalogHotReloadResult:
        if self._boundary is None:
            return CatalogHotReloadResult(
                mutation_id=request.mutation_id,
                before_revision=request.expected_revision,
                after_revision=request.expected_revision,
                changed=False,
                blocker_code=BLOCKER_MISSING_BOUNDARY,
                policy_action="missing_boundary",
            )

        before = current_catalog_revision()
        if before != request.expected_revision:
            return CatalogHotReloadResult(
                mutation_id=request.mutation_id,
                before_revision=before,
                after_revision=before,
                changed=False,
                blocker_code=BLOCKER_PRECONDITION_REVISION,
                policy_action="precondition_revision_mismatch",
            )

        candidate = build_catalog_entries_for_preset(request.preset)
        target_revision = project_target_revision(before, candidate)
        mutation_request = build_integration_catalog_hot_reload_mutation_request(
            mutation_id=request.mutation_id,
            principal=self._principal,
            preset=request.preset,
            current_revision=before,
            target_revision=target_revision,
        )
        authorization = self._boundary.authorize(mutation_request)
        if not authorization.permitted:
            return CatalogHotReloadResult(
                mutation_id=request.mutation_id,
                before_revision=before,
                after_revision=before,
                changed=False,
                authorization_evidence=authorization.evidence,
                blocker_code=BLOCKER_POLICY,
                policy_action=str(authorization.decision.action),
            )

        after_authorize = current_catalog_revision()
        if after_authorize != before:
            return CatalogHotReloadResult(
                mutation_id=request.mutation_id,
                before_revision=after_authorize,
                after_revision=after_authorize,
                changed=False,
                authorization_evidence=authorization.evidence,
                blocker_code=BLOCKER_POST_AUTH_STALE,
                policy_action="post_authorization_stale_revision",
            )

        replace_result = replace_catalog_if_revision(
            expected_revision=before,
            candidate_entries=candidate,
        )
        changed = replace_result.outcome is CatalogReplaceOutcome.COMMITTED
        return CatalogHotReloadResult(
            mutation_id=request.mutation_id,
            before_revision=replace_result.before_revision,
            after_revision=replace_result.after_revision,
            changed=changed,
            authorization_evidence=authorization.evidence,
            blocker_code=(
                BLOCKER_POST_AUTH_STALE
                if replace_result.outcome is CatalogReplaceOutcome.REVISION_CONFLICT
                else None
            ),
            policy_action=(
                "post_authorization_cas_conflict"
                if replace_result.outcome is CatalogReplaceOutcome.REVISION_CONFLICT
                else str(authorization.decision.action)
            ),
        )
