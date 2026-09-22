# © Artur Czarnecki. All rights reserved.

"""Governed operator service for vector index prepare (GR-12-A4-R2-R1)."""

from __future__ import annotations

from intergrax.applications._shared.vector_index_admin_governance import (
    build_vector_index_prepare_mutation_request,
)
from intergrax.applications._shared.vector_index_configuration_projection import (
    VECTOR_INDEX_ABSENT_REVISION,
    configuration_revision_token,
    current_revision_from_description,
    project_vector_index_description,
    target_revision_from_spec,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.vector_index_operator import (
    VectorIndexPrepareOperatorRequest,
    VectorIndexPrepareOperatorResult,
)
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexAdministration,
    VectorIndexCompatibilityError,
    VectorIndexPrepareOutcome,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

BLOCKER_INVALID_IDENTITY = "VECTOR_INDEX_PREPARE_BLOCKED_INVALID_IDENTITY"
BLOCKER_MISSING_BOUNDARY = "VECTOR_INDEX_PREPARE_BLOCKED_MISSING_BOUNDARY"
BLOCKER_MISSING_PRINCIPAL = "VECTOR_INDEX_PREPARE_BLOCKED_MISSING_PRINCIPAL"
BLOCKER_POLICY = "VECTOR_INDEX_PREPARE_BLOCKED_BY_POLICY"
BLOCKER_POST_AUTH_STALE = "VECTOR_INDEX_PREPARE_BLOCKED_POST_AUTHORIZATION_STALE_REVISION"
BLOCKER_COMPATIBILITY = "VECTOR_INDEX_PREPARE_BLOCKED_INCOMPATIBLE_INDEX"


class VectorIndexAdminService:
    """Composition-owned governed vector prepare — no default ALLOW."""

    def __init__(
        self,
        *,
        vector_index_administration: VectorIndexAdministration,
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None,
    ) -> None:
        self._admin = vector_index_administration
        self._boundary = mutation_authorization_boundary

    @property
    def vector_index_administration(self) -> VectorIndexAdministration:
        return self._admin

    @property
    def mutation_authorization_boundary(
        self,
    ) -> ControlPlaneMutationAuthorizationBoundary | None:
        return self._boundary

    def prepare(
        self,
        request: VectorIndexPrepareOperatorRequest,
        *,
        principal: RequestIdentity | None = None,
    ) -> VectorIndexPrepareOperatorResult:
        identity = request.spec.identity
        if principal is None:
            return self._blocked(
                request,
                before_revision=VECTOR_INDEX_ABSENT_REVISION,
                after_revision=VECTOR_INDEX_ABSENT_REVISION,
                blocker_code=BLOCKER_MISSING_PRINCIPAL,
                policy_action="missing_operator_principal",
            )
        if self._boundary is None:
            return self._blocked(
                request,
                before_revision=VECTOR_INDEX_ABSENT_REVISION,
                after_revision=VECTOR_INDEX_ABSENT_REVISION,
                blocker_code=BLOCKER_MISSING_BOUNDARY,
                policy_action="missing_boundary",
            )
        if not identity.logical_name.strip() or not identity.tenant_id.strip():
            return self._blocked(
                request,
                before_revision=VECTOR_INDEX_ABSENT_REVISION,
                after_revision=VECTOR_INDEX_ABSENT_REVISION,
                blocker_code=BLOCKER_INVALID_IDENTITY,
                policy_action="invalid_operator_identity",
            )

        description = self._admin.describe_index(identity)
        before_revision = current_revision_from_description(description)
        target_revision = target_revision_from_spec(request.spec)

        mutation_request = build_vector_index_prepare_mutation_request(
            mutation_id=request.mutation_id,
            principal=principal,
            identity=identity,
            current_revision=before_revision,
            target_revision=target_revision,
        )
        authorization = self._boundary.authorize(mutation_request)
        if not authorization.permitted:
            return VectorIndexPrepareOperatorResult(
                mutation_id=request.mutation_id,
                before_revision=before_revision,
                after_revision=before_revision,
                changed=False,
                authorization_evidence=authorization.evidence,
                blocker_code=BLOCKER_POLICY,
                policy_action=authorization.decision.action.value,
            )

        authorized_current = before_revision
        reread = self._admin.describe_index(identity)
        reread_current = current_revision_from_description(reread)
        if reread_current != authorized_current:
            return VectorIndexPrepareOperatorResult(
                mutation_id=request.mutation_id,
                before_revision=reread_current,
                after_revision=reread_current,
                changed=False,
                authorization_evidence=authorization.evidence,
                blocker_code=BLOCKER_POST_AUTH_STALE,
                policy_action="post_authorization_stale_revision",
            )

        try:
            prepare_result = self._admin.prepare_index(request.spec)
        except VectorIndexCompatibilityError:
            after_revision = current_revision_from_description(reread)
            return VectorIndexPrepareOperatorResult(
                mutation_id=request.mutation_id,
                before_revision=before_revision,
                after_revision=after_revision,
                changed=False,
                authorization_evidence=authorization.evidence,
                blocker_code=BLOCKER_COMPATIBILITY,
                policy_action="vector_index_incompatible",
            )

        outcome = prepare_result.outcome
        changed = outcome is VectorIndexPrepareOutcome.CREATED
        final_description = prepare_result.description
        if final_description.exists:
            projection = project_vector_index_description(final_description)
            assert projection is not None
            after_revision = configuration_revision_token(projection)
        else:
            after_revision = VECTOR_INDEX_ABSENT_REVISION

        return VectorIndexPrepareOperatorResult(
            mutation_id=request.mutation_id,
            before_revision=before_revision,
            after_revision=after_revision,
            changed=changed,
            outcome=outcome,
            authorization_evidence=authorization.evidence,
            blocker_code=None,
            policy_action=authorization.decision.action.value,
        )

    @staticmethod
    def _blocked(
        request: VectorIndexPrepareOperatorRequest,
        *,
        before_revision: str,
        after_revision: str,
        blocker_code: str,
        policy_action: str,
    ) -> VectorIndexPrepareOperatorResult:
        return VectorIndexPrepareOperatorResult(
            mutation_id=request.mutation_id,
            before_revision=before_revision,
            after_revision=after_revision,
            changed=False,
            blocker_code=blocker_code,
            policy_action=policy_action,
        )
