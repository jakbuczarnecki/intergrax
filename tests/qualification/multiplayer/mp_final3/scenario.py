# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-3 scenario runner — invokes existing public services only.

Owner annotations (domain → contract):
  Principal/Membership/Authority → Collaborative Work → collaborative_work
  WorkItem / Assignment → Collaborative Work → collaborative_work
  WorkArtifact / Version → Collaborative Work → collaborative_work
  Decision binding → Multiplayer binding; Decision truth → Decision/Governance
  ContextView → Multiplayer eligibility/projection → context_view*
  Collaborative Activity → Collaborative Work → collaborative_activity*
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivity,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityQuery,
)
from intergrax.contracts.collaborative_activity_read import CollaborativeActivityReadRequest
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBinding,
    CreateCollaborativeDecisionBindingRequest,
)
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    Assignment,
    CreateAssignmentRequest,
    CreateWorkArtifactRequest,
    CreateWorkItemRequest,
    MembershipResolutionMode,
    PublishWorkArtifactVersionRequest,
    WorkItem,
)
from intergrax.contracts.context_view import (
    ContextView,
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewRequest,
    ContextViewScope,
)
from intergrax.contracts.context_view_composition import ContextViewCompositionRequest
from intergrax.contracts.context_view_visibility_policy import ContextViewPolicyOutcome
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    DecisionProposalRef,
    decision_lineage_ref,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.collaborative_work.repository import INITIAL_RECORD_REVISION
from tests.qualification.multiplayer.mp_final3.host import (
    AUTHORIZED_PRINCIPAL,
    MpFinal3Host,
    READER_PRINCIPAL,
    TENANT_A,
    WS_A,
)

_ARTIFACT_DIGEST = "sha256:" + ("b" * 64)

WORK_ITEM_ID = "mpf3-work-item-1"
ASSIGNMENT_ID = "mpf3-assignment-1"
ARTIFACT_ID = "mpf3-artifact-1"
VERSION_1_ID = "mpf3-artifact-ver-1"
VERSION_2_ID = "mpf3-artifact-ver-2"


@dataclass(frozen=True, slots=True)
class CapabilityWideScenarioResult:
    """Cross-primitive IDs produced by the certified happy-path scenario."""

    work_item: WorkItem
    assignment: Assignment
    artifact_version_id: str
    binding: CollaborativeDecisionBinding
    decision_proposal: DecisionProposalRef
    context_view: ContextView
    activities: tuple[CollaborativeActivity, ...]


def _artifact_content_ref(*, content_ref: str) -> ArtifactContentRef:
    return ArtifactContentRef(
        content_ref=content_ref,
        media_type="application/json",
        integrity_digest=_ARTIFACT_DIGEST,
    )


def _decision_proposal(*, tenant_id: str = TENANT_A) -> DecisionProposalRef:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="mp-final-3", subject="capability-wide-e2e"),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    return DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )


def read_authorized_activity_page(
    host: MpFinal3Host,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WS_A,
    acting_principal_id: str = READER_PRINCIPAL,
    limit: int = 100,
):
    membership = host.bundle.membership.get_for_principal(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=acting_principal_id,
    )
    assert membership is not None
    return host.activity_read_service.read_page(
        CollaborativeActivityReadRequest(
            query=CollaborativeActivityQuery(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                limit=limit,
            ),
            acting_principal_id=acting_principal_id,
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=membership,
        ),
    )


def compose_context_view_for_principal(
    host: MpFinal3Host,
    *,
    tenant_id: str,
    workspace_id: str,
    acting_principal_id: str,
    work_item_id: str | None = None,
    publish_activity: bool = True,
) -> tuple[ContextViewPolicyOutcome, ContextView | None]:
    scope = ContextViewScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
    )
    request = ContextViewRequest(
        scope=scope,
        acting_principal_id=acting_principal_id,
        operation_id="collaborative_work.context_view.compose",
        requested_categories=(ContextViewCategory.COLLABORATIVE_WORK,),
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
    )
    decision = host.context_view_evaluator.evaluate(request)
    if decision.outcome is not ContextViewPolicyOutcome.ALLOW:
        return decision.outcome, None
    identity = RequestIdentity(
        tenant_id=tenant_id,
        auth_subject=acting_principal_id,
        principal_type=PrincipalType.USER,
        user_id=acting_principal_id,
    )
    composer = (
        host.context_view_composer
        if publish_activity
        else host.raw_context_view_composer
    )
    view = composer.compose(
        ContextViewCompositionRequest(
            request=request,
            policy_decision=decision,
            principal_identity=identity,
        ),
    )
    return decision.outcome, view


def run_capability_wide_happy_path(host: MpFinal3Host) -> CapabilityWideScenarioResult:
    """Authorized principal walks Principal→…→Activity through public services."""
    # Phase B — WorkItem (Collaborative Work)
    work_item = host.work_service.create_work_item(
        CreateWorkItemRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=WORK_ITEM_ID,
            acting_principal_id=AUTHORIZED_PRINCIPAL,
            idempotency_key="mpf3-wi-create",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )

    # Phase C — Assignment (Collaborative Work)
    assignment = host.work_service.create_assignment(
        CreateAssignmentRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            assignment_id=ASSIGNMENT_ID,
            work_item_id=WORK_ITEM_ID,
            principal_id=AUTHORIZED_PRINCIPAL,
            acting_principal_id=AUTHORIZED_PRINCIPAL,
            idempotency_key="mpf3-asg-create",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )

    # Phase D — WorkArtifact + version (Collaborative Work; content via ArtifactContentRef)
    published = host.artifact_service.create_artifact(
        CreateWorkArtifactRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=WORK_ITEM_ID,
            work_artifact_id=ARTIFACT_ID,
            work_artifact_version_id=VERSION_1_ID,
            acting_principal_id=AUTHORIZED_PRINCIPAL,
            content_ref=_artifact_content_ref(
                content_ref=f"content://{TENANT_A}/{WS_A}/mpf3-body-v1",
            ),
            idempotency_key="mpf3-artifact-create",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    host.artifact_service.publish_version(
        PublishWorkArtifactVersionRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=WORK_ITEM_ID,
            work_artifact_id=ARTIFACT_ID,
            work_artifact_version_id=VERSION_2_ID,
            expected_revision=INITIAL_RECORD_REVISION,
            acting_principal_id=AUTHORIZED_PRINCIPAL,
            content_ref=_artifact_content_ref(
                content_ref=f"content://{TENANT_A}/{WS_A}/mpf3-body-v2",
            ),
            idempotency_key="mpf3-artifact-publish",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )

    # Phase E — Decision binding (Multiplayer binding; DecisionProposalRef from Decision/Governance)
    decision_proposal = _decision_proposal()
    binding = host.decision_binding_service.create_binding(
        CreateCollaborativeDecisionBindingRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=WORK_ITEM_ID,
            decision_proposal=decision_proposal,
            acting_principal_id=AUTHORIZED_PRINCIPAL,
            idempotency_key="mpf3-decision-binding",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )

    # Phase F — ContextView (eligibility/projection; CW refs only — no Memory/RAG ownership)
    outcome, context_view = compose_context_view_for_principal(
        host,
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        acting_principal_id=AUTHORIZED_PRINCIPAL,
        work_item_id=WORK_ITEM_ID,
    )
    assert outcome is ContextViewPolicyOutcome.ALLOW
    assert context_view is not None

    # Phase G — Collaborative Activity (authorized read path)
    page = read_authorized_activity_page(host)
    return CapabilityWideScenarioResult(
        work_item=work_item,
        assignment=assignment,
        artifact_version_id=published.version.work_artifact_version_id,
        binding=binding,
        decision_proposal=decision_proposal,
        context_view=context_view,
        activities=page.activities,
    )


def assert_happy_path_consistency(result: CapabilityWideScenarioResult) -> None:
    """Cross-primitive ID / provenance consistency for the certified scenario."""
    assert result.work_item.work_item_id == WORK_ITEM_ID
    assert result.work_item.tenant_id == TENANT_A
    assert result.work_item.workspace_id == WS_A
    assert result.assignment.work_item_id == WORK_ITEM_ID
    assert result.assignment.principal_id == AUTHORIZED_PRINCIPAL
    assert result.assignment.assignment_id == ASSIGNMENT_ID
    assert result.artifact_version_id == VERSION_1_ID
    assert result.binding.work_item_id == WORK_ITEM_ID
    assert result.binding.decision_proposal.identity.decision_id == (
        result.decision_proposal.identity.decision_id
    )
    loaded = result.binding  # create return is authoritative; get_binding available on service
    assert loaded.created_by_principal_id == AUTHORIZED_PRINCIPAL

    cw_refs = [
        entry.source_ref
        for entry in result.context_view.entries
        if type(entry.source_ref) is ContextViewCollaborativeWorkSourceRef
    ]
    assert cw_refs, "ContextView must project Collaborative Work references"
    work_item_ids = {ref.work_item_id for ref in cw_refs if ref.work_item_id}
    assert WORK_ITEM_ID in work_item_ids
    artifact_version_refs = [
        ref.work_artifact_version
        for ref in cw_refs
        if ref.work_artifact_version is not None
    ]
    assert artifact_version_refs, (
        "ContextView must project WorkArtifactVersionRef when artifacts exist"
    )
    assert ARTIFACT_ID in {v.work_artifact_id for v in artifact_version_refs}
    assert VERSION_2_ID in {
        v.work_artifact_version_id for v in artifact_version_refs
    }, "ContextView collaborative-work projection must surface current published version"

    observed = [a.activity_type for a in result.activities]
    required = (
        CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
        CollaborativeActivityBuiltinType.ASSIGNMENT_CREATED,
        CollaborativeActivityBuiltinType.WORK_ARTIFACT_CREATED,
        CollaborativeActivityBuiltinType.WORK_ARTIFACT_VERSION_PUBLISHED,
        CollaborativeActivityBuiltinType.COLLABORATIVE_DECISION_BINDING_CREATED,
        CollaborativeActivityBuiltinType.CONTEXT_VIEW_COMPOSED,
    )
    for required_type in required:
        assert required_type in observed
    for activity in result.activities:
        assert activity.scope.tenant_id == TENANT_A
        assert activity.scope.workspace_id == WS_A
