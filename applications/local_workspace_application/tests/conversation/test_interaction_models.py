# © Artur Czarnecki. All rights reserved.

"""Tests for conversational interaction plan models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_workspace_application.conversation.interaction_models import (
    ConversationClarification,
    ConversationInteractionPlan,
    ConversationPlanningAttachment,
    ConversationPlanningRequest,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddLocalReferencesPlannedAction,
    KnowledgeAddWebUrlsPlannedAction,
    LocalReference,
    WorkspaceActivatePlannedAction,
    WorkspaceCreatePlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)


def _magazyn_target() -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn")


@pytest.mark.unit
def test_valid_plan_with_multiple_actions() -> None:
    plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="a1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://www.cenniki.pl",),
                evidence_quotes=("cenniki.pl",),
            ),
            KnowledgeAddLocalReferencesPlannedAction(
                action_id="a2",
                action_type="knowledge.add_local_references",
                workspace=_magazyn_target(),
                references=(LocalReference(kind="file", value=r"c:\moje dokumenty\cenniki.xls"),),
                evidence_quotes=("cenniki.xls",),
            ),
        ),
    )
    assert len(plan.actions) == 2
    assert plan.actions[0].action_type == "knowledge.add_web_urls"
    assert plan.actions[1].action_type == "knowledge.add_local_references"


@pytest.mark.unit
def test_rejects_unknown_action_type() -> None:
    with pytest.raises(ValidationError):
        ConversationInteractionPlan.model_validate(
            {
                "plan_version": "1",
                "response_mode": "aggregate",
                "actions": [
                    {
                        "action_id": "x1",
                        "action_type": "workspace.unknown",
                    }
                ],
            }
        )


@pytest.mark.unit
def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ConversationPlanningRequest(
            message_text="hello",
            attachments=(
                ConversationPlanningAttachment(
                    attachment_id="att-1",
                    file_name="doc.pdf",
                    slack_token="secret",  # type: ignore[call-arg]
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_duplicate_action_id() -> None:
    action = KnowledgeAddWebUrlsPlannedAction(
        action_id="dup",
        action_type="knowledge.add_web_urls",
        workspace=_magazyn_target(),
        urls=("https://example.com",),
    )
    with pytest.raises(ValidationError, match="duplicate action_id"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
            actions=(action, action),
        )


@pytest.mark.unit
def test_rejects_dependency_to_missing_action() -> None:
    with pytest.raises(ValidationError, match="unknown dependency"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
            actions=(
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_web_urls",
                    workspace=_magazyn_target(),
                    urls=("https://example.com",),
                    depends_on=("missing",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError, match="self dependency"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
            actions=(
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_web_urls",
                    workspace=_magazyn_target(),
                    urls=("https://example.com",),
                    depends_on=("a1",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_dependency_cycle() -> None:
    with pytest.raises(ValidationError, match="dependency cycle"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
            actions=(
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_web_urls",
                    workspace=_magazyn_target(),
                    urls=("https://a.example",),
                    depends_on=("a2",),
                ),
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="a2",
                    action_type="knowledge.add_web_urls",
                    workspace=_magazyn_target(),
                    urls=("https://b.example",),
                    depends_on=("a1",),
                ),
            ),
        )


@pytest.mark.unit
def test_created_by_action_must_reference_workspace_create() -> None:
    with pytest.raises(ValidationError, match="created_by_action must reference workspace.create"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
            actions=(
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="not-create",
                    action_type="knowledge.add_web_urls",
                    workspace=_magazyn_target(),
                    urls=("https://example.com",),
                ),
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_web_urls",
                    workspace=WorkspaceReference(
                        kind=WorkspaceReferenceKind.created_by_action,
                        value="not-create",
                    ),
                    urls=("https://example.org",),
                    depends_on=("not-create",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_created_by_action_without_required_dependency() -> None:
    with pytest.raises(ValidationError, match="must depend on the workspace.create action"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
            actions=(
                WorkspaceCreatePlannedAction(
                    action_id="create-1",
                    action_type="workspace.create",
                    name="Nowy",
                ),
                KnowledgeAddWebUrlsPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_web_urls",
                    workspace=WorkspaceReference(
                        kind=WorkspaceReferenceKind.created_by_action,
                        value="create-1",
                    ),
                    urls=("https://example.com",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_empty_plan_without_clarification() -> None:
    with pytest.raises(ValidationError, match="at least one action or clarification"):
        ConversationInteractionPlan(
            plan_version="1",
            response_mode="aggregate",
        )


@pytest.mark.unit
def test_valid_mixed_source_plan_with_shared_workspace_target() -> None:
    request = ConversationPlanningRequest(
        message_text=(
            "dołącz informacje o cennikach ze strony https://www.cenniki.pl "
            "oraz dorzuć moją kopię lokalną cenników z "
            r"c:\moje dokumenty\cenniki.xls "
            'a to wszystko do workspace "magazyn"'
        ),
        attachments=(
            ConversationPlanningAttachment(attachment_id="att-1", file_name="extra.pdf"),
        ),
    )
    plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="url-1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://www.cenniki.pl",),
            ),
            KnowledgeAddLocalReferencesPlannedAction(
                action_id="local-1",
                action_type="knowledge.add_local_references",
                workspace=_magazyn_target(),
                references=(LocalReference(kind="file", value=r"c:\moje dokumenty\cenniki.xls"),),
            ),
            KnowledgeAddAttachmentsPlannedAction(
                action_id="att-action",
                action_type="knowledge.add_attachments",
                workspace=_magazyn_target(),
                attachment_ids=("att-1",),
            ),
        ),
    )
    from local_workspace_application.conversation.interaction_planner import (
        validate_plan_against_request,
    )

    validate_plan_against_request(plan, request)
    workspace_refs = [
        action.workspace
        for action in plan.actions
        if hasattr(action, "workspace")
    ]
    assert all(ref.kind == WorkspaceReferenceKind.name and ref.value == "magazyn" for ref in workspace_refs)
    assert not any(action.action_type == "workspace.activate" for action in plan.actions)


@pytest.mark.unit
def test_workspace_target_does_not_imply_activate_action() -> None:
    plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            KnowledgeAddWebUrlsPlannedAction(
                action_id="a1",
                action_type="knowledge.add_web_urls",
                workspace=_magazyn_target(),
                urls=("https://example.com",),
            ),
        ),
    )
    assert all(action.action_type != "workspace.activate" for action in plan.actions)

    activate_plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        actions=(
            WorkspaceActivatePlannedAction(
                action_id="act-1",
                action_type="workspace.activate",
                workspace=_magazyn_target(),
                evidence_quotes=("przełącz mnie na workspace magazyn",),
            ),
        ),
    )
    assert activate_plan.actions[0].action_type == "workspace.activate"


@pytest.mark.unit
def test_valid_plan_with_clarification_only() -> None:
    plan = ConversationInteractionPlan(
        plan_version="1",
        response_mode="aggregate",
        clarifications=(
            ConversationClarification(
                clarification_id="c1",
                question="Which workspace did you mean?",
                blocks_action_ids=(),
            ),
        ),
    )
    assert plan.clarifications[0].clarification_id == "c1"
