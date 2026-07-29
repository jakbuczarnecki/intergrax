# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    SemanticFailureCategory,
)
from local_workspace_application.benchmarks.local_model_qualification.corpus import (
    case_by_id,
)
from local_workspace_application.benchmarks.local_model_qualification.evaluator import evaluate_semantics
from local_workspace_application.conversation.interaction_draft_models import (
    ConversationClarificationDraft,
    ConversationInteractionDraft,
    DraftWebUrlSource,
    KnowledgeAddSourcesDraftAction,
    NameDraftWorkspaceReference,
    WorkspaceActivateDraftAction,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationClarification,
    ConversationInteractionPlan,
    KnowledgeAddSourcesPlannedAction,
    WebUrlExtractedObject,
    WorkspaceActivatePlannedAction,
    WorkspaceDeletePlannedAction,
    WorkspaceListPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
    MessageTextEvidenceSpan,
)
from local_workspace_application.conversation.interaction_plan_compiler import compile_interaction_draft


def _compile_case(case_id: str) -> ConversationInteractionPlan:
    case = case_by_id(case_id)
    draft = _reference_draft(case_id)
    return compile_interaction_draft(draft, case.request)


def _reference_draft(case_id: str) -> ConversationInteractionDraft:
    if case_id == "planner.workspace_list":
        return ConversationInteractionDraft(
            actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
        )
    if case_id == "planner.target_workspace_without_activation":
        return ConversationInteractionDraft(
            actions=(
                KnowledgeAddSourcesDraftAction(
                    action_type="knowledge.add_sources",
                    workspace=NameDraftWorkspaceReference(
                        kind=WorkspaceReferenceKind.name,
                        value="magazyn",
                    ),
                    sources=(
                        DraftWebUrlSource(
                            object_type="web_url",
                            value="https://example.com/docs",
                        ),
                    ),
                ),
            )
        )
    if case_id == "planner.explicit_workspace_activation":
        return ConversationInteractionDraft(
            actions=(
                WorkspaceActivateDraftAction(
                    action_type="workspace.activate",
                    workspace=NameDraftWorkspaceReference(
                        kind=WorkspaceReferenceKind.name,
                        value="magazyn",
                    ),
                ),
            )
        )
    if case_id == "planner.ambiguous_missing_workspace_target":
        return ConversationInteractionDraft(
            clarifications=(
                ConversationClarificationDraft(
                    question="Which workspace should receive the source?",
                ),
            )
        )
    raise KeyError(case_id)


def test_all_reference_success_cases_pass() -> None:
    for case_id in (
        "planner.workspace_list",
        "planner.target_workspace_without_activation",
        "planner.explicit_workspace_activation",
    ):
        case = case_by_id(case_id)
        plan = _compile_case(case_id)
        result = evaluate_semantics(plan, case)
        assert result.passed, case_id


def test_unnecessary_workspace_activate_detected() -> None:
    case = case_by_id("planner.target_workspace_without_activation")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                source_object_ids=("o1",),
            ),
            WorkspaceActivatePlannedAction(
                action_id="a2",
                action_type="workspace.activate",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
            ),
        ),
        objects=(
            WebUrlExtractedObject(
                object_id="o1",
                object_type="web_url",
                value="https://example.com/docs",
                evidence=MessageTextEvidenceSpan(
                    source="message_text",
                    start=5,
                    end=30,
                    text="https://example.com/docs",
                ),
            ),
        ),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert SemanticFailureCategory.UNNECESSARY_WORKSPACE_ACTIVATE.value in result.failure_categories


def test_unexpected_workspace_delete_detected() -> None:
    case = case_by_id("planner.workspace_list")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceListPlannedAction(action_id="a1", action_type="workspace.list"),
            WorkspaceDeletePlannedAction(
                action_id="a2",
                action_type="workspace.delete",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.name, value="archiwum"),
            ),
        ),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert SemanticFailureCategory.UNEXPECTED_STATE_CHANGE.value in result.failure_categories


def test_wrong_workspace_kind_detected() -> None:
    case = case_by_id("planner.active_workspace_source_add")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.name, value="finanse"),
                source_object_ids=("o1",),
            ),
        ),
        objects=(
            WebUrlExtractedObject(
                object_id="o1",
                object_type="web_url",
                value="https://example.com/guide",
                evidence=MessageTextEvidenceSpan(
                    source="message_text",
                    start=5,
                    end=30,
                    text="https://example.com/guide",
                ),
            ),
        ),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert SemanticFailureCategory.WRONG_WORKSPACE_REFERENCE.value in result.failure_categories


def test_missing_clarification_detected() -> None:
    case = case_by_id("planner.ambiguous_missing_workspace_target")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(WorkspaceListPlannedAction(action_id="a1", action_type="workspace.list"),),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert SemanticFailureCategory.MISSING_CLARIFICATION.value in result.failure_categories


def test_unnecessary_clarification_detected() -> None:
    case = case_by_id("planner.workspace_list")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(WorkspaceListPlannedAction(action_id="a1", action_type="workspace.list"),),
        clarifications=(
            ConversationClarification(
                clarification_id="c1",
                question="Which workspace?",
            ),
        ),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert SemanticFailureCategory.UNNECESSARY_CLARIFICATION.value in result.failure_categories


def test_failure_precedence() -> None:
    case = case_by_id("planner.workspace_list")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceListPlannedAction(action_id="a1", action_type="workspace.list"),
            WorkspaceDeletePlannedAction(
                action_id="a2",
                action_type="workspace.delete",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.name, value="archiwum"),
            ),
        ),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert result.primary_failure_category == SemanticFailureCategory.UNEXPECTED_STATE_CHANGE.value


def test_unsafe_state_change_count() -> None:
    case = case_by_id("planner.workspace_list")
    plan = ConversationInteractionPlan(
        plan_version="2",
        actions=(
            WorkspaceDeletePlannedAction(
                action_id="a1",
                action_type="workspace.delete",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.name, value="archiwum"),
            ),
        ),
        response_mode="aggregate",
    )
    result = evaluate_semantics(plan, case)
    assert result.unsafe_state_change_count == 1
