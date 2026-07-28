# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for the semantic draft → canonical plan compiler."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_workspace_application.conversation.interaction_draft_models import (
    ConversationClarificationDraft,
    ConversationInteractionDraft,
    DraftLocalFileReferenceSource,
    DraftWebUrlSource,
    DraftWorkspaceReference,
    KnowledgeAddAttachmentsDraftAction,
    KnowledgeAddSourcesDraftAction,
    SourceCandidateAttachDraftAction,
    SourceCandidateListDraftAction,
    SourceListDraftAction,
    WorkspaceActivateDraftAction,
    WorkspaceAskDraftAction,
    WorkspaceCreateDraftAction,
    WorkspaceDeleteDraftAction,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningRequest,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_plan_compiler import (
    ConversationDraftCompilationError,
    ConversationDraftCompilationErrorCode,
    compile_interaction_draft,
)


def _request(message: str) -> ConversationPlanningRequest:
    return ConversationPlanningRequest(message_text=message)


@pytest.mark.unit
def test_one_named_workspace_web_source() -> None:
    message = 'dodaj https://portal.vendor.io do workspace "projekty"'
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="projekty"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://portal.vendor.io"),),
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request(message))
    assert len(plan.objects) == 1
    assert len(plan.actions) == 1
    assert plan.actions[0].action_id == "action-1"
    assert plan.objects[0].object_id == "object-1"
    assert plan.actions[0].source_object_ids == ("object-1",)  # type: ignore[attr-defined]
    start = message.index("https://portal.vendor.io")
    assert plan.objects[0].evidence.start == start
    assert plan.objects[0].evidence.end == start + len("https://portal.vendor.io")
    assert plan.objects[0].evidence.text == "https://portal.vendor.io"


@pytest.mark.unit
def test_mixed_routing() -> None:
    message = (
        "link https://docs.vendor.io wrzuć do workspace numer 1, "
        r"a pliki D:\share\alpha.txt i D:\share\beta.txt dodaj do workspace numer 2"
    )
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://docs.vendor.io"),),
            ),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                sources=(
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value=r"D:\share\alpha.txt",
                    ),
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value=r"D:\share\beta.txt",
                    ),
                ),
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request(message))
    assert len(plan.objects) == 3
    assert len(plan.actions) == 2
    assert plan.actions[0].workspace.value == "1"  # type: ignore[attr-defined]
    assert plan.actions[1].workspace.value == "2"  # type: ignore[attr-defined]
    assert all(action.action_type != "workspace.activate" for action in plan.actions)


@pytest.mark.unit
def test_windows_path_grounding() -> None:
    message = r"dodaj C:\my files\report v2.xlsx do magazyn"
    path = r"C:\my files\report v2.xlsx"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value=path,
                    ),
                ),
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request(message))
    obj = plan.objects[0]
    start = message.index(path)
    assert obj.value == path
    assert obj.evidence.start == start
    assert obj.evidence.end == start + len(path)
    assert obj.evidence.text == path


@pytest.mark.unit
def test_source_not_found() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="x"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://missing.example"),),
            ),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request("no urls here"))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.source_value_not_found


@pytest.mark.unit
def test_repeated_value_without_occurrence() -> None:
    message = "add https://dup.example and again https://dup.example"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="x"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://dup.example"),),
            ),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request(message))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.source_occurrence_required


@pytest.mark.unit
def test_repeated_value_with_occurrence() -> None:
    message = "add https://dup.example and again https://dup.example"
    value = "https://dup.example"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="x"),
                sources=(DraftWebUrlSource(object_type="web_url", value=value, occurrence=2),),
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request(message))
    first = message.index(value)
    second = message.index(value, first + 1)
    assert plan.objects[0].evidence.start == second


@pytest.mark.unit
def test_occurrence_out_of_range() -> None:
    message = "add https://dup.example and again https://dup.example"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="x"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://dup.example", occurrence=3),),
            ),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request(message))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.source_occurrence_out_of_range


@pytest.mark.unit
def test_same_source_reused_by_two_actions() -> None:
    message = "dodaj https://shared.vendor.io do magazyn"
    source = DraftWebUrlSource(object_type="web_url", value="https://shared.vendor.io")
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(source,),
            ),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                sources=(source,),
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request(message))
    assert len(plan.objects) == 1
    assert plan.actions[0].source_object_ids == ("object-1",)  # type: ignore[attr-defined]
    assert plan.actions[1].source_object_ids == ("object-1",)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_conflicting_source_declaration() -> None:
    message = "dodaj https://shared.vendor.io do magazyn"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://shared.vendor.io"),),
            ),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                sources=(
                    DraftLocalFileReferenceSource(
                        object_type="local_file_reference",
                        reference_kind="file",
                        value="https://shared.vendor.io",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request(message))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.conflicting_source_declaration


@pytest.mark.unit
def test_stable_ids() -> None:
    message = "dodaj https://stable.vendor.io do magazyn"
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://stable.vendor.io"),),
            ),
        ),
    )
    request = _request(message)
    first = compile_interaction_draft(draft, request)
    second = compile_interaction_draft(draft, request)
    assert first == second


@pytest.mark.unit
def test_action_dependencies() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceCreateDraftAction(action_type="workspace.create", name="alpha"),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.created_by_action, value="alpha"),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://dep.vendor.io"),),
                depends_on_action_numbers=(1,),
            ),
        ),
    )
    message = "create alpha and add https://dep.vendor.io"
    plan = compile_interaction_draft(draft, _request(message))
    assert plan.actions[1].depends_on == ("action-1",)


@pytest.mark.unit
def test_invalid_action_number() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceListDraftAction(action_type="workspace.list", depends_on_action_numbers=(2,)),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request("list workspaces"))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.invalid_action_reference


@pytest.mark.unit
def test_self_dependency() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceListDraftAction(action_type="workspace.list", depends_on_action_numbers=(1,)),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request("list workspaces"))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.self_action_reference


@pytest.mark.unit
def test_created_workspace_reference() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceCreateDraftAction(action_type="workspace.create", name="research"),
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(
                    kind=WorkspaceReferenceKind.created_by_action,
                    value="research",
                ),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://research.vendor.io"),),
            ),
        ),
    )
    message = "create research and add https://research.vendor.io"
    plan = compile_interaction_draft(draft, _request(message))
    assert plan.actions[1].workspace.kind == WorkspaceReferenceKind.created_by_action  # type: ignore[attr-defined]
    assert plan.actions[1].workspace.value == "action-1"  # type: ignore[attr-defined]
    assert "action-1" in plan.actions[1].depends_on


@pytest.mark.unit
def test_missing_created_workspace() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(
                    kind=WorkspaceReferenceKind.created_by_action,
                    value="missing",
                ),
                sources=(DraftWebUrlSource(object_type="web_url", value="https://x.example"),),
            ),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request("add https://x.example"))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.invalid_created_workspace_reference


@pytest.mark.unit
def test_duplicate_created_workspace_names() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceCreateDraftAction(action_type="workspace.create", name="dup"),
            WorkspaceCreateDraftAction(action_type="workspace.create", name="dup"),
        ),
    )
    with pytest.raises(ConversationDraftCompilationError) as exc_info:
        compile_interaction_draft(draft, _request("create dup twice"))
    assert exc_info.value.code == ConversationDraftCompilationErrorCode.ambiguous_created_workspace_reference


@pytest.mark.unit
def test_clarification_mapping() -> None:
    draft = ConversationInteractionDraft(
        actions=(WorkspaceListDraftAction(action_type="workspace.list"),),
        clarifications=(
            ConversationClarificationDraft(
                question="Which workspace should be used?",
                blocks_action_numbers=(1,),
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request("list workspaces"))
    assert plan.clarifications[0].clarification_id == "clarification-1"
    assert plan.clarifications[0].blocks_action_ids == ("action-1",)


@pytest.mark.unit
def test_every_non_source_action_variant() -> None:
    message = "operate on magazyn"
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceListDraftAction(action_type="workspace.list"),
            WorkspaceCreateDraftAction(action_type="workspace.create", name="new-ws"),
            WorkspaceActivateDraftAction(
                action_type="workspace.activate",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
            ),
            WorkspaceDeleteDraftAction(
                action_type="workspace.delete",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
            ),
            SourceListDraftAction(
                action_type="source.list",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
            ),
            SourceCandidateListDraftAction(
                action_type="source_candidate.list",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
            ),
            SourceCandidateAttachDraftAction(
                action_type="source_candidate.attach",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                candidate_reference_kind="name",
                candidate_reference="jira",
            ),
            KnowledgeAddAttachmentsDraftAction(
                action_type="knowledge.add_attachments",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn"),
                attachment_ids=("att-1",),
            ),
            WorkspaceAskDraftAction(
                action_type="workspace.ask",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.active),
                question="what is in magazyn?",
            ),
        ),
    )
    plan = compile_interaction_draft(draft, _request(message))
    assert [action.action_type for action in plan.actions] == [
        "workspace.list",
        "workspace.create",
        "workspace.activate",
        "workspace.delete",
        "source.list",
        "source_candidate.list",
        "source_candidate.attach",
        "knowledge.add_attachments",
        "workspace.ask",
    ]
    assert plan.actions[1].name == "new-ws"  # type: ignore[attr-defined]
    assert plan.actions[6].candidate_reference == "jira"  # type: ignore[attr-defined]
    assert plan.actions[7].attachment_ids == ("att-1",)  # type: ignore[attr-defined]
    assert plan.actions[8].question == "what is in magazyn?"  # type: ignore[attr-defined]


@pytest.mark.unit
def test_final_canonical_validation_remains_active() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            WorkspaceListDraftAction(action_type="workspace.list", depends_on_action_numbers=(2,)),
            WorkspaceListDraftAction(action_type="workspace.list", depends_on_action_numbers=(1,)),
        ),
    )
    with pytest.raises(ValidationError, match="dependency cycle"):
        compile_interaction_draft(draft, _request("list workspaces"))


@pytest.mark.unit
def test_no_deterministic_source_discovery() -> None:
    message = "nothing requested but https://hidden.vendor.io is present"
    draft = ConversationInteractionDraft(
        actions=(WorkspaceListDraftAction(action_type="workspace.list"),),
    )
    plan = compile_interaction_draft(draft, _request(message))
    assert plan.objects == ()
