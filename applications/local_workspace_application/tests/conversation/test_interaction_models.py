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
    ConversationPlanningSourceCandidate,
    ConversationPlanningWorkspace,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    LocalFileReferenceExtractedObject,
    MessageTextEvidenceSpan,
    WebUrlExtractedObject,
    WorkspaceActivatePlannedAction,
    WorkspaceCreatePlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)


def _magazyn_target() -> WorkspaceReference:
    return WorkspaceReference(kind=WorkspaceReferenceKind.name, value="magazyn")


def _span(message: str, substring: str) -> MessageTextEvidenceSpan:
    start = message.index(substring)
    return MessageTextEvidenceSpan(
        source="message_text",
        start=start,
        end=start + len(substring),
        text=substring,
    )


def _web_object(
    object_id: str,
    message: str,
    url: str,
) -> WebUrlExtractedObject:
    return WebUrlExtractedObject(
        object_id=object_id,
        object_type="web_url",
        value=url,
        evidence=_span(message, url),
    )


def _local_object(
    object_id: str,
    message: str,
    path: str,
    *,
    reference_kind: str = "file",
) -> LocalFileReferenceExtractedObject:
    return LocalFileReferenceExtractedObject(
        object_id=object_id,
        object_type="local_file_reference",
        reference_kind=reference_kind,  # type: ignore[arg-type]
        value=path,
        evidence=_span(message, path),
    )


@pytest.mark.unit
def test_valid_plan_with_objects_and_knowledge_add_sources() -> None:
    message = (
        "dołącz https://www.cenniki.pl oraz "
        r"c:\moje dokumenty\cenniki.xls do workspace magazyn"
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            _web_object("url-1", message, "https://www.cenniki.pl"),
            _local_object("local-1", message, r"c:\moje dokumenty\cenniki.xls"),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1", "local-1"),
            ),
        ),
    )
    assert len(plan.objects) == 2
    assert plan.actions[0].action_type == "knowledge.add_sources"


@pytest.mark.unit
def test_rejects_duplicate_object_id() -> None:
    message = "dodaj https://example.com do magazyn"
    obj = _web_object("dup", message, "https://example.com")
    with pytest.raises(ValidationError, match="duplicate object_id"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(obj, obj),
            actions=(
                KnowledgeAddSourcesPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_sources",
                    workspace=_magazyn_target(),
                    source_object_ids=("dup",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_duplicate_source_object_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate source_object_id"):
        KnowledgeAddSourcesPlannedAction(
            action_id="a1",
            action_type="knowledge.add_sources",
            workspace=_magazyn_target(),
            source_object_ids=("obj-1", "obj-1"),
        )


@pytest.mark.unit
def test_rejects_empty_source_object_ids() -> None:
    with pytest.raises(ValidationError):
        KnowledgeAddSourcesPlannedAction(
            action_id="a1",
            action_type="knowledge.add_sources",
            workspace=_magazyn_target(),
            source_object_ids=(),
        )


@pytest.mark.unit
def test_rejects_invalid_object_type() -> None:
    with pytest.raises(ValidationError):
        ConversationInteractionPlan.model_validate(
            {
                "plan_version": "2",
                "response_mode": "aggregate",
                "objects": [
                    {
                        "object_id": "x1",
                        "object_type": "attachment",
                        "value": "https://example.com",
                        "evidence": {
                            "source": "message_text",
                            "start": 0,
                            "end": 19,
                            "text": "https://example.com",
                        },
                    }
                ],
                "actions": [],
                "clarifications": [
                    {
                        "clarification_id": "c1",
                        "question": "Which workspace?",
                    }
                ],
            }
        )


@pytest.mark.unit
def test_rejects_end_lte_start() -> None:
    with pytest.raises(ValidationError, match="evidence end must be > start"):
        MessageTextEvidenceSpan(
            source="message_text",
            start=5,
            end=5,
            text="x",
        )


@pytest.mark.unit
def test_rejects_negative_start() -> None:
    with pytest.raises(ValidationError, match="evidence start must be >= 0"):
        MessageTextEvidenceSpan(
            source="message_text",
            start=-1,
            end=3,
            text="abc",
        )


@pytest.mark.unit
def test_rejects_value_with_nul() -> None:
    with pytest.raises(ValidationError):
        WebUrlExtractedObject(
            object_id="url-1",
            object_type="web_url",
            value="https://exa\x00mple.com",
            evidence=MessageTextEvidenceSpan(
                source="message_text",
                start=0,
                end=5,
                text="https",
            ),
        )


@pytest.mark.unit
def test_preserves_spaces_slashes_and_case_without_normalization() -> None:
    path = r"C:\Folder With Spaces\File.XLSX"
    message = f"dodaj {path} do magazyn"
    obj = _local_object("local-1", message, path)
    assert obj.value == path
    assert obj.evidence.text == path
    assert obj.value == obj.evidence.text


@pytest.mark.unit
def test_valid_plan_with_multiple_actions_and_different_object_groups() -> None:
    message = (
        "ten adres https://cenniki.pl wrzuć do workspace numer 1, "
        r"a pliki C:\cenniki\hurt.xlsx i C:\cenniki\detal.xlsx dodaj do workspace numer 2"
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            _web_object("url-1", message, "https://cenniki.pl"),
            _local_object("local-1", message, r"C:\cenniki\hurt.xlsx"),
            _local_object("local-2", message, r"C:\cenniki\detal.xlsx"),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="1"),
                source_object_ids=("url-1",),
            ),
            KnowledgeAddSourcesPlannedAction(
                action_id="a2",
                action_type="knowledge.add_sources",
                workspace=WorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value="2"),
                source_object_ids=("local-1", "local-2"),
            ),
        ),
    )
    assert len(plan.actions) == 2
    assert plan.plan_version == "2"


@pytest.mark.unit
def test_v2_plan_structured_output_roundtrip() -> None:
    message = "dodaj https://example.com do magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(_web_object("url-1", message, "https://example.com"),),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    payload = plan.model_dump(mode="json")
    restored = ConversationInteractionPlan.model_validate(payload)
    assert restored.plan_version == "2"
    assert restored.objects[0].object_type == "web_url"
    assert restored.actions[0].action_type == "knowledge.add_sources"


@pytest.mark.unit
def test_rejects_unknown_action_type() -> None:
    with pytest.raises(ValidationError):
        ConversationInteractionPlan.model_validate(
            {
                "plan_version": "2",
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
    message = "dodaj https://example.com do magazyn"
    action = KnowledgeAddSourcesPlannedAction(
        action_id="dup",
        action_type="knowledge.add_sources",
        workspace=_magazyn_target(),
        source_object_ids=("url-1",),
    )
    with pytest.raises(ValidationError, match="duplicate action_id"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(_web_object("url-1", message, "https://example.com"),),
            actions=(action, action),
        )


@pytest.mark.unit
def test_rejects_dependency_to_missing_action() -> None:
    message = "dodaj https://example.com do magazyn"
    with pytest.raises(ValidationError, match="unknown dependency"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(_web_object("url-1", message, "https://example.com"),),
            actions=(
                KnowledgeAddSourcesPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_sources",
                    workspace=_magazyn_target(),
                    source_object_ids=("url-1",),
                    depends_on=("missing",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_self_dependency() -> None:
    message = "dodaj https://example.com do magazyn"
    with pytest.raises(ValidationError, match="self dependency"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(_web_object("url-1", message, "https://example.com"),),
            actions=(
                KnowledgeAddSourcesPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_sources",
                    workspace=_magazyn_target(),
                    source_object_ids=("url-1",),
                    depends_on=("a1",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_dependency_cycle() -> None:
    message = "dodaj https://a.example i https://b.example do magazyn"
    with pytest.raises(ValidationError, match="dependency cycle"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(
                _web_object("url-1", message, "https://a.example"),
                _web_object("url-2", message, "https://b.example"),
            ),
            actions=(
                KnowledgeAddSourcesPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_sources",
                    workspace=_magazyn_target(),
                    source_object_ids=("url-1",),
                    depends_on=("a2",),
                ),
                KnowledgeAddSourcesPlannedAction(
                    action_id="a2",
                    action_type="knowledge.add_sources",
                    workspace=_magazyn_target(),
                    source_object_ids=("url-2",),
                    depends_on=("a1",),
                ),
            ),
        )


@pytest.mark.unit
def test_created_by_action_must_reference_workspace_create() -> None:
    message = "dodaj https://example.com do magazyn"
    with pytest.raises(ValidationError, match="created_by_action must reference workspace.create"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(_web_object("url-1", message, "https://example.com"),),
            actions=(
                KnowledgeAddSourcesPlannedAction(
                    action_id="not-create",
                    action_type="knowledge.add_sources",
                    workspace=_magazyn_target(),
                    source_object_ids=("url-1",),
                ),
                KnowledgeAddSourcesPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_sources",
                    workspace=WorkspaceReference(
                        kind=WorkspaceReferenceKind.created_by_action,
                        value="not-create",
                    ),
                    source_object_ids=("url-1",),
                    depends_on=("not-create",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_created_by_action_without_required_dependency() -> None:
    message = "dodaj https://example.com do magazyn"
    with pytest.raises(ValidationError, match="must depend on the workspace.create action"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            objects=(_web_object("url-1", message, "https://example.com"),),
            actions=(
                WorkspaceCreatePlannedAction(
                    action_id="create-1",
                    action_type="workspace.create",
                    name="Nowy",
                ),
                KnowledgeAddSourcesPlannedAction(
                    action_id="a1",
                    action_type="knowledge.add_sources",
                    workspace=WorkspaceReference(
                        kind=WorkspaceReferenceKind.created_by_action,
                        value="create-1",
                    ),
                    source_object_ids=("url-1",),
                ),
            ),
        )


@pytest.mark.unit
def test_rejects_empty_plan_without_clarification() -> None:
    with pytest.raises(ValidationError, match="at least one action or clarification"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
        )


@pytest.mark.unit
def test_valid_mixed_source_plan_with_shared_workspace_target() -> None:
    message = (
        "dołącz informacje o cennikach ze strony https://www.cenniki.pl "
        "oraz dorzuć moją kopię lokalną cenników z "
        r"c:\moje dokumenty\cenniki.xls "
        'a to wszystko do workspace "magazyn"'
    )
    request = ConversationPlanningRequest(
        message_text=message,
        attachments=(
            ConversationPlanningAttachment(attachment_id="att-1", file_name="extra.pdf"),
        ),
    )
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(
            _web_object("url-1", message, "https://www.cenniki.pl"),
            _local_object("local-1", message, r"c:\moje dokumenty\cenniki.xls"),
        ),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="sources-1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1", "local-1"),
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
        action.workspace for action in plan.actions if hasattr(action, "workspace")
    ]
    assert all(ref.kind == WorkspaceReferenceKind.name and ref.value == "magazyn" for ref in workspace_refs)
    assert not any(action.action_type == "workspace.activate" for action in plan.actions)


@pytest.mark.unit
def test_workspace_target_does_not_imply_activate_action() -> None:
    message = "dodaj https://example.com do workspace magazyn"
    plan = ConversationInteractionPlan(
        plan_version="2",
        response_mode="aggregate",
        objects=(_web_object("url-1", message, "https://example.com"),),
        actions=(
            KnowledgeAddSourcesPlannedAction(
                action_id="a1",
                action_type="knowledge.add_sources",
                workspace=_magazyn_target(),
                source_object_ids=("url-1",),
            ),
        ),
    )
    assert all(action.action_type != "workspace.activate" for action in plan.actions)

    activate_plan = ConversationInteractionPlan(
        plan_version="2",
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
        plan_version="2",
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


@pytest.mark.unit
def test_rejects_whitespace_only_attachment_id() -> None:
    with pytest.raises(ValidationError):
        ConversationPlanningAttachment(attachment_id=" ", file_name="doc.pdf")


@pytest.mark.unit
def test_rejects_tab_workspace_id() -> None:
    with pytest.raises(ValidationError):
        ConversationPlanningWorkspace(workspace_id="\t", name="default", is_active=True)


@pytest.mark.unit
def test_rejects_nul_candidate_id() -> None:
    with pytest.raises(ValidationError):
        ConversationPlanningSourceCandidate(
            candidate_id="\x00candidate",
            label="Contracts",
            source_type="local_folder",
            available=True,
        )


@pytest.mark.unit
def test_rejects_newline_action_id() -> None:
    with pytest.raises(ValidationError):
        KnowledgeAddSourcesPlannedAction(
            action_id="\n",
            action_type="knowledge.add_sources",
            workspace=_magazyn_target(),
            source_object_ids=("url-1",),
        )


@pytest.mark.unit
def test_rejects_bool_size_bytes() -> None:
    with pytest.raises(ValidationError):
        ConversationPlanningAttachment(attachment_id="att-1", size_bytes=True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_rejects_duplicate_clarification_id() -> None:
    clarification = ConversationClarification(
        clarification_id="c-dup",
        question="Which workspace?",
    )
    with pytest.raises(ValidationError, match="duplicate clarification_id"):
        ConversationInteractionPlan(
            plan_version="2",
            response_mode="aggregate",
            clarifications=(clarification, clarification),
        )


@pytest.mark.unit
def test_rejects_duplicate_attachment_ids_in_action() -> None:
    with pytest.raises(ValidationError, match="duplicate attachment_id"):
        KnowledgeAddAttachmentsPlannedAction(
            action_id="att-action",
            action_type="knowledge.add_attachments",
            workspace=_magazyn_target(),
            attachment_ids=("att-1", "att-1"),
        )


@pytest.mark.unit
def test_rejects_duplicate_dependency() -> None:
    with pytest.raises(ValidationError, match="duplicate depends_on entry"):
        KnowledgeAddSourcesPlannedAction(
            action_id="a2",
            action_type="knowledge.add_sources",
            workspace=_magazyn_target(),
            source_object_ids=("url-1",),
            depends_on=("a1", "a1"),
        )


@pytest.mark.unit
def test_rejects_invalid_candidate_ordinal_reference() -> None:
    from local_workspace_application.conversation.interaction_models import (
        SourceCandidateAttachPlannedAction,
    )

    for invalid in ("0", "-1", "abc", "1.5"):
        with pytest.raises(ValidationError):
            SourceCandidateAttachPlannedAction(
                action_id="cand-1",
                action_type="source_candidate.attach",
                workspace=_magazyn_target(),
                candidate_reference_kind="ordinal",
                candidate_reference=invalid,
            )


@pytest.mark.unit
def test_accepts_valid_candidate_ordinal_reference() -> None:
    from local_workspace_application.conversation.interaction_models import (
        SourceCandidateAttachPlannedAction,
    )

    for valid in ("1", "12"):
        action = SourceCandidateAttachPlannedAction(
            action_id=f"cand-{valid}",
            action_type="source_candidate.attach",
            workspace=_magazyn_target(),
            candidate_reference_kind="ordinal",
            candidate_reference=valid,
        )
        assert action.candidate_reference == valid
