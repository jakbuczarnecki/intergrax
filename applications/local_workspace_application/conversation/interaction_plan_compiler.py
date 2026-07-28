# © Artur Czarnecki. All rights reserved.

"""Deterministic compilation of semantic interaction drafts into canonical plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from local_workspace_application.conversation.interaction_draft_models import (
    ConversationClarificationDraft,
    ConversationInteractionDraft,
    DraftLocalFileReferenceSource,
    DraftPlannedAction,
    DraftSource,
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
    ConversationClarification,
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    ExtractedObject,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    LocalFileReferenceExtractedObject,
    MessageTextEvidenceSpan,
    PlannedAction,
    SourceCandidateAttachPlannedAction,
    SourceCandidateListPlannedAction,
    SourceListPlannedAction,
    WebUrlExtractedObject,
    WorkspaceActivatePlannedAction,
    WorkspaceAskPlannedAction,
    WorkspaceCreatePlannedAction,
    WorkspaceDeletePlannedAction,
    WorkspaceListPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)


class ConversationDraftCompilationErrorCode(str, Enum):
    source_value_not_found = "source_value_not_found"
    source_occurrence_required = "source_occurrence_required"
    source_occurrence_out_of_range = "source_occurrence_out_of_range"
    invalid_action_reference = "invalid_action_reference"
    self_action_reference = "self_action_reference"
    invalid_created_workspace_reference = "invalid_created_workspace_reference"
    ambiguous_created_workspace_reference = "ambiguous_created_workspace_reference"
    conflicting_source_declaration = "conflicting_source_declaration"


class ConversationDraftCompilationError(ValueError):
    """Stable compilation failure without user content or model output."""

    def __init__(self, code: ConversationDraftCompilationErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


class _PositionKey(NamedTuple):
    start: int
    end: int
    value: str


class _DeclaredSource(NamedTuple):
    object_type: str
    reference_kind: str | None


@dataclass(frozen=True)
class _GroundedSource:
    object_id: str
    canonical: ExtractedObject


def compile_interaction_draft(
    draft: ConversationInteractionDraft,
    request: ConversationPlanningRequest,
) -> ConversationInteractionPlan:
    """Compile a semantic draft into the canonical interaction plan v2 contract."""
    action_count = len(draft.actions)
    action_ids = tuple(f"action-{index}" for index in range(1, action_count + 1))
    action_id_by_number = {index: action_ids[index - 1] for index in range(1, action_count + 1)}

    create_name_to_action_id: dict[str, str] = {}
    for index, action in enumerate(draft.actions, start=1):
        if action.action_type != "workspace.create":
            continue
        assert isinstance(action, WorkspaceCreateDraftAction)
        name = action.name
        if name in create_name_to_action_id:
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.ambiguous_created_workspace_reference
            )
        create_name_to_action_id[name] = action_ids[index - 1]

    grounded_by_object_id: dict[str, _GroundedSource] = {}
    grounded_by_position: dict[_PositionKey, _GroundedSource] = {}
    position_declarations: dict[_PositionKey, _DeclaredSource] = {}
    object_order: list[str] = []
    next_object_number = 1

    def ground_source(source: DraftSource) -> str:
        nonlocal next_object_number
        span = _ground_source_span(source, request.message_text)
        reference_kind = source.reference_kind if isinstance(source, DraftLocalFileReferenceSource) else None
        position = _PositionKey(start=span.start, end=span.end, value=source.value)
        declaration = _DeclaredSource(object_type=source.object_type, reference_kind=reference_kind)
        prior_declaration = position_declarations.get(position)
        if prior_declaration is not None:
            if prior_declaration.object_type != declaration.object_type:
                raise ConversationDraftCompilationError(
                    ConversationDraftCompilationErrorCode.conflicting_source_declaration
                )
            if (
                declaration.object_type == "local_file_reference"
                and prior_declaration.reference_kind != declaration.reference_kind
            ):
                raise ConversationDraftCompilationError(
                    ConversationDraftCompilationErrorCode.conflicting_source_declaration
                )
        else:
            position_declarations[position] = declaration

        existing = grounded_by_position.get(position)
        if existing is not None:
            return existing.object_id

        object_id = f"object-{next_object_number}"
        next_object_number += 1
        if isinstance(source, DraftWebUrlSource):
            canonical: ExtractedObject = WebUrlExtractedObject(
                object_id=object_id,
                object_type="web_url",
                value=source.value,
                evidence=span,
            )
        else:
            canonical = LocalFileReferenceExtractedObject(
                object_id=object_id,
                object_type="local_file_reference",
                reference_kind=source.reference_kind,
                value=source.value,
                evidence=span,
            )
        grounded = _GroundedSource(object_id=object_id, canonical=canonical)
        grounded_by_position[position] = grounded
        grounded_by_object_id[object_id] = grounded
        object_order.append(object_id)
        return object_id

    source_object_ids_by_action: list[tuple[str, ...]] = []
    for action in draft.actions:
        if action.action_type != "knowledge.add_sources":
            source_object_ids_by_action.append(())
            continue
        assert isinstance(action, KnowledgeAddSourcesDraftAction)
        ids: list[str] = []
        for source in action.sources:
            ids.append(ground_source(source))
        source_object_ids_by_action.append(tuple(ids))

    objects = tuple(grounded_by_object_id[object_id].canonical for object_id in object_order)

    compiled_actions: list[PlannedAction] = []
    for index, (action, action_id) in enumerate(zip(draft.actions, action_ids, strict=True), start=1):
        depends_on = _map_action_numbers(
            action.depends_on_action_numbers,
            action_id_by_number=action_id_by_number,
            self_action_number=index,
        )
        workspace_ref, extra_depends = _compile_workspace_reference(
            _extract_draft_workspace(action),
            create_name_to_action_id=create_name_to_action_id,
        )
        depends_on = _merge_depends_on(depends_on, extra_depends)
        compiled = _compile_action(
            action,
            action_id=action_id,
            depends_on=depends_on,
            workspace=workspace_ref,
            source_object_ids=source_object_ids_by_action[index - 1],
        )
        compiled_actions.append(compiled)

    clarifications = tuple(
        _compile_clarification(item, index=clarification_index, action_id_by_number=action_id_by_number)
        for clarification_index, item in enumerate(draft.clarifications, start=1)
    )

    return ConversationInteractionPlan(
        plan_version="2",
        objects=objects,
        actions=tuple(compiled_actions),
        clarifications=clarifications,
        response_mode="aggregate",
    )


def _find_occurrence_starts(message_text: str, value: str) -> list[int]:
    if not value:
        return []
    starts: list[int] = []
    start = 0
    while start <= len(message_text) - len(value):
        index = message_text.find(value, start)
        if index < 0:
            break
        starts.append(index)
        start = index + 1
    return starts


def _ground_source_span(source: DraftSource, message_text: str) -> MessageTextEvidenceSpan:
    starts = _find_occurrence_starts(message_text, source.value)
    if not starts:
        raise ConversationDraftCompilationError(
            ConversationDraftCompilationErrorCode.source_value_not_found
        )
    if len(starts) == 1:
        if source.occurrence is not None and source.occurrence != 1:
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.source_occurrence_out_of_range
            )
        selected_start = starts[0]
    else:
        if source.occurrence is None:
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.source_occurrence_required
            )
        if source.occurrence < 1 or source.occurrence > len(starts):
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.source_occurrence_out_of_range
            )
        selected_start = starts[source.occurrence - 1]
    end = selected_start + len(source.value)
    text = message_text[selected_start:end]
    assert text == source.value
    return MessageTextEvidenceSpan(
        source="message_text",
        start=selected_start,
        end=end,
        text=text,
    )


def _map_action_numbers(
    numbers: tuple[int, ...],
    *,
    action_id_by_number: dict[int, str],
    self_action_number: int,
) -> tuple[str, ...]:
    mapped: list[str] = []
    for number in numbers:
        if number == self_action_number:
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.self_action_reference
            )
        action_id = action_id_by_number.get(number)
        if action_id is None:
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.invalid_action_reference
            )
        mapped.append(action_id)
    return tuple(mapped)


def _merge_depends_on(
    existing: tuple[str, ...],
    extra: tuple[str, ...],
) -> tuple[str, ...]:
    if not extra:
        return existing
    merged: list[str] = list(existing)
    seen = set(existing)
    for dep in extra:
        if dep not in seen:
            merged.append(dep)
            seen.add(dep)
    return tuple(merged)


def _extract_draft_workspace(action: DraftPlannedAction) -> DraftWorkspaceReference | None:
    workspace = getattr(action, "workspace", None)
    if isinstance(workspace, DraftWorkspaceReference):
        return workspace
    return None


def _compile_workspace_reference(
    workspace: DraftWorkspaceReference | None,
    *,
    create_name_to_action_id: dict[str, str],
) -> tuple[WorkspaceReference | None, tuple[str, ...]]:
    if workspace is None:
        return None, ()
    if workspace.kind != WorkspaceReferenceKind.created_by_action:
        return (
            WorkspaceReference(kind=workspace.kind, value=workspace.value),
            (),
        )
    assert workspace.value is not None
    create_action_id = create_name_to_action_id.get(workspace.value)
    if create_action_id is None:
        raise ConversationDraftCompilationError(
            ConversationDraftCompilationErrorCode.invalid_created_workspace_reference
        )
    return (
        WorkspaceReference(
            kind=WorkspaceReferenceKind.created_by_action,
            value=create_action_id,
        ),
        (create_action_id,),
    )


def _compile_action(
    action: DraftPlannedAction,
    *,
    action_id: str,
    depends_on: tuple[str, ...],
    workspace: WorkspaceReference | None,
    source_object_ids: tuple[str, ...],
) -> PlannedAction:
    common = {
        "action_id": action_id,
        "depends_on": depends_on,
        "evidence_quotes": action.evidence_quotes,
        "evidence_attachment_ids": action.evidence_attachment_ids,
    }
    if isinstance(action, WorkspaceListDraftAction):
        return WorkspaceListPlannedAction(action_type="workspace.list", **common)
    if isinstance(action, WorkspaceCreateDraftAction):
        return WorkspaceCreatePlannedAction(
            action_type="workspace.create",
            name=action.name,
            **common,
        )
    if isinstance(action, WorkspaceActivateDraftAction):
        assert workspace is not None
        return WorkspaceActivatePlannedAction(
            action_type="workspace.activate",
            workspace=workspace,
            **common,
        )
    if isinstance(action, WorkspaceDeleteDraftAction):
        assert workspace is not None
        return WorkspaceDeletePlannedAction(
            action_type="workspace.delete",
            workspace=workspace,
            **common,
        )
    if isinstance(action, SourceListDraftAction):
        assert workspace is not None
        return SourceListPlannedAction(action_type="source.list", workspace=workspace, **common)
    if isinstance(action, SourceCandidateListDraftAction):
        assert workspace is not None
        return SourceCandidateListPlannedAction(
            action_type="source_candidate.list",
            workspace=workspace,
            **common,
        )
    if isinstance(action, SourceCandidateAttachDraftAction):
        assert workspace is not None
        return SourceCandidateAttachPlannedAction(
            action_type="source_candidate.attach",
            workspace=workspace,
            candidate_reference_kind=action.candidate_reference_kind,
            candidate_reference=action.candidate_reference,
            **common,
        )
    if isinstance(action, KnowledgeAddAttachmentsDraftAction):
        assert workspace is not None
        return KnowledgeAddAttachmentsPlannedAction(
            action_type="knowledge.add_attachments",
            workspace=workspace,
            attachment_ids=action.attachment_ids,
            **common,
        )
    if isinstance(action, KnowledgeAddSourcesDraftAction):
        assert workspace is not None
        return KnowledgeAddSourcesPlannedAction(
            action_type="knowledge.add_sources",
            workspace=workspace,
            source_object_ids=source_object_ids,
            **common,
        )
    if isinstance(action, WorkspaceAskDraftAction):
        assert workspace is not None
        return WorkspaceAskPlannedAction(
            action_type="workspace.ask",
            workspace=workspace,
            question=action.question,
            **common,
        )
    raise TypeError("unsupported draft action type")


def _compile_clarification(
    clarification: ConversationClarificationDraft,
    *,
    index: int,
    action_id_by_number: dict[int, str],
) -> ConversationClarification:
    blocked_ids: list[str] = []
    for number in clarification.blocks_action_numbers:
        action_id = action_id_by_number.get(number)
        if action_id is None:
            raise ConversationDraftCompilationError(
                ConversationDraftCompilationErrorCode.invalid_action_reference
            )
        blocked_ids.append(action_id)
    return ConversationClarification(
        clarification_id=f"clarification-{index}",
        question=clarification.question,
        blocks_action_ids=tuple(blocked_ids),
    )
