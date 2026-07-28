# © Artur Czarnecki. All rights reserved.

"""Channel-neutral structured interaction plan models for LKW conversational planning."""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

_MAX_MESSAGE_TEXT_LEN = 16_000
_MAX_ATTACHMENTS = 50
_MAX_WORKSPACES = 100
_MAX_SOURCE_CANDIDATES = 100
_MAX_RECENT_TURNS = 20
_MAX_ACTIONS = 50
_MAX_CLARIFICATIONS = 20
_MAX_STRING_FIELD_LEN = 2_000
_MAX_ACTION_ID_LEN = 128
_MAX_EVIDENCE_QUOTES = 20
_MAX_OBJECTS = 50
_MAX_SOURCE_OBJECT_IDS_PER_ACTION = 50
_MAX_ATTACHMENT_IDS_PER_ACTION = 50

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _validate_opaque_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("opaque identifier must be str")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("opaque identifier must be non-empty after trim")
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("opaque identifier must not contain control characters")
    return trimmed


def _validate_required_safe_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("safe text must be str")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("safe text must be non-empty after trim")
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("safe text must not contain control characters")
    return trimmed


def _validate_optional_safe_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("safe text must be str")
    trimmed = value.strip()
    if not trimmed:
        return None
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("safe text must not contain control characters")
    return trimmed


def _reject_nul_in_user_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("text must be str")
    if "\x00" in value:
        raise ValueError("text must not contain NUL")
    return value


def _validate_extracted_text_value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("extracted text value must be str")
    if not value:
        raise ValueError("extracted text value must be non-empty")
    if "\x00" in value:
        raise ValueError("extracted text value must not contain NUL")
    return value


def _validate_exact_int(value: object) -> object:
    if value is None:
        return value
    if type(value) is not int:
        raise ValueError("size_bytes must be int")
    return value


def _reject_duplicate_strings(values: tuple[str, ...], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


OpaqueId = Annotated[
    str,
    BeforeValidator(_validate_opaque_id),
    Field(min_length=1, max_length=_MAX_ACTION_ID_LEN),
]
RequiredSafeText = Annotated[
    str,
    BeforeValidator(_validate_required_safe_text),
    Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN),
]
OptionalSafeText = Annotated[
    str | None,
    BeforeValidator(_validate_optional_safe_text),
]
UserText = Annotated[str, BeforeValidator(_reject_nul_in_user_text)]
ExtractedTextValue = Annotated[
    str,
    BeforeValidator(_validate_extracted_text_value),
    Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN),
]


class ConversationPlanningAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    file_name: OptionalSafeText = Field(default=None, max_length=512)
    content_type: Annotated[
        str | None,
        BeforeValidator(_validate_optional_safe_text),
        Field(default=None, max_length=256),
    ] = None
    size_bytes: int | None = Field(default=None, ge=0)

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _validate_size_bytes(cls, value: object) -> object:
        return _validate_exact_int(value)


class ConversationPlanningWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    name: Annotated[str, BeforeValidator(_validate_required_safe_text), Field(min_length=1, max_length=256)]
    is_active: bool


class ConversationPlanningSourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    label: Annotated[str, BeforeValidator(_validate_required_safe_text), Field(min_length=1, max_length=256)]
    source_type: Annotated[
        str, BeforeValidator(_validate_required_safe_text), Field(min_length=1, max_length=128)
    ]
    available: bool


class ConversationPlanningTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    text: Annotated[str, BeforeValidator(_reject_nul_in_user_text), Field(max_length=_MAX_MESSAGE_TEXT_LEN)]


class ConversationPlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_text: UserText = Field(default="", max_length=_MAX_MESSAGE_TEXT_LEN)
    attachments: tuple[ConversationPlanningAttachment, ...] = ()
    available_workspaces: tuple[ConversationPlanningWorkspace, ...] = ()
    active_workspace_id: OpaqueId | None = Field(default=None, max_length=_MAX_ACTION_ID_LEN)
    available_source_candidates: tuple[ConversationPlanningSourceCandidate, ...] = ()
    recent_turns: tuple[ConversationPlanningTurn, ...] = ()

    @model_validator(mode="after")
    def _validate_request(self) -> Self:
        if not self.message_text.strip() and not self.attachments:
            raise ValueError("message_text must be non-empty unless attachments are present")

        attachment_ids = [a.attachment_id for a in self.attachments]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("attachment IDs must be unique")

        workspace_ids = [w.workspace_id for w in self.available_workspaces]
        if len(workspace_ids) != len(set(workspace_ids)):
            raise ValueError("workspace IDs must be unique")

        candidate_ids = [c.candidate_id for c in self.available_source_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")

        if len(self.attachments) > _MAX_ATTACHMENTS:
            raise ValueError("too many attachments")
        if len(self.available_workspaces) > _MAX_WORKSPACES:
            raise ValueError("too many workspaces")
        if len(self.available_source_candidates) > _MAX_SOURCE_CANDIDATES:
            raise ValueError("too many source candidates")
        if len(self.recent_turns) > _MAX_RECENT_TURNS:
            raise ValueError("too many recent turns")

        if self.active_workspace_id is not None:
            if self.active_workspace_id not in set(workspace_ids):
                raise ValueError("active_workspace_id must exist in available_workspaces")

        return self


class WorkspaceReferenceKind(str, Enum):
    active = "active"
    name = "name"
    ordinal = "ordinal"
    created_by_action = "created_by_action"


class WorkspaceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: WorkspaceReferenceKind
    value: str | None = Field(default=None, max_length=_MAX_STRING_FIELD_LEN)

    @model_validator(mode="after")
    def _validate_kind_value(self) -> Self:
        if self.kind == WorkspaceReferenceKind.active:
            if self.value is not None:
                raise ValueError("active workspace reference requires value=None")
        else:
            if self.value is None or not self.value.strip():
                raise ValueError(f"{self.kind.value} requires non-empty value")
            if self.kind == WorkspaceReferenceKind.ordinal:
                if not self.value.isdigit() or int(self.value) < 1:
                    raise ValueError("ordinal must be a positive integer string")
        return self


class _PlannedActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    depends_on: tuple[OpaqueId, ...] = ()
    evidence_quotes: tuple[str, ...] = Field(default=(), max_length=_MAX_EVIDENCE_QUOTES)
    evidence_attachment_ids: tuple[OpaqueId, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_technical_ids(self) -> Self:
        _reject_duplicate_strings(self.depends_on, label="depends_on entry")
        _reject_duplicate_strings(self.evidence_attachment_ids, label="evidence_attachment_id")
        return self


class WorkspaceListPlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.list"]


class WorkspaceCreatePlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.create"]
    name: Annotated[str, BeforeValidator(_validate_required_safe_text), Field(min_length=1, max_length=256)]


class WorkspaceActivatePlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.activate"]
    workspace: WorkspaceReference


class WorkspaceDeletePlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.delete"]
    workspace: WorkspaceReference


class SourceListPlannedAction(_PlannedActionBase):
    action_type: Literal["source.list"]
    workspace: WorkspaceReference


class SourceCandidateListPlannedAction(_PlannedActionBase):
    action_type: Literal["source_candidate.list"]
    workspace: WorkspaceReference


class SourceCandidateAttachPlannedAction(_PlannedActionBase):
    action_type: Literal["source_candidate.attach"]
    workspace: WorkspaceReference
    candidate_reference_kind: Literal["name", "ordinal"]
    candidate_reference: RequiredSafeText = Field(max_length=_MAX_STRING_FIELD_LEN)

    @model_validator(mode="after")
    def _validate_candidate_reference(self) -> Self:
        if self.candidate_reference_kind == "ordinal":
            if not self.candidate_reference.isdigit() or int(self.candidate_reference) < 1:
                raise ValueError("ordinal candidate_reference must be a positive integer string")
        return self


class KnowledgeAddAttachmentsPlannedAction(_PlannedActionBase):
    action_type: Literal["knowledge.add_attachments"]
    workspace: WorkspaceReference
    attachment_ids: tuple[OpaqueId, ...] = Field(
        min_length=1, max_length=_MAX_ATTACHMENT_IDS_PER_ACTION
    )

    @model_validator(mode="after")
    def _validate_unique_attachment_ids(self) -> Self:
        _reject_duplicate_strings(self.attachment_ids, label="attachment_id")
        return self


class MessageTextEvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["message_text"]
    start: int
    end: int
    text: ExtractedTextValue

    @model_validator(mode="after")
    def _validate_span_bounds(self) -> Self:
        if self.start < 0:
            raise ValueError("evidence start must be >= 0")
        if self.end <= self.start:
            raise ValueError("evidence end must be > start")
        return self


class WebUrlExtractedObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    object_type: Literal["web_url"]
    value: ExtractedTextValue
    evidence: MessageTextEvidenceSpan


class LocalFileReferenceExtractedObject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    object_type: Literal["local_file_reference"]
    reference_kind: Literal["file", "folder", "unknown"]
    value: ExtractedTextValue
    evidence: MessageTextEvidenceSpan


ExtractedObject = Annotated[
    WebUrlExtractedObject | LocalFileReferenceExtractedObject,
    Field(discriminator="object_type"),
]


class KnowledgeAddSourcesPlannedAction(_PlannedActionBase):
    action_type: Literal["knowledge.add_sources"]
    workspace: WorkspaceReference
    source_object_ids: tuple[OpaqueId, ...] = Field(
        min_length=1, max_length=_MAX_SOURCE_OBJECT_IDS_PER_ACTION
    )

    @model_validator(mode="after")
    def _validate_unique_source_object_ids(self) -> Self:
        _reject_duplicate_strings(self.source_object_ids, label="source_object_id")
        return self


class WorkspaceAskPlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.ask"]
    workspace: WorkspaceReference
    question: str = Field(min_length=1, max_length=_MAX_MESSAGE_TEXT_LEN)


PlannedAction = Annotated[
    WorkspaceListPlannedAction
    | WorkspaceCreatePlannedAction
    | WorkspaceActivatePlannedAction
    | WorkspaceDeletePlannedAction
    | SourceListPlannedAction
    | SourceCandidateListPlannedAction
    | SourceCandidateAttachPlannedAction
    | KnowledgeAddAttachmentsPlannedAction
    | KnowledgeAddSourcesPlannedAction
    | WorkspaceAskPlannedAction,
    Field(discriminator="action_type"),
]


class ConversationClarification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: OpaqueId = Field(max_length=_MAX_ACTION_ID_LEN)
    question: RequiredSafeText = Field(max_length=_MAX_STRING_FIELD_LEN)
    blocks_action_ids: tuple[OpaqueId, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_blocks(self) -> Self:
        _reject_duplicate_strings(self.blocks_action_ids, label="blocks_action_id")
        return self


class ConversationInteractionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_version: Literal["2"]
    objects: tuple[ExtractedObject, ...] = ()
    actions: tuple[PlannedAction, ...] = ()
    clarifications: tuple[ConversationClarification, ...] = ()
    response_mode: Literal["aggregate"]

    @model_validator(mode="after")
    def _validate_plan_structure(self) -> Self:
        if not self.actions and not self.clarifications:
            raise ValueError("plan must contain at least one action or clarification")

        if len(self.objects) > _MAX_OBJECTS:
            raise ValueError("too many objects")
        if len(self.actions) > _MAX_ACTIONS:
            raise ValueError("too many actions")
        if len(self.clarifications) > _MAX_CLARIFICATIONS:
            raise ValueError("too many clarifications")

        object_ids = [obj.object_id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("duplicate object_id")

        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("duplicate action_id")

        clarification_ids = [item.clarification_id for item in self.clarifications]
        if len(clarification_ids) != len(set(clarification_ids)):
            raise ValueError("duplicate clarification_id")

        object_id_set = set(object_ids)
        action_id_set = set(action_ids)
        for action in self.actions:
            if action.action_type != "knowledge.add_sources":
                continue
            for source_id in action.source_object_ids:
                if source_id not in object_id_set:
                    raise ValueError(f"unknown source object: {source_id}")
        for action in self.actions:
            for dep in action.depends_on:
                if dep not in action_id_set:
                    raise ValueError(f"unknown dependency: {dep}")
                if dep == action.action_id:
                    raise ValueError("self dependency not allowed")

        if _has_dependency_cycle(self.actions):
            raise ValueError("dependency cycle detected")

        for clarification in self.clarifications:
            for blocked_id in clarification.blocks_action_ids:
                if blocked_id not in action_id_set:
                    raise ValueError(f"unknown blocked action: {blocked_id}")

        create_action_ids = {
            action.action_id
            for action in self.actions
            if action.action_type == "workspace.create"
        }
        for action in self.actions:
            workspace = _extract_workspace_reference(action)
            if workspace is None:
                continue
            if workspace.kind != WorkspaceReferenceKind.created_by_action:
                continue
            create_id = workspace.value
            if create_id not in create_action_ids:
                raise ValueError("created_by_action must reference workspace.create action")
            if create_id not in action.depends_on:
                raise ValueError(
                    "action using created_by_action must depend on the workspace.create action"
                )

        return self


def _extract_workspace_reference(action: PlannedAction) -> WorkspaceReference | None:
    if hasattr(action, "workspace"):
        workspace = action.workspace  # type: ignore[attr-defined]
        if isinstance(workspace, WorkspaceReference):
            return workspace
    return None


def _has_dependency_cycle(actions: tuple[PlannedAction, ...]) -> bool:
    graph: dict[str, list[str]] = {action.action_id: list(action.depends_on) for action in actions}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in graph.get(node, []):
            if visit(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(action_id) for action_id in graph)


def collect_user_text_context(request: ConversationPlanningRequest) -> tuple[str, ...]:
    """Return message_text plus recent user turn texts for evidence and reference checks."""
    parts: list[str] = []
    if request.message_text:
        parts.append(request.message_text)
    for turn in request.recent_turns:
        if turn.role == "user" and turn.text:
            parts.append(turn.text)
    return tuple(parts)


def request_attachment_ids(request: ConversationPlanningRequest) -> frozenset[str]:
    return frozenset(attachment.attachment_id for attachment in request.attachments)
