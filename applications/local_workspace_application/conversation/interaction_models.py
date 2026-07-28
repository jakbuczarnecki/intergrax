# © Artur Czarnecki. All rights reserved.

"""Channel-neutral structured interaction plan models for LKW conversational planning."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
_MAX_URLS_PER_ACTION = 50
_MAX_LOCAL_REFS_PER_ACTION = 50
_MAX_ATTACHMENT_IDS_PER_ACTION = 50


class ConversationPlanningAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_id: str = Field(min_length=1, max_length=_MAX_ACTION_ID_LEN)
    file_name: str | None = Field(default=None, max_length=512)
    content_type: str | None = Field(default=None, max_length=256)
    size_bytes: int | None = Field(default=None, ge=0)


class ConversationPlanningWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str = Field(min_length=1, max_length=_MAX_ACTION_ID_LEN)
    name: str = Field(min_length=1, max_length=256)
    is_active: bool


class ConversationPlanningSourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=_MAX_ACTION_ID_LEN)
    label: str = Field(min_length=1, max_length=256)
    source_type: str = Field(min_length=1, max_length=128)
    available: bool


class ConversationPlanningTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    text: str = Field(max_length=_MAX_MESSAGE_TEXT_LEN)


class ConversationPlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_text: str = Field(default="", max_length=_MAX_MESSAGE_TEXT_LEN)
    attachments: tuple[ConversationPlanningAttachment, ...] = ()
    available_workspaces: tuple[ConversationPlanningWorkspace, ...] = ()
    active_workspace_id: str | None = Field(default=None, max_length=_MAX_ACTION_ID_LEN)
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

    action_id: str = Field(min_length=1, max_length=_MAX_ACTION_ID_LEN)
    depends_on: tuple[str, ...] = ()
    evidence_quotes: tuple[str, ...] = Field(default=(), max_length=_MAX_EVIDENCE_QUOTES)
    evidence_attachment_ids: tuple[str, ...] = ()


class WorkspaceListPlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.list"]


class WorkspaceCreatePlannedAction(_PlannedActionBase):
    action_type: Literal["workspace.create"]
    name: str = Field(min_length=1, max_length=256)


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
    candidate_reference: str = Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN)


class KnowledgeAddAttachmentsPlannedAction(_PlannedActionBase):
    action_type: Literal["knowledge.add_attachments"]
    workspace: WorkspaceReference
    attachment_ids: tuple[str, ...] = Field(min_length=1, max_length=_MAX_ATTACHMENT_IDS_PER_ACTION)


class KnowledgeAddWebUrlsPlannedAction(_PlannedActionBase):
    action_type: Literal["knowledge.add_web_urls"]
    workspace: WorkspaceReference
    urls: tuple[str, ...] = Field(min_length=1, max_length=_MAX_URLS_PER_ACTION)


class LocalReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["file", "folder", "unknown"]
    value: str = Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN)


class KnowledgeAddLocalReferencesPlannedAction(_PlannedActionBase):
    action_type: Literal["knowledge.add_local_references"]
    workspace: WorkspaceReference
    references: tuple[LocalReference, ...] = Field(
        min_length=1, max_length=_MAX_LOCAL_REFS_PER_ACTION
    )


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
    | KnowledgeAddWebUrlsPlannedAction
    | KnowledgeAddLocalReferencesPlannedAction
    | WorkspaceAskPlannedAction,
    Field(discriminator="action_type"),
]


class ConversationClarification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_id: str = Field(min_length=1, max_length=_MAX_ACTION_ID_LEN)
    question: str = Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN)
    blocks_action_ids: tuple[str, ...] = ()


class ConversationInteractionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_version: Literal["1"]
    actions: tuple[PlannedAction, ...] = ()
    clarifications: tuple[ConversationClarification, ...] = ()
    response_mode: Literal["aggregate"]

    @model_validator(mode="after")
    def _validate_plan_structure(self) -> Self:
        if not self.actions and not self.clarifications:
            raise ValueError("plan must contain at least one action or clarification")

        if len(self.actions) > _MAX_ACTIONS:
            raise ValueError("too many actions")
        if len(self.clarifications) > _MAX_CLARIFICATIONS:
            raise ValueError("too many clarifications")

        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("duplicate action_id")

        action_id_set = set(action_ids)
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
