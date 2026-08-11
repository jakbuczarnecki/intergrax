# © Artur Czarnecki. All rights reserved.

"""Semantic draft models for LLM interaction planning — no technical IDs or evidence offsets."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictStr, model_validator

_MAX_ACTIONS = 50
_MAX_CLARIFICATIONS = 20
_MAX_STRING_FIELD_LEN = 2_000
_MAX_MESSAGE_TEXT_LEN = 16_000
_MAX_EVIDENCE_QUOTES = 20
_MAX_ATTACHMENT_IDS_PER_ACTION = 50
_MAX_SOURCE_OBJECTS_PER_ACTION = 50
_MAX_ACTION_ID_LEN = 128

_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _validate_required_safe_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("safe text must be str")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("safe text must be non-empty after trim")
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("safe text must not contain control characters")
    return trimmed


def _validate_opaque_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("opaque identifier must be str")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("opaque identifier must be non-empty after trim")
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError("opaque identifier must not contain control characters")
    return trimmed


def _validate_exact_positive_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("value must be exact int")
    if value < 1:
        raise ValueError("value must be positive")
    return value


def _validate_exact_positive_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return _validate_exact_positive_int(value)


def _validate_source_value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("source value must be str")
    if not value:
        raise ValueError("source value must be non-empty")
    if "\x00" in value:
        raise ValueError("source value must not contain NUL")
    return value


def _reject_duplicate_ints(values: tuple[int, ...], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


ExactPositiveInt = Annotated[int, BeforeValidator(_validate_exact_positive_int)]
ExactPositiveIntOrNone = Annotated[
    int | None,
    BeforeValidator(_validate_exact_positive_int_or_none),
]
RequiredSafeText = Annotated[
    str,
    BeforeValidator(_validate_required_safe_text),
    Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN),
]
OpaqueId = Annotated[
    str,
    BeforeValidator(_validate_opaque_id),
    Field(min_length=1, max_length=_MAX_ACTION_ID_LEN),
]
SourceValue = Annotated[
    str,
    BeforeValidator(_validate_source_value),
    Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN),
]


def _validate_exact_workspace_reference_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("workspace reference text must be str")
    if not value.strip():
        raise ValueError("workspace reference text must be non-empty")
    if _ASCII_CONTROL.search(value):
        raise ValueError("workspace reference text must not contain control characters")
    return value


ExactWorkspaceReferenceText = Annotated[
    str,
    BeforeValidator(_validate_exact_workspace_reference_text),
    Field(min_length=1, max_length=_MAX_STRING_FIELD_LEN),
]


class DraftWorkspaceReferenceBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActiveDraftWorkspaceReference(DraftWorkspaceReferenceBase):
    kind: Literal["active"]
    value: None = None


class NameDraftWorkspaceReference(DraftWorkspaceReferenceBase):
    kind: Literal["name"]
    value: ExactWorkspaceReferenceText


class OrdinalDraftWorkspaceReference(DraftWorkspaceReferenceBase):
    kind: Literal["ordinal"]
    value: Annotated[
        StrictStr,
        Field(
            min_length=1,
            max_length=_MAX_STRING_FIELD_LEN,
            pattern=r"^0*[1-9][0-9]*$",
        ),
    ]


class CreatedByActionDraftWorkspaceReference(DraftWorkspaceReferenceBase):
    kind: Literal["created_by_action"]
    value: ExactWorkspaceReferenceText


DraftWorkspaceReference = Annotated[
    ActiveDraftWorkspaceReference
    | NameDraftWorkspaceReference
    | OrdinalDraftWorkspaceReference
    | CreatedByActionDraftWorkspaceReference,
    Field(discriminator="kind"),
]


class DraftWebUrlSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Literal["web_url"]
    value: SourceValue
    occurrence: ExactPositiveIntOrNone = None


class DraftLocalFileReferenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Literal["local_file_reference"]
    reference_kind: Literal["file", "folder", "unknown"]
    value: SourceValue
    occurrence: ExactPositiveIntOrNone = None


DraftSource = Annotated[
    DraftWebUrlSource | DraftLocalFileReferenceSource,
    Field(discriminator="object_type"),
]


class _DraftActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depends_on_action_numbers: tuple[ExactPositiveInt, ...] = ()
    evidence_quotes: tuple[str, ...] = Field(default=(), max_length=_MAX_EVIDENCE_QUOTES)
    evidence_attachment_ids: tuple[OpaqueId, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_references(self) -> Self:
        _reject_duplicate_ints(self.depends_on_action_numbers, label="depends_on_action_number")
        seen: set[str] = set()
        for attachment_id in self.evidence_attachment_ids:
            if attachment_id in seen:
                raise ValueError("duplicate evidence_attachment_id")
            seen.add(attachment_id)
        return self


class WorkspaceListDraftAction(_DraftActionBase):
    action_type: Literal["workspace.list"]


class WorkspaceCreateDraftAction(_DraftActionBase):
    action_type: Literal["workspace.create"]
    name: Annotated[str, BeforeValidator(_validate_required_safe_text), Field(min_length=1, max_length=256)]


class WorkspaceActivateDraftAction(_DraftActionBase):
    action_type: Literal["workspace.activate"]
    workspace: DraftWorkspaceReference


class WorkspaceDeleteDraftAction(_DraftActionBase):
    action_type: Literal["workspace.delete"]
    workspace: DraftWorkspaceReference


class SourceListDraftAction(_DraftActionBase):
    action_type: Literal["source.list"]
    workspace: DraftWorkspaceReference


class SourceCandidateListDraftAction(_DraftActionBase):
    action_type: Literal["source_candidate.list"]
    workspace: DraftWorkspaceReference


class SourceCandidateAttachDraftAction(_DraftActionBase):
    action_type: Literal["source_candidate.attach"]
    workspace: DraftWorkspaceReference
    candidate_reference_kind: Literal["name", "ordinal"]
    candidate_reference: RequiredSafeText = Field(max_length=_MAX_STRING_FIELD_LEN)

    @model_validator(mode="after")
    def _validate_candidate_reference(self) -> Self:
        if self.candidate_reference_kind == "ordinal":
            if not self.candidate_reference.isdigit() or int(self.candidate_reference) < 1:
                raise ValueError("ordinal candidate_reference must be a positive integer string")
        return self


class KnowledgeAddAttachmentsDraftAction(_DraftActionBase):
    action_type: Literal["knowledge.add_attachments"]
    workspace: DraftWorkspaceReference
    attachment_ids: tuple[OpaqueId, ...] = Field(
        min_length=1, max_length=_MAX_ATTACHMENT_IDS_PER_ACTION
    )

    @model_validator(mode="after")
    def _validate_unique_attachment_ids(self) -> Self:
        seen: set[str] = set()
        for attachment_id in self.attachment_ids:
            if attachment_id in seen:
                raise ValueError("duplicate attachment_id")
            seen.add(attachment_id)
        return self


class KnowledgeAddSourcesDraftAction(_DraftActionBase):
    action_type: Literal["knowledge.add_sources"]
    workspace: DraftWorkspaceReference
    sources: tuple[DraftSource, ...] = Field(
        min_length=1, max_length=_MAX_SOURCE_OBJECTS_PER_ACTION
    )


class KnowledgeConnectionsListDraftAction(_DraftActionBase):
    action_type: Literal["knowledge.connections.list"]


class KnowledgeResourcesListDraftAction(_DraftActionBase):
    action_type: Literal["knowledge.resources.list"]
    connection_ref: OpaqueId = Field(max_length=128)
    source_kind: OpaqueId = Field(max_length=64)
    page_token: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def _validate_page_token(self) -> Self:
        if self.page_token is not None and not self.page_token.strip():
            raise ValueError("page_token must not be blank")
        return self


class KnowledgeCapabilitiesListDraftAction(_DraftActionBase):
    action_type: Literal["knowledge.capabilities.list"]
    connection_ref: OpaqueId = Field(max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _validate_resource_id(self) -> Self:
        if self.remote_resource_id is not None and not self.remote_resource_id.strip():
            raise ValueError("remote_resource_id must not be blank")
        return self


class WorkspaceAskDraftAction(_DraftActionBase):
    action_type: Literal["workspace.ask"]
    workspace: DraftWorkspaceReference
    question: str = Field(min_length=1, max_length=_MAX_MESSAGE_TEXT_LEN)


class CitationInspectDraftAction(_DraftActionBase):
    action_type: Literal["citation.inspect"]
    workspace: DraftWorkspaceReference
    citation_reference_kind: Literal["ordinal"]
    citation_reference: RequiredSafeText = Field(max_length=_MAX_STRING_FIELD_LEN)

    @model_validator(mode="after")
    def _validate_citation_reference(self) -> Self:
        if not self.citation_reference.isdigit() or int(self.citation_reference) < 1:
            raise ValueError("ordinal citation_reference must be a positive integer string")
        return self


DraftPlannedAction = Annotated[
    WorkspaceListDraftAction
    | WorkspaceCreateDraftAction
    | WorkspaceActivateDraftAction
    | WorkspaceDeleteDraftAction
    | SourceListDraftAction
    | SourceCandidateListDraftAction
    | SourceCandidateAttachDraftAction
    | KnowledgeAddAttachmentsDraftAction
    | KnowledgeAddSourcesDraftAction
    | KnowledgeConnectionsListDraftAction
    | KnowledgeResourcesListDraftAction
    | KnowledgeCapabilitiesListDraftAction
    | WorkspaceAskDraftAction
    | CitationInspectDraftAction,
    Field(discriminator="action_type"),
]


class ConversationClarificationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: RequiredSafeText = Field(max_length=_MAX_STRING_FIELD_LEN)
    blocks_action_numbers: tuple[ExactPositiveInt, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_blocks(self) -> Self:
        _reject_duplicate_ints(self.blocks_action_numbers, label="blocks_action_number")
        return self


class ConversationInteractionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[DraftPlannedAction, ...] = ()
    clarifications: tuple[ConversationClarificationDraft, ...] = ()

    @model_validator(mode="after")
    def _validate_draft_structure(self) -> Self:
        if not self.actions and not self.clarifications:
            raise ValueError("draft must contain at least one action or clarification")
        if len(self.actions) > _MAX_ACTIONS:
            raise ValueError("too many actions")
        if len(self.clarifications) > _MAX_CLARIFICATIONS:
            raise ValueError("too many clarifications")
        return self

