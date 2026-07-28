# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for semantic interaction draft models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_workspace_application.conversation.interaction_draft_models import (
    CANONICAL_ACTION_TYPES,
    DRAFT_ACTION_TYPES,
    ConversationClarificationDraft,
    ConversationInteractionDraft,
    DraftLocalFileReferenceSource,
    DraftWebUrlSource,
    KnowledgeAddSourcesDraftAction,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import WorkspaceReferenceKind
from local_workspace_application.conversation.interaction_draft_models import DraftWorkspaceReference


@pytest.mark.unit
def test_draft_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ConversationInteractionDraft.model_validate(
            {"actions": [], "clarifications": [], "unexpected": True}
        )


@pytest.mark.unit
def test_draft_schema_generated_from_class() -> None:
    schema = ConversationInteractionDraft.model_json_schema()
    assert schema["title"] == "ConversationInteractionDraft"
    assert "properties" in schema


@pytest.mark.unit
def test_draft_schema_excludes_technical_fields() -> None:
    schema_text = str(ConversationInteractionDraft.model_json_schema())
    forbidden = (
        "action_id",
        "object_id",
        "clarification_id",
        "source_object_ids",
        '"start"',
        '"end"',
    )
    for token in forbidden:
        assert token not in schema_text


@pytest.mark.unit
def test_every_canonical_action_has_draft_variant() -> None:
    assert DRAFT_ACTION_TYPES == CANONICAL_ACTION_TYPES


@pytest.mark.unit
def test_action_discriminators_use_action_type_literals() -> None:
    schema = ConversationInteractionDraft.model_json_schema()
    actions_schema = str(schema)
    for action_type in CANONICAL_ACTION_TYPES:
        assert action_type in actions_schema


@pytest.mark.unit
def test_source_discriminators_use_object_type_literals() -> None:
    schema = ConversationInteractionDraft.model_json_schema()
    assert "web_url" in str(schema)
    assert "local_file_reference" in str(schema)


@pytest.mark.parametrize(
    ("value",),
    [
        (True,),
        (False,),
        ("1",),
        (1.0,),
        (0,),
        (-1,),
    ],
)
@pytest.mark.unit
def test_occurrence_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError):
        DraftWebUrlSource(object_type="web_url", value="https://a.example", occurrence=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value",),
    [
        (True,),
        ("1",),
        (1.0,),
        (0,),
        (-1,),
    ],
)
@pytest.mark.unit
def test_action_number_rejects_non_exact_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        WorkspaceListDraftAction(action_type="workspace.list", depends_on_action_numbers=(value,))  # type: ignore[arg-type]


@pytest.mark.unit
def test_draft_requires_action_or_clarification() -> None:
    with pytest.raises(ValidationError, match="at least one action or clarification"):
        ConversationInteractionDraft(actions=(), clarifications=())


@pytest.mark.unit
def test_duplicate_action_numbers_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate depends_on_action_number"):
        WorkspaceListDraftAction(
            action_type="workspace.list",
            depends_on_action_numbers=(1, 1),
        )


@pytest.mark.unit
def test_duplicate_blocks_action_numbers_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate blocks_action_number"):
        ConversationClarificationDraft(question="Which workspace?", blocks_action_numbers=(1, 1))


@pytest.mark.unit
def test_valid_draft_with_sources() -> None:
    draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=DraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="projekty"),
                sources=(
                    DraftWebUrlSource(object_type="web_url", value="https://portal.vendor.io"),
                ),
            ),
        ),
    )
    assert len(draft.actions) == 1


@pytest.mark.unit
def test_local_source_reference_kind_required() -> None:
    source = DraftLocalFileReferenceSource(
        object_type="local_file_reference",
        reference_kind="file",
        value=r"C:\data\file.txt",
    )
    assert source.reference_kind == "file"
