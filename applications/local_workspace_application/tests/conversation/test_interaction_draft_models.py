# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for semantic interaction draft models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_workspace_application.conversation.interaction_draft_models import (
    ConversationClarificationDraft,
    ConversationInteractionDraft,
    DraftLocalFileReferenceSource,
    DraftWebUrlSource,
    KnowledgeAddSourcesDraftAction,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_draft_models import DraftWorkspaceReference

_EXPECTED_ACTION_TYPES = frozenset(
    {
        "workspace.list",
        "workspace.create",
        "workspace.activate",
        "workspace.delete",
        "source.list",
        "source_candidate.list",
        "source_candidate.attach",
        "knowledge.add_attachments",
        "knowledge.add_sources",
        "workspace.ask",
    }
)

_FORBIDDEN_SCHEMA_PROPERTIES = frozenset(
    {
        "plan_version",
        "response_mode",
        "objects",
        "action_id",
        "object_id",
        "clarification_id",
        "source_object_ids",
        "evidence",
        "start",
        "end",
        "depends_on",
        "blocks_action_ids",
    }
)

_REQUIRED_SEMANTIC_PROPERTIES = frozenset(
    {
        "actions",
        "clarifications",
        "depends_on_action_numbers",
        "blocks_action_numbers",
        "sources",
        "occurrence",
    }
)


def _collect_action_type_literals(node: object) -> set[str]:
    literals: set[str] = set()
    if isinstance(node, dict):
        if "action_type" in node:
            action_type_schema = node["action_type"]
            if isinstance(action_type_schema, dict):
                const_value = action_type_schema.get("const")
                if isinstance(const_value, str):
                    literals.add(const_value)
                enum_values = action_type_schema.get("enum")
                if isinstance(enum_values, list):
                    literals.update(value for value in enum_values if isinstance(value, str))
        for key in ("properties", "$defs", "items"):
            child = node.get(key)
            if child is not None:
                literals.update(_collect_action_type_literals(child))
        for key in ("oneOf", "anyOf", "allOf"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    literals.update(_collect_action_type_literals(child))
        for value in node.values():
            if value is not node.get("properties") and value is not node.get("$defs"):
                literals.update(_collect_action_type_literals(value))
    elif isinstance(node, list):
        for item in node:
            literals.update(_collect_action_type_literals(item))
    return literals


def _collect_schema_property_names(node: object) -> set[str]:
    names: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(properties.keys())
            for child in properties.values():
                names.update(_collect_schema_property_names(child))
        for key in ("$defs", "items"):
            child = node.get(key)
            if child is not None:
                names.update(_collect_schema_property_names(child))
        for key in ("oneOf", "anyOf", "allOf"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    names.update(_collect_schema_property_names(child))
        for key, value in node.items():
            if key not in {"properties", "$defs", "items", "oneOf", "anyOf", "allOf"}:
                names.update(_collect_schema_property_names(value))
    elif isinstance(node, list):
        for item in node:
            names.update(_collect_schema_property_names(item))
    return names


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
    schema = ConversationInteractionDraft.model_json_schema()
    property_names = _collect_schema_property_names(schema)
    assert not property_names & _FORBIDDEN_SCHEMA_PROPERTIES
    assert _REQUIRED_SEMANTIC_PROPERTIES <= property_names


@pytest.mark.unit
def test_every_canonical_action_has_draft_variant() -> None:
    canonical = _collect_action_type_literals(ConversationInteractionPlan.model_json_schema())
    draft = _collect_action_type_literals(ConversationInteractionDraft.model_json_schema())
    assert canonical
    assert draft
    assert draft == canonical
    assert canonical == _EXPECTED_ACTION_TYPES


@pytest.mark.unit
def test_action_discriminators_use_action_type_literals() -> None:
    draft_action_types = _collect_action_type_literals(
        ConversationInteractionDraft.model_json_schema()
    )
    assert draft_action_types == _EXPECTED_ACTION_TYPES


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
