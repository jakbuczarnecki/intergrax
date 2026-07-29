# © Artur Czarnecki. All rights reserved.

"""Deterministic tests for semantic interaction draft models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_workspace_application.conversation.interaction_draft_models import (
    ActiveDraftWorkspaceReference,
    ConversationClarificationDraft,
    ConversationInteractionDraft,
    CreatedByActionDraftWorkspaceReference,
    DraftLocalFileReferenceSource,
    DraftWebUrlSource,
    DraftWorkspaceReference,
    KnowledgeAddSourcesDraftAction,
    NameDraftWorkspaceReference,
    OrdinalDraftWorkspaceReference,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    WorkspaceReferenceKind,
)

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
                workspace=NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="projekty"),
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


def _resolve_schema_ref(schema: dict, ref: str) -> dict:
    if not ref.startswith("#/$defs/"):
        raise ValueError(f"unsupported ref: {ref}")
    def_name = ref.removeprefix("#/$defs/")
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        raise ValueError("schema has no $defs")
    branch = defs.get(def_name)
    if not isinstance(branch, dict):
        raise ValueError(f"unknown schema def: {def_name}")
    return branch


def _kind_const_or_enum(branch: dict) -> str:
    kind_schema = branch.get("properties", {}).get("kind", {})
    if not isinstance(kind_schema, dict):
        raise ValueError("kind schema missing")
    const_value = kind_schema.get("const")
    if isinstance(const_value, str):
        return const_value
    enum_values = kind_schema.get("enum")
    if isinstance(enum_values, list) and len(enum_values) == 1 and isinstance(enum_values[0], str):
        return enum_values[0]
    raise ValueError("kind const/enum not found")


def _workspace_union_schema() -> dict:
    from pydantic import TypeAdapter

    return TypeAdapter(DraftWorkspaceReference).json_schema()


@pytest.mark.unit
def test_workspace_reference_discriminator_schema() -> None:
    schema = _workspace_union_schema()
    discriminator = schema.get("discriminator")
    assert isinstance(discriminator, dict)
    assert discriminator.get("propertyName") == "kind"
    mapping = discriminator.get("mapping")
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {kind.value for kind in WorkspaceReferenceKind}
    branches = [_resolve_schema_ref(schema, ref) for ref in mapping.values()]
    assert len(branches) == 4
    assert len({_kind_const_or_enum(branch) for branch in branches}) == 4


@pytest.mark.unit
def test_workspace_reference_variant_schemas() -> None:
    schema = _workspace_union_schema()
    mapping = schema["discriminator"]["mapping"]
    branches = {key: _resolve_schema_ref(schema, ref) for key, ref in mapping.items()}

    active = branches[WorkspaceReferenceKind.active.value]
    active_value = active["properties"]["value"]
    assert _kind_const_or_enum(active) == WorkspaceReferenceKind.active.value
    assert active_value.get("type") == "null" or active_value.get("const") is None

    name = branches[WorkspaceReferenceKind.name.value]
    assert _kind_const_or_enum(name) == WorkspaceReferenceKind.name.value
    assert "value" in name.get("required", [])
    assert name["properties"]["value"].get("type") == "string"

    ordinal = branches[WorkspaceReferenceKind.ordinal.value]
    assert _kind_const_or_enum(ordinal) == WorkspaceReferenceKind.ordinal.value
    assert "value" in ordinal.get("required", [])
    assert ordinal["properties"]["value"].get("type") == "string"
    assert ordinal["properties"]["value"].get("pattern") == r"^0*[1-9][0-9]*$"

    created = branches[WorkspaceReferenceKind.created_by_action.value]
    assert _kind_const_or_enum(created) == WorkspaceReferenceKind.created_by_action.value
    assert "value" in created.get("required", [])
    assert created["properties"]["value"].get("type") == "string"


@pytest.mark.unit
def test_full_draft_schema_uses_workspace_discriminator() -> None:
    schema = ConversationInteractionDraft.model_json_schema()
    schema_text = str(schema)
    assert WorkspaceReferenceKind.active.value in schema_text
    assert WorkspaceReferenceKind.name.value in schema_text
    assert WorkspaceReferenceKind.ordinal.value in schema_text
    assert WorkspaceReferenceKind.created_by_action.value in schema_text
    property_names = _collect_schema_property_names(schema)
    assert "workspace" in property_names


@pytest.mark.unit
def test_active_workspace_reference_runtime() -> None:
    omitted = ActiveDraftWorkspaceReference(kind=WorkspaceReferenceKind.active)
    assert omitted.value is None
    explicit_null = ActiveDraftWorkspaceReference(kind=WorkspaceReferenceKind.active, value=None)
    assert explicit_null.value is None
    with pytest.raises(ValidationError):
        ActiveDraftWorkspaceReference(kind=WorkspaceReferenceKind.active, value="x")


@pytest.mark.unit
def test_name_workspace_reference_runtime() -> None:
    ref = NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value="  magazyn  ")
    assert ref.value == "  magazyn  "
    for invalid in (None, "", "   ", "bad\x07value"):
        with pytest.raises(ValidationError):
            NameDraftWorkspaceReference(kind=WorkspaceReferenceKind.name, value=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["1", "2", "10", "01", "0007"])
@pytest.mark.unit
def test_ordinal_workspace_reference_valid(value: str) -> None:
    ref = OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value=value)
    assert ref.value == value
    from pydantic import TypeAdapter

    parsed = TypeAdapter(DraftWorkspaceReference).validate_python(
        {"kind": WorkspaceReferenceKind.ordinal.value, "value": value}
    )
    assert isinstance(parsed, OrdinalDraftWorkspaceReference)
    assert parsed.value == value


@pytest.mark.parametrize(
    "value",
    ["", "0", "000", "-1", "+1", "1.0", " 1", "1 ", "default_workspace_1", 1, 1.0, True, None],
)
@pytest.mark.unit
def test_ordinal_workspace_reference_invalid(value: object) -> None:
    with pytest.raises(ValidationError):
        OrdinalDraftWorkspaceReference(kind=WorkspaceReferenceKind.ordinal, value=value)  # type: ignore[arg-type]
    from pydantic import TypeAdapter

    with pytest.raises(ValidationError):
        TypeAdapter(DraftWorkspaceReference).validate_python(
            {"kind": WorkspaceReferenceKind.ordinal.value, "value": value}
        )


@pytest.mark.unit
def test_created_by_action_workspace_reference_runtime() -> None:
    ref = CreatedByActionDraftWorkspaceReference(
        kind=WorkspaceReferenceKind.created_by_action,
        value="  alpha  ",
    )
    assert ref.value == "  alpha  "
    with pytest.raises(ValidationError):
        CreatedByActionDraftWorkspaceReference(
            kind=WorkspaceReferenceKind.created_by_action,
            value=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        CreatedByActionDraftWorkspaceReference(
            kind=WorkspaceReferenceKind.created_by_action,
            value="   ",
        )


@pytest.mark.unit
def test_workspace_reference_discriminator_selection() -> None:
    from pydantic import TypeAdapter

    adapter = TypeAdapter(DraftWorkspaceReference)
    cases: list[tuple[dict[str, object], type]] = [
        ({"kind": WorkspaceReferenceKind.active.value}, ActiveDraftWorkspaceReference),
        ({"kind": WorkspaceReferenceKind.name.value, "value": "magazyn"}, NameDraftWorkspaceReference),
        ({"kind": WorkspaceReferenceKind.ordinal.value, "value": "1"}, OrdinalDraftWorkspaceReference),
        (
            {"kind": WorkspaceReferenceKind.created_by_action.value, "value": "alpha"},
            CreatedByActionDraftWorkspaceReference,
        ),
    ]
    for payload, expected_type in cases:
        parsed = adapter.validate_python(payload)
        assert isinstance(parsed, expected_type)
