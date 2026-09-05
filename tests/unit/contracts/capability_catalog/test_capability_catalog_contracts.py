# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 1 contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1,
    SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1,
    SCHEMA_CAPABILITY_PROVENANCE_V1,
    SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityStageVocabulary,
    NORMATIVE_CAPABILITY_STAGE_VOCABULARY,
    V1_CAPABILITY_KINDS,
)

pytestmark = pytest.mark.unit


def _source(source_id: str = "official.catalog") -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _identity(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    source_id: str = "official.catalog",
    logical_id: str = "tools.rag.search",
) -> CapabilityDiscoveryIdentity:
    return CapabilityDiscoveryIdentity(
        kind=kind,
        source=_source(source_id),
        logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
    )


def test_v1_capability_kinds_closed_to_agent_skill_tool() -> None:
    assert V1_CAPABILITY_KINDS == frozenset(CapabilityKind)
    assert len(CapabilityKind) == 3
    assert {member.value for member in CapabilityKind} == {"agent", "skill", "tool"}


def test_normative_stage_vocabulary_matches_architecture() -> None:
    expected = {
        "available",
        "discovered",
        "selected",
        "installed",
        "enabled",
        "materialized",
        "active",
        "executable",
    }
    assert {member.value for member in CapabilityStageVocabulary} == expected
    assert NORMATIVE_CAPABILITY_STAGE_VOCABULARY == frozenset(CapabilityStageVocabulary)


def test_discovery_identity_is_frozen_and_preserves_source() -> None:
    identity = _identity()
    assert identity.source.source_id == "official.catalog"
    assert identity.source.source_kind is CapabilitySourceKind.OFFICIAL
    with pytest.raises(ValidationError):
        identity.source = _source("other.source")


def test_discovery_identity_rejects_kind_mismatch() -> None:
    with pytest.raises(ValidationError, match="logical identity kind must match"):
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=_source(),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id="skills.browser",
            ),
        )


def test_discovery_identity_rejects_empty_logical_id() -> None:
    with pytest.raises(ValidationError):
        CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id="  ")


def test_provenance_optional_fields_fail_on_empty_string() -> None:
    with pytest.raises(ValidationError):
        CapabilityProvenance(source=_source(), version_label="")


def test_provenance_allows_absent_optional_fields() -> None:
    provenance = CapabilityProvenance(source=_source())
    assert provenance.version_label is None
    assert provenance.package_reference is None
    assert provenance.content_digest is None
    assert provenance.publisher is None


@pytest.mark.parametrize(
    ("model_factory", "schema_constant", "foreign_version"),
    [
        (
            lambda: CapabilitySourceIdentity(source_id="official.catalog"),
            SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1,
            "capability_source_identity.v2",
        ),
        (
            lambda: CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.rag.search",
            ),
            SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1,
            "capability_logical_identity.v2",
        ),
        (
            lambda: _identity(),
            SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1,
            "capability_discovery_identity.v2",
        ),
        (
            lambda: CapabilityProvenance(source=_source()),
            SCHEMA_CAPABILITY_PROVENANCE_V1,
            "capability_provenance.v2",
        ),
    ],
    ids=[
        "CapabilitySourceIdentity",
        "CapabilityLogicalIdentity",
        "CapabilityDiscoveryIdentity",
        "CapabilityProvenance",
    ],
)
def test_schema_version_defaults_to_v1(
    model_factory: object,
    schema_constant: str,
    foreign_version: str,
) -> None:
    model = model_factory()  # type: ignore[operator]
    assert model.schema_version == schema_constant


@pytest.mark.parametrize(
    ("model_factory", "schema_constant"),
    [
        (
            lambda: CapabilitySourceIdentity(
                schema_version=SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1,
                source_id="official.catalog",
            ),
            SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1,
        ),
        (
            lambda: CapabilityLogicalIdentity(
                schema_version=SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1,
                kind=CapabilityKind.TOOL,
                logical_id="tools.rag.search",
            ),
            SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1,
        ),
        (
            lambda: CapabilityDiscoveryIdentity(
                schema_version=SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1,
                kind=CapabilityKind.TOOL,
                source=_source(),
                logical=CapabilityLogicalIdentity(
                    kind=CapabilityKind.TOOL,
                    logical_id="tools.rag.search",
                ),
            ),
            SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1,
        ),
        (
            lambda: CapabilityProvenance(
                schema_version=SCHEMA_CAPABILITY_PROVENANCE_V1,
                source=_source(),
            ),
            SCHEMA_CAPABILITY_PROVENANCE_V1,
        ),
    ],
    ids=[
        "CapabilitySourceIdentity",
        "CapabilityLogicalIdentity",
        "CapabilityDiscoveryIdentity",
        "CapabilityProvenance",
    ],
)
def test_schema_version_accepts_explicit_v1(
    model_factory: object,
    schema_constant: str,
) -> None:
    model = model_factory()  # type: ignore[operator]
    assert model.schema_version == schema_constant


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: CapabilitySourceIdentity(
            schema_version="capability_source_identity.v2",
            source_id="official.catalog",
        ),
        lambda: CapabilityLogicalIdentity(
            schema_version="capability_logical_identity.v2",
            kind=CapabilityKind.TOOL,
            logical_id="tools.rag.search",
        ),
        lambda: CapabilityDiscoveryIdentity(
            schema_version="capability_discovery_identity.v2",
            kind=CapabilityKind.TOOL,
            source=_source(),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.rag.search",
            ),
        ),
        lambda: CapabilityProvenance(
            schema_version="capability_provenance.v2",
            source=_source(),
        ),
    ],
    ids=[
        "CapabilitySourceIdentity",
        "CapabilityLogicalIdentity",
        "CapabilityDiscoveryIdentity",
        "CapabilityProvenance",
    ],
)
def test_schema_version_rejects_foreign_version(model_factory: object) -> None:
    with pytest.raises(ValidationError):
        model_factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: CapabilitySourceIdentity(schema_version="", source_id="official.catalog"),
        lambda: CapabilityLogicalIdentity(
            schema_version="",
            kind=CapabilityKind.TOOL,
            logical_id="tools.rag.search",
        ),
        lambda: CapabilityDiscoveryIdentity(
            schema_version="",
            kind=CapabilityKind.TOOL,
            source=_source(),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.rag.search",
            ),
        ),
        lambda: CapabilityProvenance(schema_version="", source=_source()),
    ],
    ids=[
        "CapabilitySourceIdentity",
        "CapabilityLogicalIdentity",
        "CapabilityDiscoveryIdentity",
        "CapabilityProvenance",
    ],
)
def test_schema_version_rejects_empty_string(model_factory: object) -> None:
    with pytest.raises(ValidationError):
        model_factory()  # type: ignore[operator]


def test_public_import_surface() -> None:
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityStageVocabulary,
    )

    assert CapabilityKind.TOOL.value == "tool"
    assert CapabilityStageVocabulary.DISCOVERED.value == "discovered"
    assert CapabilityDiscoveryIdentity is not None
