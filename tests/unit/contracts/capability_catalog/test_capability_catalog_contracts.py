# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 1 contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryIdentityConflict,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityStageVocabulary,
    NORMATIVE_CAPABILITY_STAGE_VOCABULARY,
    V1_CAPABILITY_KINDS,
    normalize_discovery_identity_set,
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


def test_normalize_discovery_identity_set_orders_deterministically() -> None:
    first = _identity(logical_id="tools.a")
    second = _identity(logical_id="tools.b")
    normalized = normalize_discovery_identity_set((second, first))
    assert normalized == (first, second)


def test_normalize_discovery_identity_set_fails_closed_on_duplicate() -> None:
    duplicate = _identity()
    with pytest.raises(
        CapabilityDiscoveryIdentityConflict,
        match="duplicate source-qualified discovery identity",
    ):
        normalize_discovery_identity_set((duplicate, duplicate))


def test_public_import_surface() -> None:
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityStageVocabulary,
    )

    assert CapabilityKind.TOOL.value == "tool"
    assert CapabilityStageVocabulary.DISCOVERED.value == "discovered"
    assert CapabilityDiscoveryIdentity is not None
