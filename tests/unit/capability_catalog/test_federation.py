# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 2 federation tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    CapabilityCatalogConfigurationError,
    CapabilityCatalogEntry,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
    FederatedCapabilityCatalog,
    merge_capability_catalog_entries,
)
from intergrax.capability_catalog.source import CapabilityCatalogSource
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


def _source(source_id: str) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _entry(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    source_id: str = "official.catalog",
    logical_id: str = "tools.echo.ping",
    version_label: str | None = "1.0.0",
    display_label: str | None = "Echo Ping",
) -> CapabilityCatalogEntry:
    source = _source(source_id)
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=version_label,
        ),
        display_label=display_label,
    )


class _StaticSource:
    def __init__(self, source_id: str, entries: tuple[CapabilityCatalogEntry, ...]) -> None:
        self._source_id = source_id
        self._entries = entries
        self.read_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        self.read_calls += 1
        return self._entries


class _FailingSource:
    def __init__(self) -> None:
        self.read_calls = 0

    @property
    def source_id(self) -> str:
        return "zzz.failing"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        self.read_calls += 1
        raise RuntimeError("catalog backend unavailable")


def test_single_source_preserves_entries() -> None:
    entry = _entry()
    federated = FederatedCapabilityCatalog((_StaticSource("official.catalog", (entry,)),))
    snapshot = federated.snapshot()
    assert snapshot.entries == (entry,)
    assert snapshot.source_ids == ("official.catalog",)


def test_multi_source_federation_merges_agent_skill_tool_slices() -> None:
    agent = _entry(
        kind=CapabilityKind.AGENT,
        source_id="agents.official",
        logical_id="agent.dispute_analyst",
    )
    skill = _entry(
        kind=CapabilityKind.SKILL,
        source_id="skills.builtin",
        logical_id="skills.rag.search",
    )
    tool = _entry(
        kind=CapabilityKind.TOOL,
        source_id="tools.builtin",
        logical_id="tools.rag.search",
    )
    federated = FederatedCapabilityCatalog(
        (
            _StaticSource("agents.official", (agent,)),
            _StaticSource("skills.builtin", (skill,)),
            _StaticSource("tools.builtin", (tool,)),
        ),
    )
    snapshot = federated.snapshot()
    assert snapshot.entries == (agent, skill, tool)


def test_provider_order_independence() -> None:
    a = _entry(source_id="source-a", logical_id="tools.a")
    b = _entry(source_id="source-b", logical_id="tools.b")
    c = _entry(source_id="source-c", logical_id="tools.c")
    first = FederatedCapabilityCatalog(
        (
            _StaticSource("source-a", (a,)),
            _StaticSource("source-b", (b,)),
            _StaticSource("source-c", (c,)),
        ),
    ).snapshot()
    second = FederatedCapabilityCatalog(
        (
            _StaticSource("source-c", (c,)),
            _StaticSource("source-a", (a,)),
            _StaticSource("source-b", (b,)),
        ),
    ).snapshot()
    assert first.entries == second.entries
    assert first.source_ids == second.source_ids


def test_same_logical_id_different_source_both_preserved() -> None:
    left = _entry(source_id="source-a", logical_id="tools.shared")
    right = _entry(source_id="source-b", logical_id="tools.shared")
    snapshot = FederatedCapabilityCatalog(
        (
            _StaticSource("source-a", (left,)),
            _StaticSource("source-b", (right,)),
        ),
    ).snapshot()
    assert len(snapshot.entries) == 2
    assert {entry.identity.source.source_id for entry in snapshot.entries} == {
        "source-a",
        "source-b",
    }


def test_exact_duplicate_is_deterministically_deduped() -> None:
    entry = _entry()
    merged = merge_capability_catalog_entries(
        (
            ("source-a", entry),
            ("source-b", entry),
        ),
    )
    assert merged == (entry,)


def test_conflicting_duplicate_identity_fails_closed() -> None:
    base = _entry()
    conflicting = _entry(display_label="Different label")
    with pytest.raises(CapabilityCatalogIdentityConflict):
        merge_capability_catalog_entries(
            (
                ("source-a", base),
                ("source-b", conflicting),
            ),
        )


def test_provenance_round_trip_preserved() -> None:
    entry = _entry(
        source_id="enterprise.private",
        logical_id="tools.secure.scan",
        version_label="2.3.1",
    )
    snapshot = FederatedCapabilityCatalog(
        (_StaticSource("enterprise.private", (entry,)),),
    ).snapshot()
    preserved = snapshot.entries[0]
    assert preserved.provenance.version_label == "2.3.1"
    assert preserved.identity.logical.logical_id == "tools.secure.scan"
    assert preserved.identity.source.source_id == "enterprise.private"


def test_empty_catalog_is_legal_deterministic_snapshot() -> None:
    snapshot = FederatedCapabilityCatalog(
        (_StaticSource("empty.source", ()),),
    ).snapshot()
    assert snapshot.entries == ()
    assert snapshot.source_ids == ("empty.source",)


def test_provider_failure_fails_closed_without_partial_snapshot() -> None:
    healthy = _StaticSource("aaa.healthy", (_entry(source_id="aaa.healthy"),))
    failing = _FailingSource()
    federated = FederatedCapabilityCatalog((healthy, failing))
    with pytest.raises(CapabilityCatalogSourceFailure, match="zzz.failing"):
        federated.snapshot()
    assert healthy.read_calls == 1
    assert failing.read_calls == 1


def test_duplicate_source_ids_rejected_at_construction() -> None:
    source: CapabilityCatalogSource = _StaticSource("dup", ())
    with pytest.raises(CapabilityCatalogConfigurationError, match="duplicate catalog source_id"):
        FederatedCapabilityCatalog((source, source))


def test_empty_source_list_rejected_at_construction() -> None:
    with pytest.raises(CapabilityCatalogConfigurationError, match="at least one source"):
        FederatedCapabilityCatalog(())
