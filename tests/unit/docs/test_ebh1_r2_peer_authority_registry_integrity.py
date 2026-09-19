# © Artur Czarnecki. All rights reserved.

"""EBH-1-R2 — mechanical integrity gates for EAC-1 peer authority register."""

from __future__ import annotations

import pytest

from tests.unit.docs._ebh_eac1_peer_authority_registry_support import (
    canonical_owner_is_singular,
    parse_declared_peer_authority_count,
    parse_deprecated_authority_labels,
    parse_peer_authority_register,
    parse_subordinate_authority_register,
    peer_authority_type_names,
)

pytestmark = pytest.mark.unit

_CONTEXT_ASSEMBLY = "CONTEXT_ASSEMBLY_AUTHORITY"
_PRINCIPAL_SCOPED = "PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY"


def _owner_map() -> dict[str, str]:
    rows = parse_peer_authority_register()
    return {row.authority_type: row.canonical_owner for row in rows}


def test_peer_authority_registry_declared_count_matches_table() -> None:
    rows = parse_peer_authority_register()
    actual = len(rows)
    declared = parse_declared_peer_authority_count()
    assert declared == actual, (
        f"declared CURRENT_PEER_AUTHORITY_COUNT={declared} != table rows={actual}"
    )


def test_peer_authority_registry_has_unique_authority_types() -> None:
    names = peer_authority_type_names(parse_peer_authority_register())
    assert len(names) == len(set(names)), (
        f"duplicate peer authority types: "
        f"{sorted(n for n in names if names.count(n) > 1)}"
    )


def test_peer_authority_registry_has_single_canonical_owner_per_type() -> None:
    rows = parse_peer_authority_register()
    missing = [
        r.authority_type
        for r in rows
        if not r.canonical_owner.strip() or r.canonical_owner.strip() in {"—", "-", "NONE"}
    ]
    ambiguous = [
        r.authority_type
        for r in rows
        if r.canonical_owner.strip()
        and r.canonical_owner.strip() not in {"—", "-", "NONE"}
        and not canonical_owner_is_singular(r.canonical_owner)
    ]
    assert not missing, f"peer rows with missing canonical owner: {missing}"
    assert not ambiguous, f"peer rows with ambiguous canonical owner: {ambiguous}"


def test_subordinate_authorities_are_not_counted_as_peers() -> None:
    peer_names = set(peer_authority_type_names(parse_peer_authority_register()))
    subordinates = parse_subordinate_authority_register()
    assert len(subordinates) == 1
    overlap = [s.authority_type for s in subordinates if s.authority_type in peer_names]
    assert not overlap, f"subordinate authority counted as peer: {overlap}"


def test_deprecated_authority_labels_are_not_active_peers() -> None:
    deprecated = parse_deprecated_authority_labels()
    active = set(peer_authority_type_names(parse_peer_authority_register()))
    present = sorted(active & deprecated)
    assert not present, f"deprecated labels active in §4.1: {present}"


def test_contextview_authority_split_remains_intact() -> None:
    owners = _owner_map()
    assert owners.get(_CONTEXT_ASSEMBLY) == "CONTEXT_ENGINEERING"
    assert owners.get(_PRINCIPAL_SCOPED) == "COLLABORATIVE_WORK (MP-5)"
    names = peer_authority_type_names(parse_peer_authority_register())
    assert names.count(_CONTEXT_ASSEMBLY) == 1
    assert names.count(_PRINCIPAL_SCOPED) == 1
