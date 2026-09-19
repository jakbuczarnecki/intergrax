# © Artur Czarnecki. All rights reserved.

"""EBH-1-R3 — derived §5.2 authority metrics vs §4.1 register + R2 evidence provenance."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.docs._ebh_eac1_peer_authority_registry_support import (
    _COVERAGE_COMPETING_KEY,
    _COVERAGE_PEER_TYPES_KEY,
    _COVERAGE_SINGLE_OWNER_KEY,
    _COVERAGE_SUBORDINATE_KEY,
    count_peer_authorities,
    count_peer_types_with_competing_canonical_owner,
    count_peer_types_with_single_canonical_owner,
    count_subordinate_authorities,
    parse_declared_peer_authority_count,
    parse_peer_authority_coverage_metrics,
    parse_peer_authority_register,
    parse_subordinate_authority_register,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EBH1_R2 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "audits"
    / "EBH-1-R2_PEER_AUTHORITY_REGISTRY_COUNT_AND_MECHANICAL_INTEGRITY_CLOSURE.md"
)

_CONTEXT_ASSEMBLY = "CONTEXT_ASSEMBLY_AUTHORITY"
_PRINCIPAL_SCOPED = "PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY"
_R2_EVIDENCE_SHA = "31a51b1dd8c53bf63719f0bc5a02f8d89d97ef2b"


def _peer_rows():
    return parse_peer_authority_register()


def _coverage_metrics() -> dict[str, int]:
    return parse_peer_authority_coverage_metrics()


def test_peer_authority_count_metric_matches_registry() -> None:
    rows = _peer_rows()
    derived = count_peer_authorities(rows)
    declared_count = parse_declared_peer_authority_count()
    coverage = _coverage_metrics()
    assert declared_count == derived
    assert coverage[_COVERAGE_PEER_TYPES_KEY] == derived


def test_peer_authority_single_owner_metric_matches_registry() -> None:
    rows = _peer_rows()
    derived = count_peer_types_with_single_canonical_owner(rows)
    peer_total = count_peer_authorities(rows)
    coverage = _coverage_metrics()
    assert derived == peer_total
    assert coverage[_COVERAGE_SINGLE_OWNER_KEY] == derived


def test_peer_authority_competing_owner_metric_matches_registry() -> None:
    rows = _peer_rows()
    derived = count_peer_types_with_competing_canonical_owner(rows)
    coverage = _coverage_metrics()
    assert derived == 0
    assert coverage[_COVERAGE_COMPETING_KEY] == derived


def test_subordinate_authority_metric_matches_registry() -> None:
    subordinates = parse_subordinate_authority_register()
    derived = count_subordinate_authorities(subordinates)
    coverage = _coverage_metrics()
    assert coverage[_COVERAGE_SUBORDINATE_KEY] == derived


def test_r2_artifact_records_exact_evidence_head() -> None:
    text = _EBH1_R2.read_text(encoding="utf-8-sig")
    match = re.search(
        r"\*\*EBH1_R2_EVIDENCE_HEAD\*\*\s*\|\s*`([0-9a-f]{40})`",
        text,
    )
    assert match is not None, "EBH1_R2_EVIDENCE_HEAD not recorded in R2 artifact"
    assert match.group(1) == _R2_EVIDENCE_SHA


def test_r1_contextview_split_remains_intact() -> None:
    owners = {row.authority_type: row.canonical_owner for row in _peer_rows()}
    assert owners.get(_CONTEXT_ASSEMBLY) == "CONTEXT_ENGINEERING"
    assert owners.get(_PRINCIPAL_SCOPED) == "COLLABORATIVE_WORK (MP-5)"
