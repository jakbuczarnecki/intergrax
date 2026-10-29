# © Artur Czarnecki. All rights reserved.

"""DUP-X enterprise findings register (Python SSOT — mirrored in qualification docs)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class FindingSeverity(StrEnum):
    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"
    GAP = "Gap"


class FindingStatus(StrEnum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    PASS = "PASS"
    RATIFIED = "RATIFIED"
    GAP = "GAP"
    N_A_WITH_EVIDENCE = "N/A WITH EVIDENCE"


class FindingKind(StrEnum):
    DUPLICATE = "DUPLICATE"
    BLOCKER = "BLOCKER"
    SHADOW_AUTHORITY = "shadow_authority"
    BYPASS = "BYPASS"
    PARALLEL_RETAINED = "parallel_retained"
    GAP = "GAP"


@dataclass(frozen=True, slots=True)
class DupXFindingRecord:
    finding_id: str
    severity: FindingSeverity
    dup_classes: tuple[str, ...]
    kind: FindingKind
    status: FindingStatus
    source_paths: tuple[str, ...]
    symbols: tuple[str, ...]
    production_reachability: str
    canonical_owners: tuple[str, ...]
    remediation: str
    notes: str = ""


@dataclass(frozen=True, slots=True)
class DupXCandidateRecord:
    candidate_id: str
    dup_classes: tuple[str, ...]
    status: FindingStatus
    disposition: str
    parallel_retained: bool
    delta_dossier: str | None


DUP_X_AUDITED_HEAD: Final[str] = "8245b7471379414dc24385d1e770ccb56a7b0e2d"
DUP_X_WAVE1_IMPLEMENTATION_HEAD: Final[str] = "2325460ff7f1f0290f95dae03145e30e20e0d991"

_UNKNOWN_REACHABILITY_MARKERS: Final[tuple[str, ...]] = ("unknown",)

DUP_X_FINDING_RECORDS: Final[tuple[DupXFindingRecord, ...]] = (
    DupXFindingRecord(
        finding_id="DUPX-GAP-CLASS-D",
        severity=FindingSeverity.GAP,
        dup_classes=("DUP-10",),
        kind=FindingKind.GAP,
        status=FindingStatus.PASS,
        source_paths=(
            "intergrax/integrations/registry/factory.py",
            "intergrax/integrations/_shared/config.py",
            "intergrax/integrations/execution_bound_integration_resolution.py",
            "intergrax/integrations/existing_capability_configuration_facade.py",
        ),
        symbols=("resolve_from_profile", "read_integration_slug_from_env"),
        production_reachability="production-effective — factory + config canonical; Wave 1 caller matrix",
        canonical_owners=(
            "intergrax/integrations/_shared/config.py",
            "intergrax/integrations/registry/factory.py",
        ),
        remediation="Wave 1 audit closed — OD-DUPX-01 still OPEN for any prod delta",
        notes="DUP_X_CLASS_D_INTEGRATION_SELECTOR_CALLER_MATRIX.md; CAND-03 RATIFIED CANONICAL",
    ),
    DupXFindingRecord(
        finding_id="DUPX-GAP-L3-TIER3-DEDUPE",
        severity=FindingSeverity.GAP,
        dup_classes=("DUP-02",),
        kind=FindingKind.GAP,
        status=FindingStatus.PASS,
        source_paths=("applications/",),
        symbols=("SlackEventDedupeRepository", "wire_slack_companion"),
        production_reachability="LKW Slack product dedupe only — 8/9 hosts N/A",
        canonical_owners=(
            "applications/local_workspace_application/slack_companion/dedupe_repository.py",
            "intergrax/context/policy/exact_dedup.py",
        ),
        remediation="Wave 1 per-app disposition — no platform duplicate",
        notes="DUP_X_TIER3_APPLICATION_HOST_DEDUPE_DISPOSITION.md; CAND-12 N/A WITH EVIDENCE",
    ),
    DupXFindingRecord(
        finding_id="DUPX-F-COMPAT-R3-FENCE",
        severity=FindingSeverity.MAJOR,
        dup_classes=("DUP-08", "DUP-09"),
        kind=FindingKind.BYPASS,
        status=FindingStatus.RATIFIED,
        source_paths=(
            "intergrax/runtime/transport/bootstrap.py",
            "intergrax/runtime/execution/debug_lab_nexus_loop.py",
        ),
        symbols=("build_lab", "RuntimeEventPersistence"),
        production_reachability="lab-only (EBH6-DEBT-BND-COMPAT-R3)",
        canonical_owners=("TRACE-X sanctioned production roots",),
        remediation="Qualification fence — no production import from sanctioned roots",
        notes="Negative gate test_dup_x_compat_r3_not_reachable_from_sanctioned_roots",
    ),
)

DUP_X_CANDIDATE_RECORDS: Final[tuple[DupXCandidateRecord, ...]] = (
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-01",
        dup_classes=("DUP-01", "DUP-09"),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="Legacy deprecated ToolRegistry — zero production imports of tools_base",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-02",
        dup_classes=("DUP-03", "DUP-13"),
        status=FindingStatus.RATIFIED,
        disposition="THIN TYPED ADAPTER — catalog projection",
        parallel_retained=True,
        delta_dossier="DUP_X_RESPONSIBILITY_DELTA_registry_v2.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-03",
        dup_classes=("DUP-10",),
        status=FindingStatus.RATIFIED,
        disposition="CANONICAL — factory + config integration selector (Class D matrix)",
        parallel_retained=False,
        delta_dossier="DUP_X_CLASS_D_INTEGRATION_SELECTOR_CALLER_MATRIX.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-04",
        dup_classes=("DUP-03", "DUP-11", "DUP-14"),
        status=FindingStatus.RATIFIED,
        disposition="Harness assembly projection — read-only",
        parallel_retained=True,
        delta_dossier="DUP_X_RESPONSIBILITY_DELTA_harness_registry_projection.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-05",
        dup_classes=("DUP-04", "DUP-11"),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="Pipeline stages — not duplicate lifecycle owners",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-06",
        dup_classes=("DUP-01", "DUP-05"),
        status=FindingStatus.RATIFIED,
        disposition="THIN wrapper → exact_dedup",
        parallel_retained=True,
        delta_dossier="DUP_X_RESPONSIBILITY_DELTA_context_dedup.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-07",
        dup_classes=("DUP-05",),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="Distinct semantic truths (token vs fragment vs memory)",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-08",
        dup_classes=("DUP-14",),
        status=FindingStatus.PASS,
        disposition="Qual SSOT unified — EBH-5 hints normative union",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-09",
        dup_classes=("DUP-10", "DUP-12"),
        status=FindingStatus.RATIFIED,
        disposition="DISTINCT — integration catalog vs export operator_wiring",
        parallel_retained=False,
        delta_dossier="DUP_X_RESPONSIBILITY_DELTA_observability_export_routing.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-10",
        dup_classes=("DUP-03", "DUP-10"),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="Tenant factory registry — bounded to integration category",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-11",
        dup_classes=("DUP-07", "DUP-12"),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="Diagnostic registries — evidence ports only",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-12",
        dup_classes=("DUP-02",),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="DISTINCT LKW Slack product dedupe — 8/9 Tier-3 hosts N/A",
        parallel_retained=False,
        delta_dossier="DUP_X_TIER3_APPLICATION_HOST_DEDUPE_DISPOSITION.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-13",
        dup_classes=("DUP-01",),
        status=FindingStatus.RATIFIED,
        disposition="DISTINCT bounded lifecycle — ephemeral store",
        parallel_retained=True,
        delta_dossier="DUP_X_RESPONSIBILITY_DELTA_ephemeral_tool_registry_store.md",
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-14",
        dup_classes=("DUP-05", "DUP-10"),
        status=FindingStatus.N_A_WITH_EVIDENCE,
        disposition="SQLite annex accountability — EBH-6 class E replay",
        parallel_retained=False,
        delta_dossier=None,
    ),
    DupXCandidateRecord(
        candidate_id="DUPX-CAND-15",
        dup_classes=("DUP-08", "DUP-09"),
        status=FindingStatus.RATIFIED,
        disposition="COMPAT-R3 lab-only — fenced",
        parallel_retained=False,
        delta_dossier=None,
    ),
)

DUP_CLASSES: Final[tuple[str, ...]] = tuple(f"DUP-{i:02d}" for i in range(1, 15))


def count_unresolved_duplicate_blockers() -> int:
    return sum(
        1
        for row in DUP_X_FINDING_RECORDS
        if row.kind in (FindingKind.DUPLICATE, FindingKind.BLOCKER)
        and row.status not in (FindingStatus.PASS, FindingStatus.RATIFIED, FindingStatus.N_A_WITH_EVIDENCE)
    )


def count_unresolved_shadow_authority() -> int:
    return sum(
        1
        for row in DUP_X_FINDING_RECORDS
        if row.kind == FindingKind.SHADOW_AUTHORITY
        and row.status not in (FindingStatus.PASS, FindingStatus.RATIFIED, FindingStatus.N_A_WITH_EVIDENCE)
    )


def count_open_gaps(records: tuple[DupXFindingRecord, ...] | None = None) -> int:
    rows = DUP_X_FINDING_RECORDS if records is None else records
    return sum(1 for row in rows if row.status == FindingStatus.OPEN)


def count_open_candidates(records: tuple[DupXCandidateRecord, ...] | None = None) -> int:
    rows = DUP_X_CANDIDATE_RECORDS if records is None else records
    return sum(1 for row in rows if row.status == FindingStatus.OPEN)


def findings_with_unknown_reachability(
    records: tuple[DupXFindingRecord, ...] | None = None,
) -> tuple[str, ...]:
    rows = DUP_X_FINDING_RECORDS if records is None else records
    unknown: list[str] = []
    for row in rows:
        lowered = row.production_reachability.lower()
        if any(marker in lowered for marker in _UNKNOWN_REACHABILITY_MARKERS):
            unknown.append(row.finding_id)
    return tuple(unknown)


def count_open_bypass_unresolved() -> int:
    return sum(
        1 for row in DUP_X_FINDING_RECORDS if row.kind == FindingKind.BYPASS and row.status == FindingStatus.OPEN
    )


def count_unclassified_candidates() -> int:
    return sum(1 for row in DUP_X_CANDIDATE_RECORDS if row.status == FindingStatus.GAP)


def retained_parallel_without_dossier() -> tuple[str, ...]:
    missing: list[str] = []
    for row in DUP_X_CANDIDATE_RECORDS:
        if not row.parallel_retained:
            continue
        if not row.delta_dossier:
            missing.append(row.candidate_id)
    return tuple(missing)


def findings_by_id() -> dict[str, DupXFindingRecord]:
    return {row.finding_id: row for row in DUP_X_FINDING_RECORDS}


def candidates_by_id() -> dict[str, DupXCandidateRecord]:
    return {row.candidate_id: row for row in DUP_X_CANDIDATE_RECORDS}


def assert_no_open_gaps() -> None:
    assert count_open_gaps() == 0


def assert_no_open_candidates() -> None:
    assert count_open_candidates() == 0
