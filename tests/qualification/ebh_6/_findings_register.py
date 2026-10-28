# © Artur Czarnecki. All rights reserved.

"""CORE_HARDCODING_FINDINGS_REGISTER — mechanical SSOT (VENDOR_BACKEND spec §2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_5._sqlite_touchpoint_registry import (
    SqliteTouchpointClassification,
    sqlite_touchpoint_by_path,
)
from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._evidence_manifest import evidence_git_head_sha
from tests.qualification.ebh_6._l3_reachability_evidence import mechanism_ids_for_layer
from tests.qualification.ebh_6._layer_evidence import layer_evidence_by_id


class HardcodingDetectionClass(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"  # noqa: E741
    J = "J"


class HardcodingFindingClassification(StrEnum):
    ALLOWED_ADAPTER = "ALLOWED — PROVIDER ADAPTER BOUNDARY"
    ALLOWED_FIXED = "ALLOWED — EXPLICITLY RATIFIED FIXED SUBSTRATE"
    ALLOWED_INTERNAL = "ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM"
    PROHIBITED_CORE = "PROHIBITED — CORE VENDOR HARDCODING"
    PROHIBITED_SHADOW = "PROHIBITED — SHADOW CONFIGURATION/SELECTION"
    PROHIBITED_FALLBACK = "PROHIBITED — SILENT FALLBACK"
    PROHIBITED_LEAKAGE = "PROHIBITED — VENDOR-SPECIFIC CONTRACT LEAKAGE"
    REQUIRES_ADR = "REQUIRES ARCHITECTURE DECISION"
    EVIDENCE_REQUIRED = "EVIDENCE REQUIRED"
    DISCOVERY_REQUIRED = "DISCOVERY REQUIRED — L9 LAYER SAMPLE"


class ProductionEffective(StrEnum):
    YES = "yes"
    NO = "no"


class RemediationClass(StrEnum):
    MERGE = "merge"
    MOVE_ADAPTER = "move-to-adapter"
    FAIL_CLOSED = "fail-closed"
    RATIFY_FIXED = "ratify-fixed"
    ADR = "ADR"


class FindingStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    RATIFIED = "RATIFIED"
    REPLAY_REOPENED = "REPLAY-REOPENED"
    L9_EVIDENCE_SUBMITTED = "L9_EVIDENCE_SUBMITTED"


@dataclass(frozen=True, slots=True)
class HardcodingFindingRecord:
    finding_id: str
    layer: str
    detection_class: HardcodingDetectionClass
    classification: HardcodingFindingClassification
    location: str
    production_effective: ProductionEffective
    concern_ids: tuple[str, ...]
    violated_invariant: str
    remediation_class: RemediationClass
    status: FindingStatus
    evidence_sha: str
    notes: str = ""


def _sqlite_classification(
    touch: SqliteTouchpointClassification,
) -> HardcodingFindingClassification:
    mapping = {
        SqliteTouchpointClassification.SANCTIONED_ADAPTER: HardcodingFindingClassification.ALLOWED_ADAPTER,
        SqliteTouchpointClassification.FIXED_SUBSTRATE: HardcodingFindingClassification.ALLOWED_FIXED,
        SqliteTouchpointClassification.INTERNAL: HardcodingFindingClassification.ALLOWED_INTERNAL,
        SqliteTouchpointClassification.REQUIRES_EXPLANATION: HardcodingFindingClassification.REQUIRES_ADR,
    }
    return mapping[touch]


def _sqlite_production_effective(
    touch: SqliteTouchpointClassification,
) -> ProductionEffective:
    if touch == SqliteTouchpointClassification.INTERNAL:
        return ProductionEffective.NO
    return ProductionEffective.YES


def _sqlite_findings() -> tuple[HardcodingFindingRecord, ...]:
    head = evidence_git_head_sha()
    rows: list[HardcodingFindingRecord] = []
    for idx, (path, record) in enumerate(sorted(sqlite_touchpoint_by_path().items()), start=1):
        concern = (record.inventory_concern_id,) if record.inventory_concern_id else ()
        rows.append(
            HardcodingFindingRecord(
                finding_id=f"EBH6-SQLITE-{idx:02d}",
                layer="state_persistence",
                detection_class=HardcodingDetectionClass.E,
                classification=_sqlite_classification(record.classification),
                location=path,
                production_effective=_sqlite_production_effective(record.classification),
                concern_ids=concern,
                violated_invariant="FRZ-STA / FRZ-RPL — SQLite annex accountability",
                remediation_class=RemediationClass.RATIFY_FIXED,
                status=FindingStatus.RATIFIED,
                evidence_sha=head,
                notes=record.notes or "EBH-5 SQLite annex replay @ EBH-6 HEAD",
            ),
        )
    return tuple(rows)


def _layer_primary_detection_class(layer: str) -> HardcodingDetectionClass:
    """Representative class for register row — full A–J matrix in `_layer_evidence.py`."""
    primary: dict[str, HardcodingDetectionClass] = {
        "contracts": HardcodingDetectionClass.A,
        "semantic_core": HardcodingDetectionClass.B,
        "application_services": HardcodingDetectionClass.C,
        "composition_roots": HardcodingDetectionClass.D,
        "governance": HardcodingDetectionClass.E,
        "execution": HardcodingDetectionClass.A,
        "integrations": HardcodingDetectionClass.G,
        "state_persistence": HardcodingDetectionClass.E,
        "reliability_recovery": HardcodingDetectionClass.I,
        "observability_diagnostics": HardcodingDetectionClass.G,
        "events_traceability": HardcodingDetectionClass.D,
        "agent_distribution": HardcodingDetectionClass.B,
        "rag_memory": HardcodingDetectionClass.J,
        "runtime_hosting": HardcodingDetectionClass.D,
        "tools_skills_capabilities": HardcodingDetectionClass.J,
        "background_workers_infra_adapters": HardcodingDetectionClass.H,
    }
    return primary.get(layer, HardcodingDetectionClass.A)


def _layer_coverage_findings() -> tuple[HardcodingFindingRecord, ...]:
    head = evidence_git_head_sha()
    annex = layer_evidence_by_id()
    rows: list[HardcodingFindingRecord] = []
    for idx, layer in enumerate(EBH6_AUDIT_LAYERS, start=1):
        finding_id = f"EBH6-LAYER-{idx:02d}"
        evidence = annex[finding_id]
        det = _layer_primary_detection_class(layer)
        path_count = len(evidence.closed_world_paths)
        mech_ids = mechanism_ids_for_layer(layer)
        rows.append(
            HardcodingFindingRecord(
                finding_id=finding_id,
                layer=layer,
                detection_class=det,
                classification=HardcodingFindingClassification.ALLOWED_INTERNAL,
                location=(
                    "tests/qualification/ebh_6/_l3_reachability_evidence.py — "
                    f"mechanisms={','.join(mech_ids)}; inventory_paths={path_count}"
                ),
                production_effective=ProductionEffective.YES,
                concern_ids=evidence.sys_inv_refs[:2],
                violated_invariant=f"Roadmap §3.0.8 layer `{layer}` — L3 evidence pack (implementer)",
                remediation_class=RemediationClass.MERGE,
                status=FindingStatus.L9_EVIDENCE_SUBMITTED,
                evidence_sha=head,
                notes=(
                    "Implementer L3 pack submitted — not independent RESOLVED; "
                    "closed-world inventory + per-mechanism A–J controls; "
                    "auditor Protocol 03 re-audit required."
                ),
            ),
        )
    return tuple(rows)


_TRACKED_DEBT: Final[tuple[HardcodingFindingRecord, ...]] = (
    HardcodingFindingRecord(
        finding_id="EBH6-DEBT-Q2-D2",
        layer="semantic_core",
        detection_class=HardcodingDetectionClass.F,
        classification=HardcodingFindingClassification.REQUIRES_ADR,
        location="intergrax/runtime/nexus/nexus_loop.py (AuthoritativeDecisionExposure typing seam)",
        production_effective=ProductionEffective.YES,
        concern_ids=("governance.decision_exposure",),
        violated_invariant="FRZ-TYP-03",
        remediation_class=RemediationClass.ADR,
        status=FindingStatus.RATIFIED,
        evidence_sha=evidence_git_head_sha(),
        notes="Tracked typing debt — ratified for EBH-6 wave; EBH-7 may require remediation",
    ),
    HardcodingFindingRecord(
        finding_id="EBH6-DEBT-BND-COMPAT-R3",
        layer="composition_roots",
        detection_class=HardcodingDetectionClass.H,
        classification=HardcodingFindingClassification.ALLOWED_INTERNAL,
        location="applications/debug_lab_nexus_loop.py → RuntimeEventPersistence import",
        production_effective=ProductionEffective.NO,
        concern_ids=(),
        violated_invariant="FRZ-BND-03",
        remediation_class=RemediationClass.FAIL_CLOSED,
        status=FindingStatus.RATIFIED,
        evidence_sha=evidence_git_head_sha(),
        notes="Lab-only reachability — not production-effective",
    ),
    HardcodingFindingRecord(
        finding_id="EBH6-VBN-INVENTORY-REPLAY",
        layer="integrations",
        detection_class=HardcodingDetectionClass.I,
        classification=HardcodingFindingClassification.ALLOWED_INTERNAL,
        location="tests/qualification/ebh_5/_vendor_backend_inventory.py",
        production_effective=ProductionEffective.NO,
        concern_ids=(),
        violated_invariant="FRZ-CFG-06 / FRZ-RPL-01",
        remediation_class=RemediationClass.MERGE,
        status=FindingStatus.RESOLVED,
        evidence_sha=evidence_git_head_sha(),
        notes="EBH-5 inventory unclassified=0 replay @ HEAD; EBH-6 append-only delta",
    ),
)


@lru_cache(maxsize=1)
def all_hardcoding_findings() -> tuple[HardcodingFindingRecord, ...]:
    return _TRACKED_DEBT + _layer_coverage_findings() + _sqlite_findings()


def findings_by_id() -> dict[str, HardcodingFindingRecord]:
    return {row.finding_id: row for row in all_hardcoding_findings()}


def count_unclassified() -> int:
    """Rows lacking explicit vendor classification — not L9 DISCOVERY_REQUIRED layer obligations."""
    return sum(
        1
        for row in all_hardcoding_findings()
        if row.classification == HardcodingFindingClassification.EVIDENCE_REQUIRED
    )


def count_prohibited_unresolved() -> int:
    prohibited = {
        HardcodingFindingClassification.PROHIBITED_CORE,
        HardcodingFindingClassification.PROHIBITED_SHADOW,
        HardcodingFindingClassification.PROHIBITED_FALLBACK,
        HardcodingFindingClassification.PROHIBITED_LEAKAGE,
    }
    return sum(
        1
        for row in all_hardcoding_findings()
        if row.classification in prohibited and row.status not in (FindingStatus.RESOLVED, FindingStatus.RATIFIED)
    )
