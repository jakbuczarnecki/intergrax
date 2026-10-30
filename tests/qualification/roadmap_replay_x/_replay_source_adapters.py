# © Artur Czarnecki. All rights reserved.

"""Independent source reads for ROADMAP-REPLAY-X B6 R2 (discovery ≠ consolidated register)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG
from tests.qualification.ebh_6._findings_register import all_hardcoding_findings
from tests.qualification.roadmap_replay_x._formal_reopen_register import section3_checkbox_status
from tests.qualification.roadmap_replay_x._frz_rows import FRZ_REPLAY_ROWS
from tests.qualification.roadmap_replay_x._ledger_classification import _CATEGORY_A_STAGE_IDS
from tests.qualification.roadmap_replay_x._vbn_replay_counter import REPLAY_VBN_ARTIFACT_REL_PATHS

_REPO = Path(__file__).resolve().parents[3]
_ECAP = _REPO / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md"
_CLOSURE_MATRIX = _REPO / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CLOSURE_MATRIX.md"
_ROADMAP = _REPO / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"

_PROTOCOL_GATE_BLOCKER_LABELS: frozenset[str] = frozenset(
    {
        "Independent B6",
        "Operator OD",
        "B6 Protocol 03",
    }
)

_ECAP_COUNTER_NAMES: frozenset[str] = frozenset(
    {
        "reopened_historical_stages",
        "unresolved_replay_blockers",
        "roadmap_drift",
        "historical_evidence_invalidated_by_later_changes",
        "unclassified_replay_findings",
        "vendor_neutrality_replay_failures",
    }
)


@dataclass(frozen=True, slots=True)
class DiscoveredBlockerClaim:
    blocker_id: str
    source: str
    status_hint: str
    evidence_ref: str


@dataclass(frozen=True, slots=True)
class DiscoveredReopenClaim:
    parent_stage_id: str
    source: str
    signal: str


def discover_ecap_counter_claims(ecap_text: str | None = None) -> dict[str, int]:
    text = ecap_text if ecap_text is not None else _ECAP.read_text(encoding="utf-8")
    claims: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name = cells[0].strip("`")
        if name not in _ECAP_COUNTER_NAMES:
            continue
        value_cell = cells[1]
        match = re.search(r"\b(\d+)\b", value_cell)
        if match:
            claims[name] = int(match.group(1))
    return claims


def discover_closure_matrix_blocker_claims(closure_text: str | None = None) -> tuple[DiscoveredBlockerClaim, ...]:
    text = closure_text if closure_text is not None else _CLOSURE_MATRIX.read_text(encoding="utf-8")
    claims: list[DiscoveredBlockerClaim] = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Requirement"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        requirement = cells[0].strip("-* ")
        if not requirement or requirement == "---":
            continue
        if requirement in _PROTOCOL_GATE_BLOCKER_LABELS:
            continue
        blocker_cell = cells[5] if len(cells) > 5 else ""
        blocker_norm = blocker_cell.lower().replace("*", "").strip()
        if blocker_norm in {"", "none", "open", "not closed"}:
            continue
        blocker_id = re.sub(r"[^A-Za-z0-9_-]+", "-", blocker_cell.strip("* ").strip())[:64]
        if not blocker_id:
            continue
        claims.append(
            DiscoveredBlockerClaim(
                blocker_id=blocker_id,
                source="closure_matrix",
                status_hint=blocker_cell,
                evidence_ref="docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CLOSURE_MATRIX.md",
            )
        )
    return tuple(claims)


def discover_ecap_explicit_blocker_ids(ecap_text: str | None = None) -> frozenset[str]:
    text = ecap_text if ecap_text is not None else _ECAP.read_text(encoding="utf-8")
    ids: set[str] = set(re.findall(r"\b(RX-BLK-[A-Z0-9-]+)\b", text))
    ids.update(re.findall(r"\bBLOCKER-(RX-[A-Z0-9-]+)\b", text))
    return frozenset(ids)


def discover_roadmap_category_a_reopen_claims(
    roadmap_text: str | None = None,
) -> tuple[DiscoveredReopenClaim, ...]:
    text = roadmap_text if roadmap_text is not None else _ROADMAP.read_text(encoding="utf-8")
    claims: list[DiscoveredReopenClaim] = []
    for stage_id in sorted(_CATEGORY_A_STAGE_IDS):
        checkbox = section3_checkbox_status(stage_id, text)
        if checkbox is None:
            continue
        upper = checkbox.upper()
        if "REOPEN" in upper or ("BLOCKED" in upper and "CLOSED" not in upper):
            claims.append(
                DiscoveredReopenClaim(
                    parent_stage_id=stage_id,
                    source="roadmap_section3",
                    signal=checkbox,
                )
            )
    return tuple(claims)


def discover_vbn_finding_inventory_ids(repo: Path | None = None) -> frozenset[str]:
    root = _REPO if repo is None else repo
    ids: set[str] = {row.finding_id for row in all_hardcoding_findings()}
    for family in EBH6_FAMILY_CATALOG:
        ids.add(f"ARCH-FAMILY-{family.family_id}")
    for rel in REPLAY_VBN_ARTIFACT_REL_PATHS:
        path = root / rel
        if not path.is_file():
            continue
        for match in re.finditer(r"\b(FRZ-[A-Z0-9-]+)\b", path.read_text(encoding="utf-8")):
            ids.add(match.group(1))
    return frozenset(ids)


def discover_qual_frz_inventory_ids() -> frozenset[str]:
    return frozenset(f"FRZ-{row[0]}" for row in FRZ_REPLAY_ROWS)


def discover_ecap_blocker_register_ids(ecap_text: str | None = None) -> frozenset[str]:
    """ECAP narrative rows that name replay blocker register IDs (independent of Python SSOT)."""
    text = ecap_text if ecap_text is not None else _ECAP.read_text(encoding="utf-8")
    return discover_ecap_explicit_blocker_ids(text)
