# © Artur Czarnecki. All rights reserved.

"""T-AUD-01/02 — 133× tier matrix + L3 mechanism → A–J group crosswalk (qual SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_6._aj_class_gate_scan import (
    class_gate_node_id,
    iter_applicable_layer_class_pairs,
    scanner_effective,
)
from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability


class EvidenceTier(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


_T3_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("execution", "B"),
        ("semantic_core", "F"),
    },
)


@dataclass(frozen=True, slots=True)
class TierAuditRow:
    layer: str
    detection_class: str
    tier: EvidenceTier
    outcome: str
    scanner_effective: bool
    gate_node_id: str
    l3_mechanism_ids: tuple[str, ...]


def evidence_tier_for_pair(layer: str, detection_class: str) -> EvidenceTier:
    key = (layer, detection_class)
    if key in _T3_PAIRS:
        return EvidenceTier.T3
    if scanner_effective(layer, detection_class):
        return EvidenceTier.T2
    return EvidenceTier.T1


def _mechanisms_by_layer() -> dict[str, tuple[str, ...]]:
    from tests.qualification.ebh_6._l3_reachability_evidence import all_l3_reachability_records

    by_layer: dict[str, list[str]] = {layer: [] for layer in EBH6_AUDIT_LAYERS}
    for rec in all_l3_reachability_records():
        by_layer.setdefault(rec.layer, []).append(rec.mechanism_id)
    return {layer: tuple(sorted(set(ids))) for layer, ids in by_layer.items()}


@lru_cache(maxsize=1)
def tier_audit_rows() -> tuple[TierAuditRow, ...]:
    from tests.qualification.ebh_6._l3_aj_group_reconciliation import aj_group_by_layer_class

    mechs = _mechanisms_by_layer()
    rows: list[TierAuditRow] = []
    groups = aj_group_by_layer_class()
    for layer, cls in iter_applicable_layer_class_pairs():
        tier = evidence_tier_for_pair(layer, cls)
        group = groups[(layer, cls)]
        rows.append(
            TierAuditRow(
                layer,
                cls,
                tier,
                group.outcome.value,
                scanner_effective(layer, cls),
                class_gate_node_id(layer, cls),
                mechs.get(layer, ()),
            ),
        )
    return tuple(rows)


def tier_census() -> dict[str, int]:
    counts: dict[str, int] = {t.value: 0 for t in EvidenceTier}
    for row in tier_audit_rows():
        counts[row.tier.value] += 1
    return counts


def l3_mechanism_to_aj_groups() -> dict[str, tuple[tuple[str, str], ...]]:
    """T-AUD-02 — each L3 mechanism inherits applicable layer×class pairs."""
    from tests.qualification.ebh_6._l3_reachability_evidence import all_l3_reachability_records

    out: dict[str, list[tuple[str, str]]] = {}
    for rec in all_l3_reachability_records():
        pairs: list[tuple[str, str]] = []
        for aj in layer_aj_applicability(rec.layer):
            if aj.applicable:
                pairs.append((rec.layer, aj.detection_class))
        out[rec.mechanism_id] = tuple(pairs)
    return {mid: tuple(sorted(set(pairs))) for mid, pairs in out.items()}


def format_tier_audit_dossier_section(handoff_sha: str) -> str:
    from tests.qualification.ebh_6._l3_reachability_evidence import all_l3_reachability_records

    census = tier_census()
    rows = tier_audit_rows()
    t1_rows = [r for r in rows if r.tier == EvidenceTier.T1]
    lines = [
        "## Evidence tier audit @ HEAD",
        "",
        f"**CERTIFICATION_HANDOFF_SHA:** `{handoff_sha}`",
        "",
        "| Tier | Groups | Meaning |",
        "| --- | ---: | --- |",
        f"| **T1** (`EVIDENCE_REQUIRED`) | **{census['T1']}** | Mandatory gap — no T2/T3 proof |",
        f"| **T2** (`PARTIAL_CANDIDATE`) | **{census['T2']}** | Reuse gate(s) + effective layer-scoped scanner |",
        f"| **T3** (`PASS_CANDIDATE` / `FINDING`) | **{census['T3']}** | Behavioral/arch gate or documented FINDING |",
        "",
        f"**133** applicable pairs · **{len(all_l3_reachability_records())}** L3 mechanism records · "
        f"crosswalk via `tests/qualification/ebh_6/_tier_audit.py`.",
        "",
    ]
    if t1_rows:
        lines.append("### T1 blockers (must be 0 @ B5 handoff)")
        lines.append("")
        lines.append("| Layer | Class | Gate |")
        lines.append("| --- | --- | --- |")
        for r in t1_rows[:40]:
            lines.append(f"| `{r.layer}` | {r.detection_class} | `{r.gate_node_id}` |")
        if len(t1_rows) > 40:
            lines.append(f"| … | … | +{len(t1_rows) - 40} more |")
        lines.append("")
    else:
        lines.append(
            "**T1 count @ HEAD:** 0 — tier contract satisfied for implementer handoff (B6 still requires P03).",
        )
        lines.append("")
    return "\n".join(lines)


def write_optional_session_tier_audit(path_prefix: str, handoff_sha: str) -> None:
    from pathlib import Path

    dest = Path(path_prefix) / "tier-audit.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(format_tier_audit_dossier_section(handoff_sha) + "\n", encoding="utf-8")
