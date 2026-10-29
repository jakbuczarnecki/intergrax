# © Artur Czarnecki. All rights reserved.

"""C1 — technical pre-cert evidence export from qual SSOT (no independent PASS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tests.qualification.ebh_6._evidence_manifest import EBH6_CERTIFICATION_HANDOFF_SHA, evidence_git_head_sha
from tests.qualification.ebh_6._l3_aj_group_reconciliation import aj_group_by_layer_class
from tests.qualification.ebh_6._t2_evidence_bundles import all_evidence_bundles, bundle_for_pair
from tests.qualification.ebh_6._tier_audit import tier_audit_rows


@dataclass(frozen=True, slots=True)
class PreCertPairRow:
    layer: str
    detection_class: str
    tier: str
    implementer_outcome: str
    bundle_id: str
    bundle_review: str
    invariant_ref: str
    gate_node_id: str
    primary_behavioral_gates: tuple[str, ...]
    scope_limitation: str
    negative_test_ref: str | None
    group_evidence_refs: tuple[str, ...]
    independent_disposition: str = "PENDING — independent auditor"


@dataclass(frozen=True, slots=True)
class PreCertBundleRow:
    bundle_id: str
    invariant_ref: str
    implementer_review: str
    primary_behavioral_gates: tuple[str, ...]
    scope_limitation: str
    negative_test_ref: str | None
    pair_count: int
    independent_disposition: str = "PENDING — independent auditor"


def pre_cert_pair_rows() -> tuple[PreCertPairRow, ...]:
    groups = aj_group_by_layer_class()
    rows: list[PreCertPairRow] = []
    for audit in tier_audit_rows():
        bundle = bundle_for_pair(audit.layer, audit.detection_class)
        group = groups[(audit.layer, audit.detection_class)]
        rows.append(
            PreCertPairRow(
                audit.layer,
                audit.detection_class,
                audit.tier.value,
                audit.outcome,
                audit.bundle_id,
                audit.bundle_review,
                bundle.invariant_ref,
                audit.gate_node_id,
                bundle.primary_behavioral_gates,
                bundle.scope_limitation,
                bundle.negative_test_ref,
                group.evidence_refs,
            ),
        )
    return tuple(rows)


def pre_cert_bundle_rows() -> tuple[PreCertBundleRow, ...]:
    pairs = pre_cert_pair_rows()
    counts: dict[str, int] = {}
    for row in pairs:
        counts[row.bundle_id] = counts.get(row.bundle_id, 0) + 1
    out: list[PreCertBundleRow] = []
    for bundle in all_evidence_bundles():
        bid = bundle.bundle_id.value
        out.append(
            PreCertBundleRow(
                bid,
                bundle.invariant_ref,
                bundle.implementer_review.value,
                bundle.primary_behavioral_gates,
                bundle.scope_limitation,
                bundle.negative_test_ref,
                counts.get(bid, 0),
            ),
        )
    return tuple(out)


def pre_cert_export_metadata() -> dict[str, Any]:
    return {
        "qualification_head_sha": evidence_git_head_sha(),
        "certification_handoff_substance_sha": EBH6_CERTIFICATION_HANDOFF_SHA,
        "pair_count": len(pre_cert_pair_rows()),
        "bundle_count": len(pre_cert_bundle_rows()),
        "independent_disposition": "PENDING — agent must not issue E01-PRE-01 PASS",
    }


def format_pre_cert_pair_table_markdown(max_rows: int | None = None) -> str:
    lines = [
        "| Layer | Class | Tier | Outcome | Bundle | Gate nodeid | Independent |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = pre_cert_pair_rows()
    if max_rows is not None:
        rows = rows[:max_rows]
    for row in rows:
        gate = row.gate_node_id.replace("|", "\\|")[:80]
        lines.append(
            f"| `{row.layer}` | {row.detection_class} | {row.tier} | {row.implementer_outcome} "
            f"| `{row.bundle_id}` | `{gate}` | **PENDING** |",
        )
    return "\n".join(lines)


def format_pre_cert_bundle_table_markdown() -> str:
    lines = [
        "| Bundle | Review | Pairs | Invariant | Primary gates | Scope limit | Independent |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in pre_cert_bundle_rows():
        gates = "; ".join(row.primary_behavioral_gates[:2])
        if len(row.primary_behavioral_gates) > 2:
            gates += f" (+{len(row.primary_behavioral_gates) - 2})"
        lines.append(
            f"| `{row.bundle_id}` | {row.implementer_review} | {row.pair_count} "
            f"| {row.invariant_ref[:48]}… | {gates[:60]}… | {row.scope_limitation[:40]}… | **PENDING** |",
        )
    return "\n".join(lines)
