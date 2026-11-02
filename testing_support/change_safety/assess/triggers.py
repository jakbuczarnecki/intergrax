# © Artur Czarnecki. All rights reserved.

"""W4-07 executable requalification triggers T1..T9 with a freshness counter (passive)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.contracts.verdicts import AcceptedBaselineManifest
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.builder import build_semantic_graph

_DATA_DIGESTS = {
    "owner_rules_digest": "owner_rules.v1.json",
    "exclusion_register_digest": "exclusion_register.v1.json",
    "frozen_core_registry_digest": "frozen_core_registry.v1.json",
}


@dataclass(frozen=True)
class FreshnessVerdict:
    allowed: bool
    stale_triggers: tuple[str, ...]
    evaluated: int = 0  # freshness counter: triggers evaluated
    stale_count: int = 0  # freshness counter: triggers fired


def trigger_table() -> tuple[dict[str, str], ...]:
    doc = load_evol_x_data_document("requalification_triggers.v1.json")
    rows = tuple(doc["triggers"])
    ids = [r["trigger_id"] for r in rows]
    if ids != [f"T{i}" for i in range(1, 10)]:
        raise ValueError(f"requalification triggers must be exactly T1..T9, got {ids}")
    return rows


def compute_actual_digests(inventory: ClosedWorldInventory) -> dict[str, str]:
    from testing_support.change_safety.selection.ci_manifest import load_pinned_manifest
    from testing_support.change_safety.selection.policy import POLICY_ID

    digests: dict[str, str] = {
        "graph_digest": build_semantic_graph(inventory).digest,
        "inventory_digest": inventory.inventory_digest,
        "ruleset_version": inventory.ruleset_version,
        "extractor_digest": inventory.extractor_digest,
        "selection_policy_digest": stable_digest_hex(
            {"policy": POLICY_ID, "exclusions": load_evol_x_data_document("evol_conservative_exclusions.v1.json")}
        ),
        "vl3_manifest_digest": str(load_pinned_manifest().get("manifest_content_digest", "")),
    }
    for key, filename in _DATA_DIGESTS.items():
        digests[key] = stable_digest_hex(load_evol_x_data_document(filename))
    return digests


def build_baseline_manifest(*, accepted_at_sha: str, digests: dict[str, str]) -> AcceptedBaselineManifest:
    return AcceptedBaselineManifest(accepted_at_sha=accepted_at_sha, digests=tuple(sorted(digests.items())))


def evaluate_freshness(baseline: AcceptedBaselineManifest, actual: dict[str, str]) -> FreshnessVerdict:
    """Fail-closed: any trigger whose digest key is missing from either side counts as stale."""
    accepted = dict(baseline.digests)
    stale: list[str] = []
    rows = trigger_table()
    for row in rows:
        key = row["digest_key"]
        if key not in accepted or key not in actual or accepted[key] != actual[key]:
            stale.append(row["trigger_id"])
    return FreshnessVerdict(allowed=not stale, stale_triggers=tuple(stale), evaluated=len(rows), stale_count=len(stale))


def assess_evidence_freshness(
    *,
    expected_graph_digest: str,
    actual_graph_digest: str,
    expected_inventory_digest: str,
    actual_inventory_digest: str,
) -> FreshnessVerdict:
    """Backward-compatible narrow entry point (T1/T2 only)."""
    stale = []
    if expected_graph_digest != actual_graph_digest:
        stale.append("T1")
    if expected_inventory_digest != actual_inventory_digest:
        stale.append("T2")
    return FreshnessVerdict(allowed=not stale, stale_triggers=tuple(stale), evaluated=2, stale_count=len(stale))
