# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.data.loaders import load_evol_x_data_document


@dataclass(frozen=True)
class FreshnessVerdict:
    allowed: bool
    stale_triggers: tuple[str, ...]


def assess_evidence_freshness(
    *,
    expected_graph_digest: str,
    actual_graph_digest: str,
    expected_inventory_digest: str,
    actual_inventory_digest: str,
) -> FreshnessVerdict:
    load_evol_x_data_document("requalification_triggers.v1.json")
    stale: list[str] = []
    if expected_graph_digest != actual_graph_digest:
        stale.append("T-GRAPH-DIGEST")
    if expected_inventory_digest != actual_inventory_digest:
        stale.append("T-INVENTORY-DIGEST")
    return FreshnessVerdict(allowed=not stale, stale_triggers=tuple(stale))
