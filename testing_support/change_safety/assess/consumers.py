# © Artur Czarnecki. All rights reserved.

"""Consumer closure + CW-06 / XG-02 second-method (text) oracle for critical (contract) consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import CONSUMPTION_RELS, ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import GraphEdge, module_id_for_path


@dataclass(frozen=True)
class ConsumerIndex:
    schema_version: str
    consumers_by_target: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ConsumerClosureReport:
    schema_version: str
    critical_targets: int
    text_consumer_pairs: int
    omissions: tuple[str, ...]
    documented_non_consumers: int

    @property
    def orphan_count(self) -> int:
        return len(self.omissions)


def build_consumer_index(inventory: ClosedWorldInventory) -> ConsumerIndex:
    mapping: dict[str, set[str]] = {}
    for edge in inventory.edges:
        if edge.rel not in CONSUMPTION_RELS:
            continue
        mapping.setdefault(edge.target_id, set()).add(edge.source_id)
    frozen = {k: tuple(sorted(v)) for k, v in sorted(mapping.items())}
    return ConsumerIndex(schema_version="evol_consumer_closure.v1", consumers_by_target=frozen)


def orphan_import_targets(edges: tuple[GraphEdge, ...], known_module_ids: frozenset[str]) -> tuple[str, ...]:
    orphans: list[str] = []
    for edge in edges:
        if edge.rel != "IMPORTS":
            continue
        if edge.target_id.startswith("module:") and edge.target_id not in known_module_ids:
            orphans.append(edge.target_id)
    return tuple(sorted(set(orphans)))


def _load_non_consumers() -> frozenset[tuple[str, str]]:
    doc: dict[str, Any] = load_evol_x_data_document("critical_consumer_register.v1.json")
    return frozenset((row["consumer"], row["target"]) for row in doc.get("non_consumers", ()))


def _resolve_longest_module(dotted: str, module_ids: frozenset[str]) -> str | None:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        mid = "module:" + ".".join(parts[:i])
        if mid in module_ids:
            return mid
    return None


def critical_text_pairs(inventory: ClosedWorldInventory) -> tuple[tuple[str, str], ...]:
    """Resolved (consumer, contract-target) pairs found by the text method (self/package refs excluded)."""
    modules = inventory.module_ids
    pairs: set[tuple[str, str]] = set()
    for path, refs in inventory.contract_text_refs:
        src = module_id_for_path(path)
        for ref in refs:
            target = _resolve_longest_module(ref, modules)
            if target is None:
                continue
            s_d, t_d = src[len("module:") :], target[len("module:") :]
            if src == target or s_d.startswith(t_d + ".") or t_d.startswith(s_d + "."):
                continue
            pairs.add((src, target))
    return tuple(sorted(pairs))


def critical_consumer_closure(inventory: ClosedWorldInventory) -> ConsumerClosureReport:
    """Second method: regex text scan (comments/docstrings stripped) vs graph consumption edges.

    A text reference to a contract module in code position that has no graph consumption edge from the same
    consumer is a **critical omission** (CW-06 / XG-02) unless explicitly registered as NON_CONSUMER.
    """
    non_consumers = _load_non_consumers()
    modules = inventory.module_ids
    pairs_in_graph: set[tuple[str, str]] = {
        (e.source_id, e.target_id) for e in inventory.edges if e.rel in CONSUMPTION_RELS
    }
    omissions: list[str] = []
    pair_total = 0
    documented = 0
    targets: set[str] = set()
    for path, refs in inventory.contract_text_refs:
        src = module_id_for_path(path)
        for ref in refs:
            target = _resolve_longest_module(ref, modules)
            if target is None:
                continue
            src_dotted, tgt_dotted = src[len("module:") :], target[len("module:") :]
            if src == target or src_dotted.startswith(tgt_dotted + ".") or tgt_dotted.startswith(src_dotted + "."):
                continue
            targets.add(target)
            pair_total += 1
            if (src, target) in pairs_in_graph:
                continue
            if (src, target) in non_consumers:
                documented += 1
                continue
            omissions.append(f"{src}->{target}")
    return ConsumerClosureReport(
        schema_version="evol_consumer_closure.v1",
        critical_targets=len(targets),
        text_consumer_pairs=pair_total,
        omissions=tuple(sorted(set(omissions))),
        documented_non_consumers=documented,
    )


def count_critical_consumer_omissions(inventory: ClosedWorldInventory) -> int:
    """XG-02 real oracle (replaces hard-coded 0)."""
    return critical_consumer_closure(inventory).orphan_count
