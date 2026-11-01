# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any

from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import (
    DISCOVERY_ROOTS,
    _module_id,
)
from testing_support.change_safety.discovery.git_paths import git_ls_files


@dataclass(frozen=True)
class ClosedWorldGateCheck:
    gate_id: str
    passed: bool
    detail: str
    metrics: dict[str, int | str]


@dataclass(frozen=True)
class ClosedWorldGateReport:
    schema_version: str
    checks: tuple[ClosedWorldGateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def check(self, gate_id: str) -> ClosedWorldGateCheck | None:
        for check in self.checks:
            if check.gate_id == gate_id:
                return check
        return None


def _path_excluded(rel: str, exclusion_entries: tuple[dict[str, Any], ...]) -> bool:
    normalized = rel.replace("\\", "/")
    for entry in exclusion_entries:
        if entry.get("status") != "ACCEPTED":
            continue
        for pattern in entry.get("glob_patterns", ()):
            if fnmatch.fnmatch(normalized, pattern):
                return True
    return False


def _owners_for_path(path: str, owners: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    normalized = path.replace("\\", "/")
    matched: list[str] = []
    for owner in owners:
        for prefix in owner.get("paths_prefix", ()):
            p = str(prefix).rstrip("/")
            if normalized == p or normalized.startswith(f"{p}/"):
                matched.append(str(owner["owner_id"]))
    return tuple(sorted(set(matched)))


def _run_cw01(
    inventory: ClosedWorldInventory,
    exclusion_entries: tuple[dict[str, Any], ...],
) -> ClosedWorldGateCheck:
    classified = {n.node_id for n in inventory.nodes if n.node_id.startswith("module:")}
    unclassified: list[str] = []
    for root in DISCOVERY_ROOTS:
        prefix = root.rstrip("/")
        for rel in git_ls_files(prefix):
            if not rel.endswith(".py"):
                continue
            if _path_excluded(rel, exclusion_entries):
                continue
            node_id = _module_id(rel)
            if node_id not in classified:
                unclassified.append(rel)
    passed = len(unclassified) == 0
    detail = "candidates \\ (classified ∪ excluded) = ∅" if passed else f"unclassified={len(unclassified)}"
    return ClosedWorldGateCheck(
        gate_id="CW-01",
        passed=passed,
        detail=detail,
        metrics={"unclassified": len(unclassified)},
    )


def _run_cw02(
    inventory: ClosedWorldInventory,
    owner_rules: dict[str, Any],
) -> ClosedWorldGateCheck:
    owners = tuple(owner_rules.get("owners", ()))
    unowned: list[str] = []
    multi_owner: list[str] = []
    for node in inventory.nodes:
        if not node.node_id.startswith("module:"):
            continue
        rel = node.node_id[len("module:") :].replace(".", "/") + ".py"
        matched = _owners_for_path(rel, owners)
        if not matched:
            unowned.append(rel)
        elif len(matched) > 1:
            multi_owner.append(rel)
    passed = not unowned and not multi_owner
    detail = "exactly one owner per classified module" if passed else f"unowned={len(unowned)} multi={len(multi_owner)}"
    return ClosedWorldGateCheck(
        gate_id="CW-02",
        passed=passed,
        detail=detail,
        metrics={"unowned": len(unowned), "multi_owner": len(multi_owner)},
    )


def _run_cw04(exclusion_entries: tuple[dict[str, Any], ...]) -> ClosedWorldGateCheck:
    open_proof = tuple(
        sorted(
            str(entry["exclusion_id"])
            for entry in exclusion_entries
            if entry.get("status") == "OPEN" and entry.get("requires_w2_proof")
        )
    )
    open_ids = tuple(sorted(str(entry["exclusion_id"]) for entry in exclusion_entries if entry.get("status") == "OPEN"))
    passed = len(open_proof) == 0
    detail = (
        "accepted exclusions registered; OPEN rows tracked for W2 proofs" if open_ids else "exclusion register complete"
    )
    return ClosedWorldGateCheck(
        gate_id="CW-04",
        passed=passed,
        detail=detail,
        metrics={"open_exclusions": len(open_ids), "open_ids": ",".join(open_ids) or "none"},
    )


def run_closed_world_gates(
    inventory: ClosedWorldInventory,
    *,
    owner_rules: dict[str, Any],
    exclusion_register: dict[str, Any],
) -> ClosedWorldGateReport:
    entries = tuple(exclusion_register.get("entries", ()))
    checks = (
        _run_cw01(inventory, entries),
        _run_cw02(inventory, owner_rules),
        _run_cw04(entries),
    )
    return ClosedWorldGateReport(schema_version="evol_closed_world_gates.v1", checks=checks)
