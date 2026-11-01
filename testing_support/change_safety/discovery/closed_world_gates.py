# © Artur Czarnecki. All rights reserved.

"""CW-01..CW-10 closed-world gates (pure functions of inventory + registers; no git/IO in gate bodies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from testing_support.change_safety.assess.consumers import critical_consumer_closure
from testing_support.change_safety.discovery.closed_world import (
    ClosedWorldInventory,
    accepted_exclusion_patterns,
    owners_for_path,
    path_matches_patterns,
)


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


def _module_paths(inventory: ClosedWorldInventory) -> frozenset[str]:
    return frozenset(n.path for n in inventory.nodes if n.category == "module" and n.path)


def _entries(register: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(register.get("entries", ()))


def run_cw01(inventory: ClosedWorldInventory, exclusion_register: dict[str, Any]) -> ClosedWorldGateCheck:
    patterns = accepted_exclusion_patterns(_entries(exclusion_register))
    classified = _module_paths(inventory)
    excluded = set(inventory.excluded_paths)
    unclassified = sorted(p for p in inventory.universe_paths if p not in classified and p not in excluded)
    bad_excluded = sorted(p for p in excluded if not path_matches_patterns(p, patterns))
    passed = not unclassified and not bad_excluded
    return ClosedWorldGateCheck(
        "CW-01",
        passed,
        "candidates \\ (classified ∪ excluded) = ∅"
        if passed
        else f"unclassified={len(unclassified)} bad_excluded={len(bad_excluded)}",
        {
            "candidates": len(inventory.universe_paths),
            "unclassified": len(unclassified),
            "bad_excluded": len(bad_excluded),
        },
    )


def run_cw02(inventory: ClosedWorldInventory, owner_rules: dict[str, Any]) -> ClosedWorldGateCheck:
    owners = tuple(owner_rules.get("owners", ()))
    unowned: list[str] = []
    multi: list[str] = []
    for node in inventory.nodes:
        if node.category != "module":
            continue
        matched = owners_for_path(node.path, owners)
        if not matched:
            unowned.append(node.path)
        elif len(matched) > 1:
            multi.append(node.path)
    passed = not unowned and not multi
    return ClosedWorldGateCheck(
        "CW-02",
        passed,
        "exactly one owner per classified module" if passed else f"unowned={len(unowned)} multi={len(multi)}",
        {"unowned": len(unowned), "multi_owner": len(multi)},
    )


def run_cw03(inventory: ClosedWorldInventory) -> ClosedWorldGateCheck:
    """Every IMPORTS target is a classified node; every dangling in-universe import is explicit UNKNOWN."""
    modules = inventory.module_ids
    orphan_targets = sorted({e.target_id for e in inventory.edges if e.rel == "IMPORTS" and e.target_id not in modules})
    unknown_edges = {(e.source_id, e.target_id) for e in inventory.edges if e.rel == "DEPENDS_UNKNOWN"}
    silent = sorted(
        f"{src}->{dotted}"
        for src, dotted in inventory.unresolved_imports
        if (src, f"unknown:import:{dotted}") not in unknown_edges
    )
    passed = not orphan_targets and not silent
    return ClosedWorldGateCheck(
        "CW-03",
        passed,
        "all imports resolve to K∪X or explicit UNKNOWN"
        if passed
        else f"orphans={len(orphan_targets)} silent={len(silent)}",
        {
            "orphan_targets": len(orphan_targets),
            "silent_dangling": len(silent),
            "explicit_unknown_imports": len(inventory.unresolved_imports),
        },
    )


def run_cw04(inventory: ClosedWorldInventory, exclusion_register: dict[str, Any]) -> ClosedWorldGateCheck:
    entries = _entries(exclusion_register)
    open_ids = sorted(str(e["exclusion_id"]) for e in entries if e.get("status") == "OPEN")
    classified = _module_paths(inventory)
    in_scope_missing: list[str] = []
    for entry in entries:
        if entry.get("predicate") != "IN_SCOPE":
            continue
        patterns = tuple(entry.get("glob_patterns", ()))
        in_scope_missing.extend(
            p for p in inventory.universe_paths if path_matches_patterns(p, patterns) and p not in classified
        )
    k_to_x = inventory.k_to_x_edges
    passed = not open_ids and not k_to_x and not in_scope_missing
    return ClosedWorldGateCheck(
        "CW-04",
        passed,
        "exclusion predicates valid (no K→X edge; IN_SCOPE rows classified)"
        if passed
        else f"open={open_ids} k_to_x={len(k_to_x)} in_scope_missing={len(in_scope_missing)}",
        {"open_exclusions": len(open_ids), "k_to_x_edges": len(k_to_x), "in_scope_unclassified": len(in_scope_missing)},
    )


def normalize_collected_test_id(node_id: str) -> str:
    base, _, rest = node_id.partition("::")
    name = rest.split("[", 1)[0]
    return f"{base}::{name}"


def run_cw05(
    inventory: ClosedWorldInventory,
    *,
    collected_node_ids: frozenset[str] | None,
    scope_prefix: str = "tests/qualification/evol_x/",
) -> ClosedWorldGateCheck:
    if collected_node_ids is None:
        return ClosedWorldGateCheck("CW-05", False, "collected node ids not supplied (fail-closed)", {"supplied": 0})
    ast_ids = {
        f"{path}::{name}" for path, names in inventory.test_ids if path.startswith(scope_prefix) for name in names
    }
    collected = {normalize_collected_test_id(n) for n in collected_node_ids if n.startswith(scope_prefix)}
    only_ast = sorted(ast_ids - collected)
    only_collect = sorted(collected - ast_ids)
    passed = not only_ast and not only_collect
    return ClosedWorldGateCheck(
        "CW-05",
        passed,
        "pytest collect ↔ AST reconciled" if passed else f"ast_only={len(only_ast)} collect_only={len(only_collect)}",
        {
            "ast_tests": len(ast_ids),
            "collected_tests": len(collected),
            "ast_only": len(only_ast),
            "collect_only": len(only_collect),
        },
    )


def run_cw06(inventory: ClosedWorldInventory) -> ClosedWorldGateCheck:
    report = critical_consumer_closure(inventory)
    passed = report.orphan_count == 0
    return ClosedWorldGateCheck(
        "CW-06",
        passed,
        "second-method consumer parity" if passed else f"critical_omissions={report.orphan_count}",
        {
            "critical_omissions": report.orphan_count,
            "text_consumer_pairs": report.text_consumer_pairs,
            "documented_non_consumers": report.documented_non_consumers,
        },
    )


def run_cw08(first: ClosedWorldInventory, second: ClosedWorldInventory | None) -> ClosedWorldGateCheck:
    if second is None:
        return ClosedWorldGateCheck("CW-08", False, "second build not supplied (fail-closed)", {"supplied": 0})
    passed = first.inventory_digest == second.inventory_digest
    return ClosedWorldGateCheck(
        "CW-08",
        passed,
        "dual build identical digest" if passed else "dual build digest mismatch",
        {"digest_a": first.inventory_digest, "digest_b": second.inventory_digest},
    )


def run_cw09(
    inventory: ClosedWorldInventory,
    *,
    qual_suite_paths: frozenset[str] | None,
    compat_module_paths: frozenset[str] | None,
) -> ClosedWorldGateCheck:
    if qual_suite_paths is None or compat_module_paths is None:
        return ClosedWorldGateCheck(
            "CW-09", False, "QUAL/COMPAT inventories not supplied (fail-closed)", {"supplied": 0}
        )
    tests = set(inventory.test_paths)
    modules = _module_paths(inventory)
    qual_missing = sorted(p for p in qual_suite_paths if p not in tests)
    compat_missing = sorted(p for p in compat_module_paths if p not in modules)
    passed = not qual_missing and not compat_missing
    return ClosedWorldGateCheck(
        "CW-09",
        passed,
        "parity vs QUAL-X / COMPAT-X inventories (DUP/EBH-7: no machine inventory — see register)"
        if passed
        else f"qual_missing={len(qual_missing)} compat_missing={len(compat_missing)}",
        {
            "qual_suites": len(qual_suite_paths),
            "qual_missing": len(qual_missing),
            "compat_modules": len(compat_module_paths),
            "compat_missing": len(compat_missing),
            "dup_ebh7_inventory": "NOT_MACHINE_READABLE",
        },
    )


def run_cw10(full: ClosedWorldInventory, incremental: ClosedWorldInventory | None) -> ClosedWorldGateCheck:
    if incremental is None:
        return ClosedWorldGateCheck("CW-10", False, "incremental build not supplied (fail-closed)", {"supplied": 0})
    passed = full.inventory_digest == incremental.inventory_digest
    return ClosedWorldGateCheck(
        "CW-10",
        passed,
        "incremental rebuild == full rebuild" if passed else "incremental != full",
        {"full": full.inventory_digest, "incremental": incremental.inventory_digest},
    )


def run_closed_world_gates(
    inventory: ClosedWorldInventory,
    *,
    owner_rules: dict[str, Any],
    exclusion_register: dict[str, Any],
    collected_node_ids: frozenset[str] | None = None,
    second_build: ClosedWorldInventory | None = None,
    incremental_build: ClosedWorldInventory | None = None,
    qual_suite_paths: frozenset[str] | None = None,
    compat_module_paths: frozenset[str] | None = None,
    include_cw07: bool = False,
    cw07_report: ClosedWorldGateCheck | None = None,
) -> ClosedWorldGateReport:
    checks = [
        run_cw01(inventory, exclusion_register),
        run_cw02(inventory, owner_rules),
        run_cw03(inventory),
        run_cw04(inventory, exclusion_register),
        run_cw05(inventory, collected_node_ids=collected_node_ids),
        run_cw06(inventory),
    ]
    if cw07_report is not None:
        checks.append(cw07_report)
    checks.extend(
        [
            run_cw08(inventory, second_build),
            run_cw09(inventory, qual_suite_paths=qual_suite_paths, compat_module_paths=compat_module_paths),
            run_cw10(inventory, incremental_build),
        ]
    )
    return ClosedWorldGateReport(schema_version="evol_closed_world_gates.v2", checks=tuple(checks))
