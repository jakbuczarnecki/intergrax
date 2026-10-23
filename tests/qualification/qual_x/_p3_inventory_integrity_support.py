# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P3 reconciliation helpers (consumers only — owners retain SSOT)."""

from __future__ import annotations

from pathlib import Path

from testing_support.architecture.public_contract_boundary import (
    evaluate_public_contract_dependency_boundary,
)
from testing_support.architecture.public_contract_boundary.debt_registry import (
    PUBLIC_CONTRACT_DEPENDENCY_DEBT,
)
from testing_support.architecture.public_contract_boundary.debt_validation import (
    validate_debt_registry,
)
from tests.qualification.harness_01.catalog import (
    HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES,
    HARNESS_01_RUNTIME_TOOL_INVOKER_REFERENCE_ALLOWLIST,
)
from tests.qualification.harness_01.nexus_import_inventory import (
    HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS,
    HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTERS,
)
from tests.qualification.harness_01.test_harness_01_gates import (
    _collect_higher_layer_nexus_import_files,
    _collect_production_governed_invoker_callsites,
    _collect_runtime_tool_invoker_reference_files,
)
from tests.qualification.qual_x._p2_ingress_catalog import (
    QUAL_X_P2_INGRESS_CATALOG,
    qual_x_p2_all_proof_pytest_node_ids,
)
from tests.qualification.state_x.inventory import (
    CURRENT_STATE_X_FAMILY_IDS,
    HISTORICAL_BASE_FAMILY_IDS,
    STATE_X_FAMILY_INVENTORY,
)
from testing_support.architecture.contracts_runtime_layer_boundary import (
    reconcile_contracts_runtime_import_allowlist,
)


def pytest_node_id_module_path(node_id: str) -> str:
    if "::" not in node_id:
        raise ValueError(f"expected pytest node_id, got {node_id!r}")
    return node_id.split("::", 1)[0]


def missing_proof_module_paths(node_ids: tuple[str, ...], repo_root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        rel = pytest_node_id_module_path(node_id)
        if rel in seen:
            continue
        seen.add(rel)
        if not (repo_root / rel).is_file():
            missing.append(rel)
    return tuple(missing)


def duplicate_node_ids(node_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    dups: list[str] = []
    for node_id in node_ids:
        if node_id in seen:
            dups.append(node_id)
        else:
            seen.add(node_id)
    return tuple(dups)


def reconcile_harness_qinf_runtime_tool_invoker_callsites() -> tuple[list[str], list[str]]:
    hits = _collect_production_governed_invoker_callsites()
    actual = frozenset(hits.keys())
    authorized = HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES
    return sorted(actual - authorized), sorted(authorized - actual)


def reconcile_harness_qinf_runtime_tool_invoker_references() -> tuple[list[str], list[str]]:
    refs = _collect_runtime_tool_invoker_reference_files()
    unclassified = sorted(refs - HARNESS_01_RUNTIME_TOOL_INVOKER_REFERENCE_ALLOWLIST)
    stale = sorted(HARNESS_01_RUNTIME_TOOL_INVOKER_REFERENCE_ALLOWLIST - refs)
    return unclassified, stale


def reconcile_harness_qinf_nexus_importers() -> tuple[list[str], list[str], list[str]]:
    discovered = _collect_higher_layer_nexus_import_files()
    classified = set(HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTERS)
    unclassified = sorted(discovered - classified)
    stale = sorted(classified - discovered)
    by_path = {row.path: row for row in HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS}
    incomplete = sorted(
        path
        for path, row in by_path.items()
        if not (row.reason.strip() and row.owner_layer.strip() and row.evidence.strip())
    )
    return unclassified, stale, incomplete


def reconcile_state_x_family_inventory_ids() -> tuple[list[str], list[str]]:
    inventory_ids = [entry.family_id for entry in STATE_X_FAMILY_INVENTORY]
    missing_mandatory = sorted(set(HISTORICAL_BASE_FAMILY_IDS) - set(inventory_ids))
    stale_ids = []
    if inventory_ids != list(CURRENT_STATE_X_FAMILY_IDS):
        stale_ids.append("CURRENT_STATE_X_FAMILY_IDS diverges from STATE_X_FAMILY_INVENTORY order")
    if len(inventory_ids) != len(set(inventory_ids)):
        stale_ids.append("duplicate family_id in STATE_X_FAMILY_INVENTORY")
    return missing_mandatory, stale_ids


def _dotted_module_to_repo_rel(module: str) -> str:
    return "/".join(module.split(".")) + ".py"


def reconcile_ebh_debt_registry_metadata(
    repo_root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    registry_errors = validate_debt_registry(PUBLIC_CONTRACT_DEPENDENCY_DEBT)
    missing_modules: list[str] = []
    for entry in PUBLIC_CONTRACT_DEPENDENCY_DEBT:
        rel = _dotted_module_to_repo_rel(entry.source_module)
        if not (repo_root / rel).is_file():
            missing_modules.append(rel)
    return registry_errors, tuple(missing_modules)


def ebh_public_contract_boundary_passes(repo_root: Path) -> bool:
    return evaluate_public_contract_dependency_boundary(repo_root).passed


def p2_catalog_all_primary_proof_node_ids() -> tuple[str, ...]:
    ordered: list[str] = []
    for entry in QUAL_X_P2_INGRESS_CATALOG:
        ordered.extend(entry.primary_proof_node_ids)
    return tuple(ordered)


def p2_catalog_proof_integrity(repo_root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = p2_catalog_all_primary_proof_node_ids()
    deduped = qual_x_p2_all_proof_pytest_node_ids()
    return duplicate_node_ids(raw), missing_proof_module_paths(deduped, repo_root)


def reconcile_hardening_3_contract_runtime_import_allowlist(
    repo_root: Path,
) -> tuple[list[str], list[str]]:
    reconciliation = reconcile_contracts_runtime_import_allowlist(repo_root)
    return list(reconciliation.unallowlisted_violations), list(
        reconciliation.stale_allowlist_entries
    )
