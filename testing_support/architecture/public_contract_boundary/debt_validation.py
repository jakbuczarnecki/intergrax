# © Artur Czarnecki. All rights reserved.

"""Startup validation for the EBH-2A public contract dependency debt registry."""

from __future__ import annotations

from testing_support.architecture.public_contract_boundary.models import (
    ContractDependencyDebtEntry,
)


def _semantic_edge_key(entry: ContractDependencyDebtEntry) -> tuple[str, str, str]:
    return (
        entry.source_module,
        entry.forbidden_import_module,
        entry.rule_id.value,
    )


def validate_debt_registry(
    debt_entries: tuple[ContractDependencyDebtEntry, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    seen_finding_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    for entry in debt_entries:
        if not entry.finding_id.strip():
            errors.append("Debt entry has empty finding_id")
            continue
        if entry.finding_id in seen_finding_ids:
            errors.append(f"Duplicate finding_id {entry.finding_id!r}")
        else:
            seen_finding_ids.add(entry.finding_id)
        if not entry.source_module.strip():
            errors.append(
                f"{entry.finding_id}: empty source_module",
            )
        if not entry.forbidden_import_module.strip():
            errors.append(
                f"{entry.finding_id}: empty forbidden_import_module",
            )
        edge = _semantic_edge_key(entry)
        if edge in seen_edges:
            errors.append(
                f"{entry.finding_id}: duplicate semantic debt edge "
                f"source={entry.source_module!r} "
                f"import={entry.forbidden_import_module!r} "
                f"rule={entry.rule_id.value}",
            )
        else:
            seen_edges.add(edge)
    return tuple(errors)
