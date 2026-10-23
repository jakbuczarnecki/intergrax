# © Artur Czarnecki. All rights reserved.

"""Evaluate HARDENING-3 contracts must not import runtime (except documented allowlist)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from testing_support.architecture.contracts_runtime_layer_boundary.allowlist import (
    CONTRACTS_RUNTIME_IMPORT_ALLOWLIST,
)
from testing_support.architecture.contracts_runtime_layer_boundary.import_scan import (
    collect_contract_file_runtime_imports,
)
from testing_support.architecture.contracts_runtime_layer_boundary.reconciliation import (
    reconcile_contracts_runtime_import_allowlist,
)


@dataclass(frozen=True, slots=True)
class ContractsRuntimeLayerBoundaryGateResult:
    violations: tuple[str, ...]
    stale_allowlist_entries: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations and not self.stale_allowlist_entries


def evaluate_contracts_runtime_layer_boundary(repo_root: Path) -> ContractsRuntimeLayerBoundaryGateResult:
    reconciliation = reconcile_contracts_runtime_import_allowlist(repo_root)
    if reconciliation.stale_allowlist_entries:
        return ContractsRuntimeLayerBoundaryGateResult(
            violations=(),
            stale_allowlist_entries=reconciliation.stale_allowlist_entries,
        )
    contracts_root = repo_root / "intergrax" / "contracts"
    violation_lines: list[str] = []
    for path in sorted(contracts_root.rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        runtime_imports = collect_contract_file_runtime_imports(path)
        if not runtime_imports:
            continue
        if rel in CONTRACTS_RUNTIME_IMPORT_ALLOWLIST:
            continue
        for module in runtime_imports:
            violation_lines.append(f"{rel}: {module}")
    return ContractsRuntimeLayerBoundaryGateResult(
        violations=tuple(violation_lines),
        stale_allowlist_entries=(),
    )


def format_contracts_runtime_layer_boundary_failure(
    result: ContractsRuntimeLayerBoundaryGateResult,
) -> str:
    if result.stale_allowlist_entries:
        return (
            "stale layer-boundary allowlist (no runtime import): "
            + ", ".join(result.stale_allowlist_entries)
        )
    return "contracts → runtime coupling:\n" + "\n".join(result.violations)
