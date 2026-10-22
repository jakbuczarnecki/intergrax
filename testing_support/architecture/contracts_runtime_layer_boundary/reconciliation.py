# © Artur Czarnecki. All rights reserved.

"""Bidirectional HARDENING-3 contract→runtime allowlist reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from testing_support.architecture.contracts_runtime_layer_boundary.allowlist import (
    CONTRACTS_RUNTIME_IMPORT_ALLOWLIST,
)
from testing_support.architecture.contracts_runtime_layer_boundary.import_scan import (
    collect_contract_file_runtime_imports,
)


def _allowlist_entry_is_stale(repo_root: Path, rel: str) -> bool:
    path = repo_root / rel
    if not path.is_file():
        return True
    return not collect_contract_file_runtime_imports(path)


@dataclass(frozen=True, slots=True)
class ContractsRuntimeImportAllowlistReconciliation:
    unallowlisted_violations: tuple[str, ...]
    stale_allowlist_entries: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.unallowlisted_violations and not self.stale_allowlist_entries


def reconcile_contracts_runtime_import_allowlist(
    repo_root: Path,
    *,
    allowlist: frozenset[str] | None = None,
) -> ContractsRuntimeImportAllowlistReconciliation:
    effective = allowlist if allowlist is not None else CONTRACTS_RUNTIME_IMPORT_ALLOWLIST
    contracts_root = repo_root / "intergrax" / "contracts"
    unallowlisted: list[str] = []
    for path in sorted(contracts_root.rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if not collect_contract_file_runtime_imports(path):
            continue
        if rel not in effective:
            unallowlisted.append(rel)
    stale = tuple(
        rel
        for rel in sorted(effective)
        if _allowlist_entry_is_stale(repo_root, rel)
    )
    return ContractsRuntimeImportAllowlistReconciliation(
        unallowlisted_violations=tuple(unallowlisted),
        stale_allowlist_entries=stale,
    )
