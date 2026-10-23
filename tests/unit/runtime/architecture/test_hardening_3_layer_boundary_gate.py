# © Artur Czarnecki. All rights reserved.

"""HARDENING-3 — contracts must not depend on runtime (one-way layer boundary)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.architecture.contracts_runtime_layer_boundary import (
    CONTRACTS_RUNTIME_IMPORT_ALLOWLIST,
    collect_contract_file_runtime_imports,
    evaluate_contracts_runtime_layer_boundary,
    format_contracts_runtime_layer_boundary_failure,
    reconcile_contracts_runtime_import_allowlist,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax" / "contracts"


def test_hardening_3_contracts_python_sources_utf8_no_bom_and_parseable() -> None:
    failures: list[str] = []
    for path in sorted(_CONTRACTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            failures.append(f"{rel}: UTF-8 BOM at file start")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"{rel}: UTF-8 decode failed ({exc})")
            continue
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{rel}: ast.parse failed ({exc.msg} at line {exc.lineno})")
    assert failures == [], "contract source encoding/parse gate:\n" + "\n".join(failures)


def test_hardening_3_allowlist_entries_currently_import_runtime() -> None:
    stale: list[str] = []
    for rel in sorted(CONTRACTS_RUNTIME_IMPORT_ALLOWLIST):
        path = _REPO_ROOT / rel
        if not collect_contract_file_runtime_imports(path):
            stale.append(rel)
    assert stale == [], "stale layer-boundary allowlist (no runtime import): " + ", ".join(stale)


def test_hardening_3_contracts_do_not_import_runtime_except_allowlist() -> None:
    result = evaluate_contracts_runtime_layer_boundary(_REPO_ROOT)
    assert result.passed, format_contracts_runtime_layer_boundary_failure(result)


def test_hardening_3_self_healing_contracts_have_no_runtime_imports() -> None:
    root = _CONTRACTS_ROOT / "self_healing"
    if not root.is_dir():
        return
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in collect_contract_file_runtime_imports(path):
            violations.append(f"{rel}: {module}")
    assert violations == []


def test_hardening_3_unallowlisted_runtime_import_detected_in_isolated_tree(
    tmp_path: Path,
) -> None:
    contracts = tmp_path / "intergrax" / "contracts"
    contracts.mkdir(parents=True)
    bad = contracts / "bad.py"
    bad.write_text("from intergrax.runtime.execution import x\n", encoding="utf-8")
    reconciliation = reconcile_contracts_runtime_import_allowlist(tmp_path)
    assert reconciliation.unallowlisted_violations == ("intergrax/contracts/bad.py",)


def test_hardening_3_documented_allowlist_permits_runtime_import_in_isolated_tree(
    tmp_path: Path,
) -> None:
    contracts = tmp_path / "intergrax" / "contracts"
    contracts.mkdir(parents=True)
    deferred = contracts / "deferred.py"
    deferred.write_text("import intergrax.runtime.legacy\n", encoding="utf-8")
    reconciliation = reconcile_contracts_runtime_import_allowlist(
        tmp_path,
        allowlist=frozenset({"intergrax/contracts/deferred.py"}),
    )
    assert reconciliation.unallowlisted_violations == ()
    assert reconciliation.stale_allowlist_entries == ()


def test_hardening_3_synthetic_stale_allowlist_entry_detected(tmp_path: Path) -> None:
    contracts = tmp_path / "intergrax" / "contracts"
    contracts.mkdir(parents=True)
    clean = contracts / "clean.py"
    clean.write_text("# no runtime imports\n", encoding="utf-8")
    stale_entry = "intergrax/contracts/clean.py"
    reconciliation = reconcile_contracts_runtime_import_allowlist(
        tmp_path,
        allowlist=frozenset({stale_entry}),
    )
    assert stale_entry in reconciliation.stale_allowlist_entries


def test_hardening_3_qual_x_reconciliation_uses_same_allowlist_ssot() -> None:
    from tests.qualification.qual_x._p3_inventory_integrity_support import (
        reconcile_hardening_3_contract_runtime_import_allowlist,
    )

    direct = reconcile_contracts_runtime_import_allowlist(_REPO_ROOT)
    via_qual_x = reconcile_hardening_3_contract_runtime_import_allowlist(_REPO_ROOT)
    assert direct.unallowlisted_violations == tuple(via_qual_x[0])
    assert direct.stale_allowlist_entries == tuple(via_qual_x[1])
