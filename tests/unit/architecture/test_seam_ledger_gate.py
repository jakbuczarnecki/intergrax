# © Artur Czarnecki. All rights reserved.

"""Mechanical seam ledger gate: ledger structure, import rules, and external registries."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.maintenance.seam_ledger_lib import (
    SeamFindingKind,
    classify_import_for_rule,
    collect_external_registry_findings,
    detect_unregistered_cross_boundary_import,
    error_findings,
    load_ledger,
    run_seam_ledger_check,
    validate_ledger_structure,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_LEDGER = _REPO_ROOT / "governance/architecture/seam_ledger.yaml"


def test_canonical_ledger_passes_mechanical_check() -> None:
    errors = error_findings(run_seam_ledger_check(repo_root=_REPO_ROOT, ledger_path=_CANONICAL_LEDGER))
    assert errors == [], "\n".join(f"{e.kind}: {e.message}" for e in errors)


def test_detects_duplicate_ledger_id(tmp_path: Path) -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    rules = list(ledger["import_rules"])
    rules.append(dict(rules[0]))
    ledger["import_rules"] = rules
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.dump(ledger), encoding="utf-8")
    kinds = {f.kind for f in validate_ledger_structure(load_ledger(path), repo_root=_REPO_ROOT)}
    assert SeamFindingKind.DUPLICATE in kinds


def test_detects_owner_mismatch(tmp_path: Path) -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    rules = [dict(r) for r in ledger["import_rules"]]
    rules[0] = {**rules[0], "owner": "T-IGX-000-X"}
    ledger["import_rules"] = rules
    path = tmp_path / "owner.yaml"
    path.write_text(yaml.dump(ledger), encoding="utf-8")
    kinds = {f.kind for f in validate_ledger_structure(load_ledger(path), repo_root=_REPO_ROOT)}
    assert SeamFindingKind.OWNER_MISMATCH in kinds


def test_detects_stale_registry_path(tmp_path: Path) -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    ext = [dict(r) for r in ledger["external_registries"]]
    # inject stale path via closed list in a copy of the registry loaded in-memory only for structure check
    ledger["external_registries"] = ext
    reg_mod = ext[0]["module"]
    sym = ext[0]["symbol"]
    from scripts.maintenance import seam_ledger_lib as lib

    original = lib._import_module_symbol

    closed = list(lib._import_module_symbol(reg_mod, sym))
    closed = list(closed) + [("nonexistent/stale_composition_root.py", "ROOT")]

    def patched(dotted: str, symbol: str):
        if dotted == reg_mod and symbol == sym:
            return closed
        return original(dotted, symbol)

    lib._import_module_symbol = patched  # type: ignore[method-assign]
    try:
        kinds = {
            f.kind
            for f in validate_ledger_structure(ledger, repo_root=_REPO_ROOT)
            if f.rule_id == "closed_provider_roots"
        }
    finally:
        lib._import_module_symbol = original  # type: ignore[method-assign]
    assert SeamFindingKind.STALE_ENTRY in kinds


def test_detects_forbidden_direction_on_root_package_import(tmp_path: Path) -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    rule = next(r for r in ledger["import_rules"] if r["id"] == "agents_no_direct_integrations")
    agent_root = tmp_path / "agents" / "demo_agent"
    agent_root.mkdir(parents=True)
    module = agent_root / "leaky.py"
    module.write_text("from intergrax import integrations\n", encoding="utf-8")
    finding = classify_import_for_rule(
        module="intergrax.integrations",
        source_path="agents/demo_agent/leaky.py",
        rule=rule,
    )
    assert finding is not None
    assert finding.kind is SeamFindingKind.FORBIDDEN_DIRECTION


def test_detects_forbidden_direction_on_agent_provider_import(tmp_path: Path) -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    rule = next(r for r in ledger["import_rules"] if r["id"] == "agents_no_direct_integrations")
    agent_root = tmp_path / "agents" / "demo_agent"
    agent_root.mkdir(parents=True)
    module = agent_root / "leaky.py"
    module.write_text("from intergrax.integrations.providers.foo import Bar\n", encoding="utf-8")
    finding = classify_import_for_rule(
        module="intergrax.integrations.providers.foo",
        source_path="agents/demo_agent/leaky.py",
        rule=rule,
    )
    assert finding is not None
    assert finding.kind is SeamFindingKind.FORBIDDEN_DIRECTION


def test_allowed_contract_leaf_import(tmp_path: Path) -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    rule = next(r for r in ledger["import_rules"] if r["id"] == "agents_no_direct_integrations")
    finding = classify_import_for_rule(
        module="intergrax.integrations.contracts.base",
        source_path="agents/demo_agent/ok.py",
        rule=rule,
    )
    assert finding is not None
    assert finding.kind is SeamFindingKind.ALLOWED


def test_root_package_import_from_expanded_for_seam_scan() -> None:
    from scripts.maintenance.seam_ledger_lib import _extract_imports

    source = "from intergrax import integrations\nfrom intergrax import tools\n"
    assert {module for _line, module in _extract_imports(source)} == {"intergrax.integrations", "intergrax.tools"}


def test_run_seam_ledger_check_reports_missing_seam_outside_scan_root(tmp_path: Path) -> None:
    ledger = {
        "import_rules": [
            {
                "id": "runtime_only",
                "report_missing_seam_outside_scan_roots": True,
                "scan_roots": ["intergrax/runtime"],
                "denied_module_prefixes": ["intergrax.applications"],
            }
        ],
    }
    rule_root = tmp_path / "agents" / "demo"
    rule_root.mkdir(parents=True)
    (rule_root / "leaky.py").write_text("from intergrax.applications import shadow\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.yaml"
    ledger_path.write_text(yaml.dump(ledger), encoding="utf-8")
    errors = error_findings(run_seam_ledger_check(repo_root=tmp_path, ledger_path=ledger_path))
    assert any(e.kind is SeamFindingKind.MISSING_SEAM for e in errors)


def test_detects_missing_seam_rule_for_tracked_namespace() -> None:
    ledger = {
        "import_rules": [
            {
                "id": "runtime_only",
                "report_missing_seam_outside_scan_roots": True,
                "scan_roots": ["intergrax/runtime"],
                "denied_module_prefixes": ["intergrax.applications"],
            }
        ],
    }
    finding = detect_unregistered_cross_boundary_import(
        module="intergrax.applications.foo",
        source_path="agents/x/m.py",
        ledger=ledger,
    )
    assert finding is not None
    assert finding.kind is SeamFindingKind.MISSING_SEAM


def test_external_registry_matches_rw_bnd05_gate() -> None:
    ledger = load_ledger(_CANONICAL_LEDGER)
    errors = error_findings(collect_external_registry_findings(ledger))
    assert errors == []
