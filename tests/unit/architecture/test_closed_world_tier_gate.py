# © Artur Czarnecki. All rights reserved.

"""Negative contract for the TL-FIX-A closed-world tier gate (PT-G01 S10)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import (
    ClosedWorldInventory,
    baseline_accepted_exclusion_ids,
    build_closed_world_inventory,
    build_tier_enforcement_report,
    classify_package_tier,
    collect_unclassified_dependencies,
)
from testing_support.change_safety.discovery.closed_world_gates import run_cw11, run_cw12, run_tier_closed_world_gates
from testing_support.change_safety.discovery.extractors import GraphEdge, reset_fact_cache

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FIXTURE_ROOTS = ("intergrax/",)


def _write_tier_fixture_tree(root: Path) -> None:
    for rel, body in (
        ("intergrax/__init__.py", ""),
        ("intergrax/contracts/__init__.py", ""),
        ("intergrax/contracts/allowed_peer.py", "LABEL = 'contracts'\n"),
        (
            "intergrax/contracts/allowed_import.py",
            "from intergrax.contracts import allowed_peer\n",
        ),
        ("intergrax/applications/__init__.py", ""),
        ("intergrax/applications/shadow.py", "SHADOW = 1\n"),
        (
            "intergrax/contracts/forbidden_application_import.py",
            "import intergrax.applications.shadow\n",
        ),
        (
            "intergrax/contracts/unresolved_import.py",
            "import intergrax.future_missing_owner.alpha\n",
        ),
        ("intergrax/unscoped/__init__.py", ""),
        ("intergrax/unscoped/mystery_module.py", "VALUE = 1\n"),
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def _fixture_inventory(root: Path) -> ClosedWorldInventory:
    reset_fact_cache()
    return build_closed_world_inventory(
        source_commit_sha="tier-gate-fixture",
        roots=_FIXTURE_ROOTS,
        include_tests=False,
        use_cache=False,
        fixture_root=root,
    )


@pytest.fixture()
def tier_fixture_inventory(tmp_path: Path) -> ClosedWorldInventory:
    root = tmp_path / "tree"
    _write_tier_fixture_tree(root)
    return _fixture_inventory(root)


def test_forbidden_tier_import_detected(tier_fixture_inventory: ClosedWorldInventory) -> None:
    report = build_tier_enforcement_report(tier_fixture_inventory)
    kinds = {f.kind for f in report.findings}
    assert "FORBIDDEN_TIER_IMPORT" in kinds
    assert any("forbidden_application_import" in f.location for f in report.findings)


def test_unclassified_dependency_must_be_explicit(tier_fixture_inventory: ClosedWorldInventory) -> None:
    assert not collect_unclassified_dependencies(tier_fixture_inventory)
    silent_source = "module:intergrax.contracts.allowed_import"
    silent_target = "intergrax.synthetic.unregistered_dependency"
    trimmed_edges = tuple(e for e in tier_fixture_inventory.edges if e.rel != "DEPENDS_UNKNOWN")
    broken = replace(
        tier_fixture_inventory,
        edges=trimmed_edges,
        unresolved_imports=((silent_source, silent_target),),
    )
    silent = collect_unclassified_dependencies(broken)
    assert silent
    report = build_tier_enforcement_report(broken)
    assert any(f.kind == "UNCLASSIFIED_DEPENDENCY" for f in report.findings)


def test_allowed_existing_contract_import_mechanism(tier_fixture_inventory: ClosedWorldInventory) -> None:
    report = build_tier_enforcement_report(tier_fixture_inventory)
    allowed_hits = [f for f in report.findings if "allowed_import.py" in f.location]
    assert allowed_hits == []


def test_forbidden_exclusion_register_expansion() -> None:
    register = load_evol_x_data_document("exclusion_register.v1.json")
    mutated = dict(register)
    entries = list(mutated.get("entries", ()))
    entries.append(
        {
            "exclusion_id": "X-CW-TIER-GATE-NEG",
            "status": "ACCEPTED",
            "predicate": "GENERATED_UNTRACKED",
            "glob_patterns": ["intergrax/evil_bypass/**"],
        }
    )
    mutated["entries"] = entries
    assert not run_cw12(mutated).passed


def test_two_identical_tier_enforcement_runs_match(tier_fixture_inventory: ClosedWorldInventory) -> None:
    first = build_tier_enforcement_report(tier_fixture_inventory)
    second = build_tier_enforcement_report(tier_fixture_inventory)
    assert first.enforcement_digest == second.enforcement_digest


def test_unscoped_modules_use_longest_prefix_tier_table(tier_fixture_inventory: ClosedWorldInventory) -> None:
    assert classify_package_tier("intergrax/unscoped/mystery_module.py") == "tier0"
    report = build_tier_enforcement_report(tier_fixture_inventory)
    assert not any(f.kind == "UNCLASSIFIED_MODULE" and "unscoped" in f.location for f in report.findings)


def test_in_universe_unknown_import_fails_tier_gate(tier_fixture_inventory: ClosedWorldInventory) -> None:
    src = "module:intergrax.contracts.allowed_import"
    dotted = "agents.synthetic.missing_module"
    unknown_target = f"unknown:import:{dotted}"
    augmented_edges = tier_fixture_inventory.edges + (GraphEdge(src, unknown_target, "DEPENDS_UNKNOWN"),)
    augmented = replace(
        tier_fixture_inventory,
        edges=augmented_edges,
        unresolved_imports=((src, dotted),),
    )
    report = build_tier_enforcement_report(augmented)
    assert any(f.kind == "UNKNOWN_IN_UNIVERSE_IMPORT" for f in report.findings)
    assert not run_cw11(report).passed


def test_cw12_baseline_unchanged_by_live_exclusion_register_edits() -> None:
    baseline = baseline_accepted_exclusion_ids()
    register = load_evol_x_data_document("exclusion_register.v1.json")
    edited = dict(register)
    entries = [dict(entry) for entry in edited.get("entries", ())]
    entries[0]["glob_patterns"] = list(entries[0].get("glob_patterns", ())) + ["intergrax/editorial_only/**"]
    edited["entries"] = entries
    assert baseline_accepted_exclusion_ids() == baseline
    assert run_cw12(edited).passed


def test_tier_closed_world_gate_wrapper_passes_exclusion_baseline(
    tier_fixture_inventory: ClosedWorldInventory,
) -> None:
    exclusion = load_evol_x_data_document("exclusion_register.v1.json")
    gate_report = run_tier_closed_world_gates(tier_fixture_inventory, exclusion)
    cw12 = gate_report.check("CW-12")
    assert cw12 is not None and cw12.passed


def test_tier_gate_fails_on_fixture_violations(tier_fixture_inventory: ClosedWorldInventory) -> None:
    report = build_tier_enforcement_report(tier_fixture_inventory)
    assert not run_cw11(report).passed
