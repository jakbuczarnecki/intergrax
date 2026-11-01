# © Artur Czarnecki. All rights reserved.

"""W2-05/W2-06: CW-01..CW-10 on HEAD + mutation probes (CW-07) + dual/incremental builds (CW-08/10)."""

from __future__ import annotations

import pytest
from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory, build_closed_world_inventory
from testing_support.change_safety.discovery.closed_world_gates import (
    run_closed_world_gates,
    run_cw01,
    run_cw05,
    run_cw08,
    run_cw09,
    run_cw10,
)
from testing_support.change_safety.discovery.closed_world_probes import (
    PROBES,
    ProbeInputs,
    run_cw07,
    run_mutation_probes,
)
from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_SUBSET_ROOTS = ("agents/", "testing_support/change_safety/")


def _manifest_ids() -> frozenset[str]:
    return frozenset(load_evol_x_data_document("evol_vl3_execution_manifest.v1.json")["pytest_node_ids"])


def _qual_suite_paths() -> frozenset[str]:
    return frozenset(n.split("::", 1)[0] for n in QualXSuiteCatalogAdapter().collectable_node_ids())


def _compat_module_paths() -> frozenset[str]:
    return frozenset(e.module_path.replace(".", "/") + ".py" for e in CONTRACT_SCHEMA_REGISTRY)


@pytest.fixture(scope="module")
def probe_inputs(evol_inventory: ClosedWorldInventory, evol_owner_rules: dict, evol_exclusions: dict) -> ProbeInputs:
    return ProbeInputs(evol_inventory, evol_owner_rules, evol_exclusions, _manifest_ids())


@pytest.mark.parametrize("gate_id", ("CW-01", "CW-02", "CW-03", "CW-04", "CW-06"))
def test_evol_x_closed_world_gate_passes_on_head(
    gate_id: str, evol_inventory: ClosedWorldInventory, evol_owner_rules: dict, evol_exclusions: dict
) -> None:
    report = run_closed_world_gates(evol_inventory, owner_rules=evol_owner_rules, exclusion_register=evol_exclusions)
    check = report.check(gate_id)
    assert check is not None
    assert check.passed, check.detail


def test_evol_x_cw04_x03_x06_x07_proofs_recorded(evol_inventory: ClosedWorldInventory, evol_exclusions: dict) -> None:
    by_id = {e["exclusion_id"]: e for e in evol_exclusions["entries"]}
    assert by_id["X-03"]["status"] == "ACCEPTED" and by_id["X-03"]["proof_gate"] == "CW-04"
    assert by_id["X-06"]["status"] == "IN_SCOPE_CLASSIFIED"
    assert by_id["X-07"]["status"] == "IN_SCOPE_CLASSIFIED"
    assert evol_inventory.k_to_x_edges == ()
    classified = {n.path for n in evol_inventory.nodes if n.category == "module"}
    assert any(p.startswith("platform_proofs/") for p in classified)
    assert any(p.startswith("intergrax/legacy/") for p in classified)


def test_evol_x_cw05_collect_ast_reconciled(evol_inventory: ClosedWorldInventory) -> None:
    check = run_cw05(evol_inventory, collected_node_ids=_manifest_ids())
    assert check.passed, check.detail
    assert check.metrics["ast_tests"] >= 50


def test_evol_x_cw05_fails_closed_without_collect(evol_inventory: ClosedWorldInventory) -> None:
    assert not run_cw05(evol_inventory, collected_node_ids=None).passed


def test_evol_x_cw07_mutation_probes_all_flip(probe_inputs: ProbeInputs) -> None:
    results = run_mutation_probes(probe_inputs)
    assert len(results) == len(PROBES) >= 10
    not_flipped = [r.probe_id for r in results if not r.flipped]
    assert not_flipped == []
    assert run_cw07(probe_inputs).passed


def test_evol_x_cw08_cw10_dual_and_incremental_build_subset() -> None:
    """CW-08: two independent cold builds; CW-10: content-addressed incremental build == cold build."""
    kwargs = {"source_commit_sha": "subset", "roots": _SUBSET_ROOTS, "include_tests": False}
    cold_a = build_closed_world_inventory(use_cache=False, **kwargs)
    cold_b = build_closed_world_inventory(use_cache=False, **kwargs)
    incremental = build_closed_world_inventory(use_cache=True, **kwargs)
    assert run_cw08(cold_a, cold_b).passed
    assert run_cw10(cold_a, incremental).passed
    tampered = cold_a.with_edges(cold_a.edges[:-1])
    assert not run_cw08(cold_a, tampered).passed
    assert not run_cw10(cold_a, tampered).passed


def test_evol_x_cw08_cw10_fail_closed_when_not_supplied(evol_inventory: ClosedWorldInventory) -> None:
    assert not run_cw08(evol_inventory, None).passed
    assert not run_cw10(evol_inventory, None).passed


def test_evol_x_cw09_qual_compat_parity(evol_inventory: ClosedWorldInventory) -> None:
    check = run_cw09(evol_inventory, qual_suite_paths=_qual_suite_paths(), compat_module_paths=_compat_module_paths())
    assert check.passed, check.detail
    bad = run_cw09(
        evol_inventory,
        qual_suite_paths=_qual_suite_paths() | {"tests/qualification/qual_x/test_not_there.py"},
        compat_module_paths=_compat_module_paths(),
    )
    assert not bad.passed


def test_evol_x_full_cw_report_has_ten_gates_and_fails_closed(
    evol_inventory: ClosedWorldInventory, evol_owner_rules: dict, evol_exclusions: dict, probe_inputs: ProbeInputs
) -> None:
    # CW-08/10 inputs: dual/incremental builds are proven independently on a cold subset above; the full-tree
    # cold dual build is performed by scripts/qualification/evol_x_dual_build.py (evidence step), not faked here.
    report = run_closed_world_gates(
        evol_inventory,
        owner_rules=evol_owner_rules,
        exclusion_register=evol_exclusions,
        collected_node_ids=_manifest_ids(),
        qual_suite_paths=_qual_suite_paths(),
        compat_module_paths=_compat_module_paths(),
        cw07_report=run_cw07(probe_inputs),
    )
    ids = [c.gate_id for c in report.checks]
    assert ids == [f"CW-{i:02d}" for i in range(1, 11)]
    failed = {c.gate_id for c in report.checks if not c.passed}
    # CW-08/CW-10 fail closed without independent builds (their positive proof is the subset test above).
    assert failed == {"CW-08", "CW-10"}, failed


def test_evol_x_cw01_negative_unclassified(evol_inventory: ClosedWorldInventory, evol_exclusions: dict) -> None:
    from dataclasses import replace

    bad = replace(evol_inventory, universe_paths=evol_inventory.universe_paths + ("agents/zz_unclassified.py",))
    assert not run_cw01(bad, evol_exclusions).passed
