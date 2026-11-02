# © Artur Czarnecki. All rights reserved.

"""W2-07/W2-08: impact oracle semantics + git intake (rename + class)."""

from __future__ import annotations

import pytest
from testing_support.change_safety.contracts.enums import ImpactStatus
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.graph.builder import build_semantic_graph
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import GitChangeSetPort, parse_name_status

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_parse_name_status_rename_yields_old_and_new() -> None:
    out = "R100\told/a.py\tnew/a.py\nM\tb.py\nD\tc.py\nA\td.py\n"
    statuses, renames, paths = parse_name_status(out)
    assert renames == (("old/a.py", "new/a.py"),)
    assert paths == frozenset({"old/a.py", "new/a.py", "b.py", "c.py", "d.py"})
    assert dict(statuses)["old/a.py"] == "D" and dict(statuses)["new/a.py"] == "A"
    assert dict(statuses)["b.py"] == "M"


def test_evol_x_git_change_set_fails_closed_on_bad_sha() -> None:
    with pytest.raises(RuntimeError):
        GitChangeSetPort().load(base_sha="0" * 40, head_sha="1" * 40)


def test_evol_x_impact_graph_digest_is_semantic_not_inventory(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(evol_inventory, changed_paths=frozenset({"intergrax/__init__.py"}))
    assert impact.graph_digest == build_semantic_graph(evol_inventory).digest
    assert impact.graph_digest != impact.inventory_digest
    assert impact.status is ImpactStatus.COMPLETE


def test_evol_x_impact_unknown_path_is_conservative_not_silent_complete(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(
        evol_inventory, changed_paths=frozenset({"intergrax/new_unknown_surface/data.bin", "intergrax/__init__.py"})
    )
    assert impact.status is ImpactStatus.PARTIAL
    assert any(a.node_id == "unknown:path:intergrax/new_unknown_surface/data.bin" for a in impact.affected)


def test_evol_x_impact_unknown_python_path_not_in_inventory_is_unknown(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(evol_inventory, changed_paths=frozenset({"intergrax/zz_deleted_module.py"}))
    assert impact.status is ImpactStatus.PARTIAL


def test_evol_x_impact_excluded_doc_path_is_explicit_not_unknown(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(evol_inventory, changed_paths=frozenset({"docs/project/README.md"}))
    assert impact.status is ImpactStatus.COMPLETE
    assert impact.affected[0].kind == "EXCLUDED" and impact.affected[0].rule_id == "R-EXCLUDED"


def test_evol_x_impact_transitive_consumer_witness_and_tested_by(evol_inventory: ClosedWorldInventory) -> None:
    impact = assess_path_change_impact(
        evol_inventory, changed_paths=frozenset({"testing_support/change_safety/contracts/enums.py"})
    )
    by_id = {a.node_id: a for a in impact.affected}
    transitive = [a for a in impact.affected if a.kind == "TRANSITIVE" and a.node_id.startswith("module:")]
    assert transitive and all(len(a.witness_path) >= 2 for a in transitive)
    suites = [a for a in impact.affected if a.node_id.startswith("suite:")]
    assert suites and all(a.rule_id == "REL-TESTED_BY" for a in suites)
    assert "module:testing_support.change_safety.contracts.enums" in by_id


def test_evol_x_impact_rename_class_intake_marks_both_paths(evol_inventory: ClosedWorldInventory) -> None:
    _statuses, _renames, paths = parse_name_status(
        "R090\ttesting_support/change_safety/contracts/enums.py\ttesting_support/change_safety/contracts/enums_new.py\n"
    )
    impact = assess_path_change_impact(evol_inventory, changed_paths=paths)
    ids = {a.node_id for a in impact.affected}
    assert "module:testing_support.change_safety.contracts.enums" in ids
    assert impact.status is ImpactStatus.PARTIAL  # new path not in inventory -> explicit unknown
