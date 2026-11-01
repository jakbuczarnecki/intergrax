# © Artur Czarnecki. All rights reserved.

import json
from pathlib import Path

import pytest
from testing_support.change_safety.assess.deployment_safety import DeployFacts
from testing_support.change_safety.assess.triggers import build_baseline_manifest, compute_actual_digests
from testing_support.change_safety.assess.workflow_evolution import WorkflowChange
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.ports import ChangeSet
from testing_support.change_safety.selection.selector import SelectionInputs
from testing_support.change_safety.workflow import FAIL, PASS, SKIPPED, STAGE_NAMES, run_change_workflow
from tests.qualification.evol_x.adapters.fixture_tree import inventory_for, write_tree

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_TREE = {
    "applications/app_a/y.py": "V = 1\n",
    "applications/app_a/x.py": "from applications.app_a.y import V\n",
    "applications/app_b/solo.py": "S = 1\n",
    "applications/app_c/leak.py": "from applications.app_a.y import V\n",
}
_NODE = "tests/unit/a.py::test_a"


def _factory(impact):
    return SelectionInputs(impact=impact, catalog_node_ids=frozenset({_NODE}), mandatory_node_ids=frozenset({_NODE}))


def _run(tmp_path: Path, changed: str, *, with_baseline: bool = True, **kwargs):
    inv = inventory_for(write_tree(tmp_path, _TREE))
    baseline = (
        build_baseline_manifest(accepted_at_sha="a" * 40, digests=compute_actual_digests(inv))
        if with_baseline
        else None
    )
    return run_change_workflow(
        change_set=ChangeSet(paths=frozenset({changed}), base_sha="b" * 40, head_sha="c" * 40),
        inventory=inv,
        selection_inputs_factory=_factory,
        owner_rules=load_evol_x_data_document("owner_rules.v1.json"),
        exclusion_register=load_evol_x_data_document("exclusion_register.v1.json"),
        baseline=baseline,
        **kwargs,
    )


def test_pipeline_has_all_eight_stages_in_order(tmp_path: Path) -> None:
    report = _run(tmp_path, "applications/app_b/solo.py")
    assert [(s.stage_id, s.name) for s in report.stages] == list(STAGE_NAMES)
    assert report.stage("A4").status == SKIPPED
    assert report.exit_code == 0, [(s.stage_id, s.detail) for s in report.stages if s.status == FAIL]
    assert report.stage("A7").payload["schema_version"] == "evol_change_dossier.v1"


def test_cross_app_escape_fails_safety_stage_with_nonzero_exit(tmp_path: Path) -> None:
    report = _run(tmp_path, "applications/app_a/y.py")
    assert report.stage("A6").status == FAIL and "isolation escapes" in report.stage("A6").detail
    assert report.exit_code == 2 and report.failed_stages == ("A6",)


def test_missing_baseline_fails_freshness_never_silently_passes(tmp_path: Path) -> None:
    report = _run(tmp_path, "applications/app_b/solo.py", with_baseline=False)
    assert report.stage("A8").status == FAIL and report.exit_code == 2


def test_unauthorized_core_touch_and_incompatible_contract_fail(tmp_path: Path) -> None:
    inv = inventory_for(write_tree(tmp_path, {**_TREE, "intergrax/runtime/execution/engine.py": "E = 1\n"}))
    baseline = build_baseline_manifest(accepted_at_sha="a" * 40, digests=compute_actual_digests(inv))
    common = {
        "inventory": inv,
        "selection_inputs_factory": _factory,
        "owner_rules": load_evol_x_data_document("owner_rules.v1.json"),
        "exclusion_register": load_evol_x_data_document("exclusion_register.v1.json"),
        "baseline": baseline,
    }
    cs = ChangeSet(paths=frozenset({"intergrax/runtime/execution/engine.py"}), base_sha="b" * 40, head_sha="c" * 40)
    bad = run_change_workflow(change_set=cs, **common)
    assert "unauthorized core surfaces" in bad.stage("A6").detail
    ok = run_change_workflow(change_set=cs, authorized_surfaces=frozenset({"execution"}), **common)
    assert ok.stage("A6").status == PASS
    compat = run_change_workflow(
        change_set=cs,
        authorized_surfaces=frozenset({"execution"}),
        compat=(("contract:x", CompatibilityAtChangeResult.INCOMPATIBLE),),
        **common,
    )
    assert compat.stage("A4").status == FAIL


def test_workflow_and_deploy_refusals_fail_safety_stage(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        "applications/app_b/solo.py",
        workflow_change=WorkflowChange(has_inflight=True, adopt_in_place=True),
        deploy_facts=DeployFacts(operation="rollback", durable_forward_only_applied=True),
    )
    detail = report.stage("A6").detail
    assert "WM-07" in detail and "DM-04" in detail and report.exit_code == 2


def test_cli_end_to_end_on_repo_writes_stage_files(tmp_path: Path) -> None:
    from scripts.qualification import evol_x_change_workflow as cli

    baseline = tmp_path / "baseline.json"
    assert cli.main(["--write-baseline", str(baseline)]) == 0
    code = cli.main(["--base-sha", "HEAD", "--baseline", str(baseline), "--out-dir", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] == code
    assert (code == 0) == (not summary["failed_stages"])
    assert {s["id"]: s["status"] for s in summary["stages"]}["A8"] == PASS  # same-process baseline is fresh
    assert len(list((tmp_path / "out").glob("A*_*.json"))) == 8
