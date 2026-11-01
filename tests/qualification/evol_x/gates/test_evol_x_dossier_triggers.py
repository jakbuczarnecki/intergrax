# © Artur Czarnecki. All rights reserved.

import ast
from pathlib import Path

import pytest
from testing_support.change_safety.assess.extension_safety import assess_extension_safety
from testing_support.change_safety.assess.isolation import assess_isolation
from testing_support.change_safety.assess.triggers import (
    assess_evidence_freshness,
    build_baseline_manifest,
    compute_actual_digests,
    evaluate_freshness,
    trigger_table,
)
from testing_support.change_safety.assess.workflow_evolution import WorkflowChange, classify_workflow_change
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult
from testing_support.change_safety.diagnostics import dossier as dossier_mod
from testing_support.change_safety.diagnostics.dossier import NOT_ASSESSED, QUESTIONS, build_change_dossier
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.ports import ChangeSet
from tests.qualification.evol_x.adapters.fixture_tree import inventory_for, write_tree

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_TREE = {
    "applications/app_a/y.py": "V = 1\n",
    "applications/app_a/x.py": "from applications.app_a.y import V\n",
}


def test_trigger_table_is_exactly_t1_to_t9() -> None:
    assert [r["trigger_id"] for r in trigger_table()] == [f"T{i}" for i in range(1, 10)]


def test_baseline_fresh_then_each_single_digest_flip_fires_exactly_its_trigger(tmp_path: Path) -> None:
    inv = inventory_for(write_tree(tmp_path, _TREE))
    actual = compute_actual_digests(inv)
    assert set(actual) == {r["digest_key"] for r in trigger_table()}
    baseline = build_baseline_manifest(accepted_at_sha="a" * 40, digests=actual)
    fresh = evaluate_freshness(baseline, actual)
    assert fresh.allowed and fresh.stale_count == 0 and fresh.evaluated == 9
    for row in trigger_table():
        tampered = dict(actual, **{row["digest_key"]: "tampered"})
        verdict = evaluate_freshness(baseline, tampered)
        assert verdict.stale_triggers == (row["trigger_id"],)
        assert not verdict.allowed and verdict.stale_count == 1


def test_missing_digest_is_stale_fail_closed(tmp_path: Path) -> None:
    inv = inventory_for(write_tree(tmp_path, _TREE))
    actual = compute_actual_digests(inv)
    baseline = build_baseline_manifest(
        accepted_at_sha="a" * 40, digests={k: v for k, v in actual.items() if k != "graph_digest"}
    )
    assert evaluate_freshness(baseline, actual).stale_triggers == ("T1",)
    assert evaluate_freshness(build_baseline_manifest(accepted_at_sha="a" * 40, digests=actual), {}).stale_count == 9


def test_legacy_freshness_entry_point() -> None:
    assert assess_evidence_freshness(
        expected_graph_digest="g", actual_graph_digest="g2", expected_inventory_digest="i", actual_inventory_digest="i"
    ).stale_triggers == ("T1",)


def test_dossier_answers_q1_to_q6_and_is_deterministic(tmp_path: Path) -> None:
    inv = inventory_for(write_tree(tmp_path, _TREE))
    changed = frozenset({"applications/app_a/y.py"})
    impact = assess_path_change_impact(inv, changed_paths=changed)
    actual = compute_actual_digests(inv)
    baseline = build_baseline_manifest(accepted_at_sha="a" * 40, digests=actual)
    kwargs = {
        "graph_digest": impact.graph_digest,
        "inventory_digest": inv.inventory_digest,
        "change_set": ChangeSet(
            paths=changed, base_sha="b" * 40, head_sha="c" * 40, statuses=(("applications/app_a/y.py", "M"),)
        ),
        "impact": impact,
        "isolation": assess_isolation(impact, changed_paths=changed),
        "compat": (("contract:x", CompatibilityAtChangeResult.COMPATIBLE),),
        "extension": assess_extension_safety(changed_paths=changed),
        "workflow": classify_workflow_change(WorkflowChange(has_inflight=True)),
        "freshness": evaluate_freshness(baseline, actual),
    }
    first = build_change_dossier(created_at_utc="2026-01-01T00:00:00+00:00", **kwargs)
    second = build_change_dossier(created_at_utc="2030-05-05T00:00:00+00:00", **kwargs)
    assert [s.question_id for s in first.sections] == [q for q, _ in QUESTIONS]
    assert first.content_digest == second.content_digest  # M-09 seed determinism
    assert not first.stale and "FRESH" in first.sections[5].answer
    assert first.sections[2].answer == NOT_ASSESSED  # selection not supplied: explicit, never silent PASS
    assert "M applications/app_a/y.py" in first.sections[0].answer


def test_stale_freshness_marks_dossier_stale(tmp_path: Path) -> None:
    inv = inventory_for(write_tree(tmp_path, _TREE))
    actual = compute_actual_digests(inv)
    baseline = build_baseline_manifest(accepted_at_sha="a" * 40, digests=actual)
    stale = evaluate_freshness(baseline, dict(actual, graph_digest="x"))
    doss = build_change_dossier(graph_digest="x", inventory_digest=inv.inventory_digest, freshness=stale)
    assert doss.stale and doss.stale_triggers == ("T1",)
    assert doss.sections[5].answer.startswith("STALE: T1")


def test_dossier_module_is_passive_no_runtime_imports_no_writes() -> None:
    tree = ast.parse(Path(dossier_mod.__file__).read_text(encoding="utf-8"))
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [None])
    }
    assert not any(m.split(".")[0] in {"intergrax", "agents", "applications"} for m in imported)
    calls = {getattr(n.func, "attr", getattr(n.func, "id", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"open", "write_text", "write_bytes", "mkdir", "unlink", "run", "Popen"}
