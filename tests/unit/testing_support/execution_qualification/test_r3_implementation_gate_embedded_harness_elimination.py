# © Artur Czarnecki. All rights reserved.

"""R3 implementation gate embedded harness elimination and receipt-based gate regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_R3_FINAL_MANDATORY,
    NPSC5E_R3_IMPLEMENTATION_EMBEDDED_PREDECESSOR_LABELS,
)
from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    NPSC5E_R3_IMPLEMENTATION_ORCHESTRATOR_PATH,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_R3_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
)
from testing_support.execution_qualification.catalog.suite_registry import (
    suite_id_for_pytest_arguments,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.embedded_harness_guard import (
    inventory_embedded_harness_in_module,
    semantic_test_function_names_in_module,
)
from testing_support.execution_qualification.embedded_harness_kexpr import (
    CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
    pytest_k_expression_excludes_embedded_harness,
)
from testing_support.execution_qualification.evidence_provider import (
    PlanRunQualificationEvidenceProvider,
    evaluate_final_gate_from_evidence,
)
from testing_support.execution_qualification.final_semantic_pytest import (
    NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
    NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID,
    npsc5e_r3_implementation_embedded_predecessor_suite_ids,
    npsc5e_r3_implementation_semantic_pytest_arguments,
    pytest_arguments_exclude_embedded_harness,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)

from .fake_executor import FakeQualificationSuiteExecutor

_R3_IMPLEMENTATION_MODULE = Path(NPSC5E_R3_IMPLEMENTATION_ORCHESTRATOR_PATH)

_EXPECTED_R3_IMPLEMENTATION_SEMANTIC_TEST_NAMES: frozenset[str] = frozenset(
    {
        "test_no_forbidden_recovery_runtime_names_in_production",
        "test_no_reflection_in_partial_recovery_production",
        "test_submission_port_exposes_recover_failed_slot",
        "test_one_failed_slot_recovery_preserves_siblings",
        "test_all_success_recovery_is_noop",
        "test_result_order_and_cardinality_preserved_after_recovery",
        "test_cross_process_partial_recovery",
        "test_wrong_revision_blocked",
        "test_policy_deny_blocked",
        "test_stale_recovery_writer_blocked",
        "test_duplicate_recovery_request_idempotent_via_correlation",
        "test_no_direct_child_execution_runner_in_partial_recovery",
        "test_runtime_checkpoint_topology_recovery_field_compatible_v2",
    },
)

_R3_IMPLEMENTATION_LABEL_ALIASES: dict[str, str] = {
    "NPSC-5B": "NPSC-5B Final",
}


def test_r3_implementation_module_inventory_embedded_harness(repo_root: Path) -> None:
    entries = inventory_embedded_harness_in_module(
        repo_root / _R3_IMPLEMENTATION_MODULE
    )
    assert any(e.test_function == "test_mandatory_frozen_suite_passes" for e in entries)


def test_legacy_r3_implementation_predecessor_labels_resolve_exactly_once() -> None:
    mandatory_by_label = dict(NPSC5E_R3_FINAL_MANDATORY)
    resolved_by_suite: dict[str, str] = {}
    for label in NPSC5E_R3_IMPLEMENTATION_EMBEDDED_PREDECESSOR_LABELS:
        canonical_label = _R3_IMPLEMENTATION_LABEL_ALIASES.get(label, label)
        assert canonical_label in mandatory_by_label, label
        targets = mandatory_by_label[canonical_label]
        suite_id = suite_id_for_pytest_arguments(normalize_pytest_arguments(targets))
        if suite_id in resolved_by_suite:
            pytest.fail(
                f"duplicate suite_id {suite_id!r} for labels "
                f"{resolved_by_suite[suite_id]!r} and {label!r}",
            )
        resolved_by_suite[suite_id] = label


def test_r3_implementation_orchestrator_mandatory_labels_match_ssot(
    repo_root: Path,
) -> None:
    source = (repo_root / _R3_IMPLEMENTATION_MODULE).read_text(encoding="utf-8")
    tree = ast.parse(source)
    value: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_MANDATORY_SUITES":
                value = node.value
                break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_MANDATORY_SUITES":
                    value = node.value
                    break
            if value is not None:
                break
    if value is None:
        pytest.fail("_MANDATORY_SUITES not found")
    assert isinstance(value, ast.Tuple)
    labels: list[str] = []
    for elt in value.elts:
        if not isinstance(elt, ast.Tuple) or not elt.elts:
            continue
        label_node = elt.elts[0]
        if isinstance(label_node, ast.Constant) and isinstance(label_node.value, str):
            labels.append(label_node.value)
    assert tuple(labels) == NPSC5E_R3_IMPLEMENTATION_EMBEDDED_PREDECESSOR_LABELS


def test_canonical_r3_implementation_leaf_excludes_embedded_harness() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    suite = compiled.suite_by_id[NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID]
    assert (
        suite.pytest_arguments == npsc5e_r3_implementation_semantic_pytest_arguments()
    )
    assert pytest_arguments_exclude_embedded_harness(suite.pytest_arguments)
    k_expr = suite.pytest_arguments[suite.pytest_arguments.index("-k") + 1]
    assert pytest_k_expression_excludes_embedded_harness(k_expr)
    assert CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR in suite.pytest_arguments
    assert "test_mandatory_frozen_suite_passes" in k_expr


def test_r3_implementation_registry_pytest_args_resolve_to_single_suite_id() -> None:
    suite_id = suite_id_for_pytest_arguments(
        npsc5e_r3_implementation_semantic_pytest_arguments(),
    )
    assert suite_id == NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID


def test_r3_implementation_gate_requires_all_legacy_predecessor_receipts() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    gate_id = f"npsc5e-r3.requires.{NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID}"
    gate = next(g for g in compiled.graph.gates if g.gate_id == gate_id)
    expected = (
        NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID,
        *npsc5e_r3_implementation_embedded_predecessor_suite_ids(),
    )
    assert gate.requires == expected


def test_r2_final_predecessor_resolves_to_semantic_not_full_module() -> None:
    predecessor_ids = npsc5e_r3_implementation_embedded_predecessor_suite_ids()
    assert NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID in predecessor_ids


def test_r3_implementation_predecessor_fail_propagation_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status("npsc5e-r3.mandatory.p0a", QualificationSuiteStatus.FAIL)
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r3-impl-gate-fail",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_r3_implementation_predecessor_skip_fails_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(
        NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID,
        QualificationSuiteStatus.SKIP,
    )
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r3-impl-skip",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_r3_implementation_missing_receipt_fails_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=1,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r3-impl-missing",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    provider = PlanRunQualificationEvidenceProvider.from_plan_run(
        result,
        run_id=config.run_id,
    )
    with pytest.raises(Exception, match="missing receipt"):
        evaluate_final_gate_from_evidence(
            provider=provider,
            dependency_ids=(
                NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID,
                "nonexistent.r3-implementation.predecessor",
            ),
        )


def test_npsc5f_final_no_duplicate_r3_implementation_predecessor_execution(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5F_FINAL_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r3-impl-dedup",
    )
    run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    for suite_id in compiled.plan.leaf_suite_ids:
        assert fake.invocation_counts.get(suite_id, 0) == 1


def test_npsc5e_r3_no_duplicate_r3_implementation_predecessor_execution(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r3-impl-dedup-npsc5e",
    )
    run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    for suite_id in compiled.plan.leaf_suite_ids:
        assert fake.invocation_counts.get(suite_id, 0) == 1


def test_r3_implementation_semantic_test_set_preserved_exactly(repo_root: Path) -> None:
    actual = semantic_test_function_names_in_module(
        repo_root / _R3_IMPLEMENTATION_MODULE
    )
    assert actual == _EXPECTED_R3_IMPLEMENTATION_SEMANTIC_TEST_NAMES


def test_r3_implementation_evidence_gate_all_predecessors_pass(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r3-impl-evidence",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    provider = PlanRunQualificationEvidenceProvider.from_plan_run(
        result,
        run_id=config.run_id,
    )
    gate = evaluate_final_gate_from_evidence(
        provider=provider,
        dependency_ids=(
            NPSC5E_R3_IMPLEMENTATION_SEMANTIC_SUITE_ID,
            *npsc5e_r3_implementation_embedded_predecessor_suite_ids(),
        ),
    )
    assert gate.status is QualificationSuiteStatus.PASS
