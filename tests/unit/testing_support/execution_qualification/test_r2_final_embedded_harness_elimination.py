# © Artur Czarnecki. All rights reserved.

"""R2 Final embedded harness elimination and receipt-based gate regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_R2_FINAL_EMBEDDED_PREDECESSOR_LABELS,
    NPSC5E_R2_FINAL_MANDATORY,
)
from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    NPSC5E_R2_FINAL_ORCHESTRATOR_PATH,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_R2_PROFILE_ID,
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
    npsc5e_r2_final_embedded_predecessor_suite_ids,
    npsc5e_r2_final_semantic_pytest_arguments,
    pytest_arguments_exclude_embedded_harness,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)

from .fake_executor import FakeQualificationSuiteExecutor

_R2_FINAL_MODULE = Path(NPSC5E_R2_FINAL_ORCHESTRATOR_PATH)

_EXPECTED_R2_FINAL_SEMANTIC_TEST_NAMES: frozenset[str] = frozenset(
    {
        "test_canonical_predecessor_shas_recorded",
        "test_schema_constants_frozen",
        "test_persistence_contract_exposes_revision_cas",
        "test_final_normal_resume_process_boundary_revision_increment",
        "test_final_stale_writer_scenario",
        "test_final_authority_narrowing_scenario",
        "test_final_authority_missing_scenario",
        "test_final_policy_deny_scenario",
        "test_final_terminal_scenario",
        "test_final_lineage_mismatch_scenario",
        "test_final_cross_process_resume_scenario",
        "test_final_duplicate_resume_claim_one_winner",
        "test_final_retry_interop_attempt_lifecycle_owned",
        "test_final_hitl_interop_no_checkpoint_self_approval",
        "test_final_child_interop_no_checkpoint_bypass",
        "test_no_second_checkpoint_framework_or_revision_authority",
        "test_r2_surface_no_reflection",
        "test_coordinator_provider_neutral_no_sqlite_leak",
        "test_checkpoint_does_not_rehydrate_authority",
        "test_pre_existing_cancellation_fixture_invalid_persist_gate",
        "test_pre_existing_partial_results_unrelated_to_r2",
        "test_ruff_final_test_no_new_errors",
        "test_pyright_final_test_no_new_errors",
    },
)


def test_r2_final_module_inventory_embedded_harness(repo_root: Path) -> None:
    entries = inventory_embedded_harness_in_module(repo_root / _R2_FINAL_MODULE)
    assert any(e.test_function == "test_mandatory_frozen_suite_passes" for e in entries)


def test_legacy_mandatory_labels_resolve_exactly_once() -> None:
    mandatory_by_label = dict(NPSC5E_R2_FINAL_MANDATORY)
    resolved_by_suite: dict[str, str] = {}
    for label in NPSC5E_R2_FINAL_EMBEDDED_PREDECESSOR_LABELS:
        assert label in mandatory_by_label, label
        targets = mandatory_by_label[label]
        suite_id = suite_id_for_pytest_arguments(normalize_pytest_arguments(targets))
        if suite_id in resolved_by_suite:
            pytest.fail(
                f"duplicate suite_id {suite_id!r} for labels "
                f"{resolved_by_suite[suite_id]!r} and {label!r}",
            )
        resolved_by_suite[suite_id] = label


def test_r2_final_orchestrator_mandatory_labels_match_ssot(repo_root: Path) -> None:
    source = (repo_root / _R2_FINAL_MODULE).read_text(encoding="utf-8")
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
    assert tuple(labels) == NPSC5E_R2_FINAL_EMBEDDED_PREDECESSOR_LABELS


def test_canonical_r2_final_leaf_excludes_embedded_harness() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    suite = compiled.suite_by_id[NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID]
    assert suite.pytest_arguments == npsc5e_r2_final_semantic_pytest_arguments()
    assert pytest_arguments_exclude_embedded_harness(suite.pytest_arguments)
    k_expr = suite.pytest_arguments[suite.pytest_arguments.index("-k") + 1]
    assert pytest_k_expression_excludes_embedded_harness(k_expr)
    assert CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR in suite.pytest_arguments


def test_canonical_r3_profile_r2_final_leaf_excludes_embedded_harness() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    suite = compiled.suite_by_id[NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID]
    assert pytest_arguments_exclude_embedded_harness(suite.pytest_arguments)


def test_r2_final_gate_requires_all_legacy_predecessor_receipts() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in (NPSC5E_R2_PROFILE_ID, NPSC5E_R3_PROFILE_ID):
        compiled = catalog.compile_profile(profile_id)
        branch = "npsc5e-r2" if profile_id == NPSC5E_R2_PROFILE_ID else "npsc5e-r3"
        gate_id = f"{branch}.requires.{NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID}"
        gate = next(g for g in compiled.graph.gates if g.gate_id == gate_id)
        expected = (
            NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
            *npsc5e_r2_final_embedded_predecessor_suite_ids(),
        )
        assert gate.requires == expected


def test_r2_final_predecessor_fail_propagation_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(
        "npsc5e-r3.mandatory.r2-original", QualificationSuiteStatus.FAIL
    )
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-final-gate-fail",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_r2_final_semantic_skip_propagation_fail_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(
        NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
        QualificationSuiteStatus.SKIP,
    )
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-final-skip",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_r2_final_missing_receipt_fails_closed(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=1,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-final-missing",
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
                NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
                "nonexistent.r2-final.predecessor",
            ),
        )


def test_npsc5f_final_no_duplicate_r2_final_predecessor_execution(
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
        run_id="r2-final-dedup",
    )
    run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    for suite_id in compiled.plan.leaf_suite_ids:
        assert fake.invocation_counts.get(suite_id, 0) == 1


def test_r2_final_semantic_test_set_preserved_exactly(repo_root: Path) -> None:
    actual = semantic_test_function_names_in_module(repo_root / _R2_FINAL_MODULE)
    assert actual == _EXPECTED_R2_FINAL_SEMANTIC_TEST_NAMES


def test_r2_final_evidence_gate_all_predecessors_pass(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="r2-final-evidence",
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
            NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
            *npsc5e_r2_final_embedded_predecessor_suite_ids(),
        ),
    )
    assert gate.status is QualificationSuiteStatus.PASS
