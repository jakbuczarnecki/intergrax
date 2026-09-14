# © Artur Czarnecki. All rights reserved.

"""Embedded harness elimination and receipt-based final gate regression tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    CANONICAL_ORCHESTRATOR_PATHS,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_R2_PROFILE_ID,
    NPSC5E_R3_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.embedded_harness_guard import (
    inventory_embedded_harness_in_module,
)
from testing_support.execution_qualification.evidence_provider import (
    PlanRunQualificationEvidenceProvider,
    assert_same_run_evidence,
    evaluate_final_gate_from_evidence,
)
from testing_support.execution_qualification.embedded_harness_kexpr import (
    CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR,
    embedded_harness_test_names,
)
from testing_support.execution_qualification.final_semantic_pytest import (
    NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,
    pytest_arguments_exclude_embedded_harness,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)

from .fake_executor import FakeQualificationSuiteExecutor

_R2_FINAL_MODULE = Path(
    "tests/unit/runtime/architecture/"
    "test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py",
)
_R3_FINAL_MODULE = Path(
    "tests/unit/runtime/architecture/"
    "test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py",
)


def test_t1_inventory_embedded_harness_calls(repo_root: Path) -> None:
    r2_entries = inventory_embedded_harness_in_module(repo_root / _R2_FINAL_MODULE)
    r3_entries = inventory_embedded_harness_in_module(repo_root / _R3_FINAL_MODULE)
    assert any(e.test_function == "test_mandatory_frozen_suite_passes" for e in r2_entries)
    assert any(
        e.test_function == "test_mandatory_frozen_suites_pass_via_parallel_qualification"
        for e in r3_entries
    )
    assert any(e.pattern == "call:run_npsc5e_r3_mandatory_qualification" for e in r3_entries)


@pytest.mark.parametrize("profile_id", (NPSC5E_R2_PROFILE_ID, NPSC5E_R3_PROFILE_ID))
def test_t2_t3_canonical_final_profiles_exclude_embedded_harness(
    profile_id: str,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(profile_id)
    orchestrator_hits: list[str] = []
    for suite_id in compiled.plan.leaf_suite_ids:
        suite = compiled.suite_by_id[suite_id]
        for arg in suite.pytest_arguments:
            if arg in CANONICAL_ORCHESTRATOR_PATHS:
                orchestrator_hits.append(suite_id)
                assert pytest_arguments_exclude_embedded_harness(suite.pytest_arguments), (
                    suite_id
                )
    assert orchestrator_hits, profile_id


def test_canonical_final_profiles_do_not_execute_embedded_qualification_harnesses() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in (NPSC5E_R2_PROFILE_ID, NPSC5E_R3_PROFILE_ID, NPSC5F_FINAL_PROFILE_ID):
        compiled = catalog.compile_profile(profile_id)
        for suite_id in compiled.plan.leaf_suite_ids:
            suite = compiled.suite_by_id[suite_id]
            if any(path in suite.pytest_arguments for path in CANONICAL_ORCHESTRATOR_PATHS):
                assert pytest_arguments_exclude_embedded_harness(suite.pytest_arguments)


def test_t4_t6_final_gate_evidence_all_pass(
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
        run_id="embedded-harness-t4",
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
    assert_same_run_evidence(provider, expected_run_id=config.run_id)
    gate = evaluate_final_gate_from_evidence(
        provider=provider,
        dependency_ids=(NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID,),
    )
    assert gate.status is QualificationSuiteStatus.PASS
    assert result.status is QualificationRunStatus.PASS


def test_t5_one_predecessor_fail_final_gate_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R3_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status("npsc5e-r3.mandatory.r1-final", QualificationSuiteStatus.FAIL)
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="embedded-harness-t5",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_t6_one_predecessor_skip_final_gate_fail(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5E_R2_PROFILE_ID)
    fake = FakeQualificationSuiteExecutor({})
    fake.set_suite_status(NPSC5E_R2_FINAL_SEMANTIC_SUITE_ID, QualificationSuiteStatus.SKIP)
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="embedded-harness-t6",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    assert result.status is QualificationRunStatus.FAIL


def test_t7_missing_receipt_fail_closed(
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
        run_id="embedded-harness-t7",
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
            dependency_ids=("nonexistent.suite",),
        )


def test_t8_wrong_run_identity_fail_closed(
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
        run_id="embedded-harness-t8-a",
    )
    result = run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    provider = PlanRunQualificationEvidenceProvider.from_plan_run(
        result,
        run_id="embedded-harness-t8-b",
    )
    with pytest.raises(Exception, match="run_id mismatch"):
        assert_same_run_evidence(provider, expected_run_id=config.run_id)


def test_t9_no_duplicate_physical_execution(
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
        run_id="embedded-harness-t9",
    )
    run_qualification_execution_plan(
        compiled.plan,
        config,
        coordinator=QualificationCoordinator(executor=fake),
    )
    for suite_id in compiled.plan.leaf_suite_ids:
        assert fake.invocation_counts.get(suite_id, 0) == 1


def test_t10_t11_t12_semantic_and_freeze_tests_not_in_harness_exclusion(repo_root: Path) -> None:
    for module in (_R2_FINAL_MODULE, _R3_FINAL_MODULE):
        source = (repo_root / module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
        harness = embedded_harness_test_names()
        semantic = names - harness
        assert "test_canonical_predecessor_shas_recorded" in semantic
        assert semantic, module.as_posix()


def test_t15_recovery_leaf_excludes_embedded_harness() -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile(NPSC5F_FINAL_PROFILE_ID)
    recovery = compiled.suite_by_id["npsc5f-final.recovery"]
    assert pytest_arguments_exclude_embedded_harness(recovery.pytest_arguments)
    assert CANONICAL_FINAL_EMBEDDED_HARNESS_KEXPR in recovery.pytest_arguments
