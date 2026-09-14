# © Artur Czarnecki. All rights reserved.

"""Global semantic parity certification (T1–T19) for canonical qualification profiles."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from testing_support.execution_qualification import aggregate as aggregate_module
from testing_support.execution_qualification import compiler as compiler_module
from testing_support.execution_qualification import coordinator as coordinator_module
from testing_support.execution_qualification import plan_runner as plan_runner_module
from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.expansion import (
    is_nested_orchestrator_leaf,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    CANONICAL_ORCHESTRATOR_PATHS,
)
from testing_support.execution_qualification.catalog.suite_registry import (
    pytest_to_suite_id_registry,
)
from testing_support.execution_qualification.semantic_parity import (
    GLOBAL_SEMANTIC_PARITY_MATRIX,
    QualificationSemanticParityCertifier,
    SemanticParityCertificationStatus,
)
from testing_support.execution_qualification.semantic_parity.matrix import (
    GLOBAL_SEMANTIC_PARITY_MATRIX as MATRIX,
)


@pytest.fixture
def parity_certifier(
    repo_root: Path,
    run_artifact_root: Path,
) -> QualificationSemanticParityCertifier:
    catalog = build_default_qualification_catalog()
    return QualificationSemanticParityCertifier(
        catalog,
        repo_root=repo_root,
        run_artifact_root=run_artifact_root / "global-semantic-parity",
    )


def test_global_semantic_parity_certification_passes(
    parity_certifier: QualificationSemanticParityCertifier,
) -> None:
    report = parity_certifier.certify_all(GLOBAL_SEMANTIC_PARITY_MATRIX)
    assert report.overall_status is SemanticParityCertificationStatus.PASS
    for result in report.profile_results:
        assert result.profile_pass, (
            result.profile_id,
            result.invariant_failures,
        )


def test_matrix_covers_all_catalog_profiles() -> None:
    catalog = build_default_qualification_catalog()
    matrix_ids = {case.profile_id for case in MATRIX}
    assert matrix_ids == set(catalog.profile_ids)


def test_t1_t4_coverage_and_reachability_via_certifier(
    parity_certifier: QualificationSemanticParityCertifier,
) -> None:
    report = parity_certifier.certify_all(MATRIX)
    for result in report.profile_results:
        assert result.coverage_parity
        assert result.reachability_parity


def test_t5_t11_execution_and_determinism(
    parity_certifier: QualificationSemanticParityCertifier,
) -> None:
    report = parity_certifier.certify_all(MATRIX)
    for result in report.profile_results:
        assert result.all_pass_parity
        assert result.failure_injection_parity
        assert result.skip_injection_parity
        assert result.determinism_parity
        assert result.deterministic


def test_t6_t9_injection_and_receipts(
    parity_certifier: QualificationSemanticParityCertifier,
) -> None:
    report = parity_certifier.certify_all(MATRIX)
    for result in report.profile_results:
        assert len(result.failure_rows) == result.leaf_count
        assert len(result.skip_rows) == result.leaf_count
        assert result.collect_all_parity
        assert result.receipt_parity


def test_t12_no_orchestrator_leaves() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        compiled = catalog.compile_profile(profile_id)
        for suite_id in compiled.plan.leaf_suite_ids:
            suite = compiled.suite_by_id[suite_id]
            assert not is_nested_orchestrator_leaf(suite.pytest_arguments), profile_id


def test_t13_t14_catalog_dependency_guards() -> None:
    catalog_root = Path("testing_support/execution_qualification/catalog")
    forbidden = (
        "testing_support.npsc5f_r4_regression_matrix",
        "testing_support.npsc5f_final_regression_matrix",
    )
    for path in catalog_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, (path.as_posix(), fragment)
        assert "from tests." not in source, path.as_posix()


def test_t15_generic_core_has_no_profile_specific_branches() -> None:
    forbidden_tokens = ("npsc5e-r1-final", "npsc5f-r4-final", "NPSC5F_R1_PROFILE_ID")
    for module in (
        compiler_module,
        coordinator_module,
        plan_runner_module,
        aggregate_module,
    ):
        source = inspect.getsource(module)
        for token in forbidden_tokens:
            assert token not in source, module.__name__


def test_t16_suite_identity_consistency() -> None:
    registry = pytest_to_suite_id_registry()
    by_args: dict[tuple[str, ...], str] = {}
    for args, suite_id in registry.items():
        if args in by_args and by_args[args] != suite_id:
            raise AssertionError(
                f"duplicate suite identity for {args!r}: "
                f"{by_args[args]!r} vs {suite_id!r}",
            )
        by_args[args] = suite_id


def test_t17_t18_physical_dedup_and_gate_semantics(
    parity_certifier: QualificationSemanticParityCertifier,
) -> None:
    report = parity_certifier.certify_all(MATRIX)
    for result in report.profile_results:
        assert result.physical_dedup_parity
        assert result.gate_semantics_parity


def test_aggregate_gates_no_subprocess_import() -> None:
    source = inspect.getsource(aggregate_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "subprocess"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "subprocess"


def test_no_orchestrator_path_leaves_in_any_profile() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        compiled = catalog.compile_profile(profile_id)
        for suite_id in compiled.plan.leaf_suite_ids:
            suite = compiled.suite_by_id[suite_id]
            if len(suite.pytest_arguments) == 1:
                assert suite.pytest_arguments[0] not in CANONICAL_ORCHESTRATOR_PATHS
