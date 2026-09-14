# © Artur Czarnecki. All rights reserved.

"""Global anti-regression guards for canonical qualification catalog."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

from testing_support.execution_qualification import aggregate as aggregate_module
from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.expansion import (
    is_nested_orchestrator_leaf,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator

from ..fake_executor import FakeQualificationSuiteExecutor


def test_canonical_qualification_catalog_has_no_nested_pytest_orchestrator_leaves() -> (
    None
):
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        compiled = catalog.compile_profile(profile_id)
        for suite_id in compiled.plan.leaf_suite_ids:
            suite = compiled.suite_by_id[suite_id]
            assert not is_nested_orchestrator_leaf(suite.pytest_arguments), (
                f"{profile_id}: {suite_id}"
            )


def test_canonical_aggregate_gates_do_not_launch_external_processes() -> None:
    source = inspect.getsource(aggregate_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "subprocess"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "subprocess"


def test_each_canonical_leaf_suite_id_is_unique() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        compiled = catalog.compile_profile(profile_id)
        assert len(compiled.plan.leaf_suite_ids) == len(
            set(compiled.plan.leaf_suite_ids)
        )


def test_no_canonical_catalog_imports_tests_unit_runtime_architecture() -> None:
    catalog_root = Path("testing_support/execution_qualification/catalog")
    for path in catalog_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "tests.unit.runtime.architecture" not in source, path.as_posix()
        assert "from tests." not in source, path.as_posix()


def test_each_leaf_executes_at_most_once_per_plan_run(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    catalog = build_default_qualification_catalog()
    compiled = catalog.compile_profile("npsc5f-r1-final")
    fake = FakeQualificationSuiteExecutor({})
    coordinator = QualificationCoordinator(executor=fake)
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=4,
        run_artifact_root=run_artifact_root,
        suite_timeout_seconds=30.0,
        run_id="catalog-dedup-guard",
    )
    run_qualification_execution_plan(compiled.plan, config, coordinator=coordinator)
    for suite_id in compiled.plan.leaf_suite_ids:
        assert fake.invocation_counts.get(suite_id, 0) == 1


def test_catalog_modules_do_not_import_intergrax_runtime() -> None:
    package = importlib.import_module("testing_support.execution_qualification.catalog")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        for value in vars(module).values():
            if inspect.ismodule(value) and value.__name__.startswith("intergrax."):
                raise AssertionError(f"{module_info.name} imports intergrax runtime")
