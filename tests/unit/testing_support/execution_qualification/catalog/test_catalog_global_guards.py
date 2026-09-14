# © Artur Czarnecki. All rights reserved.

"""Global anti-regression guards for canonical qualification catalog."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

from testing_support.execution_qualification import aggregate as aggregate_module
from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.expansion import (
    is_nested_orchestrator_leaf,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    CANONICAL_ORCHESTRATOR_PATHS,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    PROFILE_BUILDERS,
)
from testing_support.execution_qualification.catalog.suite_registry import (
    pytest_to_suite_id_registry,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator

from ..fake_executor import FakeQualificationSuiteExecutor


def test_all_canonical_profiles_compile() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        catalog.compile_profile(profile_id)


def test_all_canonical_profiles_are_free_of_nested_pytest_orchestrator_leaves() -> None:
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


def test_no_canonical_profile_contains_any_known_orchestrator_as_leaf() -> None:
    catalog = build_default_qualification_catalog()
    for profile_id in catalog.profile_ids:
        compiled = catalog.compile_profile(profile_id)
        for suite_id in compiled.plan.leaf_suite_ids:
            suite = compiled.suite_by_id[suite_id]
            if len(suite.pytest_arguments) == 1:
                assert suite.pytest_arguments[0] not in CANONICAL_ORCHESTRATOR_PATHS, (
                    f"{profile_id}: {suite_id} -> {suite.pytest_arguments[0]}"
                )


def test_canonical_catalog_does_not_depend_on_legacy_regression_matrix_modules() -> (
    None
):
    catalog_root = Path("testing_support/execution_qualification/catalog")
    forbidden = (
        "testing_support.npsc5f_r4_regression_matrix",
        "testing_support.npsc5f_final_regression_matrix",
    )
    for path in catalog_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, (path.as_posix(), fragment)


def test_default_catalog_profile_builder_map_is_immutable() -> None:
    from collections.abc import MutableMapping
    from types import MappingProxyType
    from typing import cast

    assert isinstance(PROFILE_BUILDERS, MappingProxyType)
    mutable_view = cast(
        MutableMapping[str, object],
        PROFILE_BUILDERS,
    )
    with pytest.raises(TypeError):
        mutable_view["x"] = lambda: None


def test_normalized_pytest_args_map_to_at_most_one_suite_id() -> None:
    registry = pytest_to_suite_id_registry()
    by_args: dict[tuple[str, ...], str] = {}
    for args, suite_id in registry.items():
        if args in by_args and by_args[args] != suite_id:
            raise AssertionError(
                f"duplicate suite identity for {args!r}: "
                f"{by_args[args]!r} vs {suite_id!r}",
            )
        by_args[args] = suite_id


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
