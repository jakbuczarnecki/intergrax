# © Artur Czarnecki. All rights reserved.

"""Tests for invocation-scoped pytest basetemp resolution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from testing_support.pytest_temp_root import (
    LEGACY_SHARED_BASETEMP_MARKER,
    PYTEST_TEMP_NAMESPACE,
    allocate_invocation_pytest_basetemp,
    apply_invocation_pytest_basetemp,
    explicit_cli_basetemp,
    get_or_allocate_invocation_basetemp,
    reset_invocation_basetemp_cache,
    resolved_pytest_basetemp,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_pytest_basetemp_cache() -> None:
    reset_invocation_basetemp_cache()


@pytest.fixture
def temp_repo_root(tmp_path: Path) -> Path:
    namespace = tmp_path / PYTEST_TEMP_NAMESPACE
    namespace.mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_sequential_allocations_use_distinct_basetemp_paths(temp_repo_root: Path) -> None:
    first = allocate_invocation_pytest_basetemp(temp_repo_root)
    second = allocate_invocation_pytest_basetemp(temp_repo_root)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_same_invocation_basetemp_resolution_is_stable(temp_repo_root: Path) -> None:
    first = get_or_allocate_invocation_basetemp(temp_repo_root)
    second = get_or_allocate_invocation_basetemp(temp_repo_root)
    assert first == second


def test_explicit_cli_basetemp_is_detected() -> None:
    assert explicit_cli_basetemp(("--basetemp=build/explicit",)) == "build/explicit"
    assert explicit_cli_basetemp(("--basetemp", "build/explicit")) == "build/explicit"
    assert explicit_cli_basetemp(("-q", "tests/foo.py")) is None


def test_generated_basetemp_path_is_windows_safe(temp_repo_root: Path) -> None:
    basetemp = allocate_invocation_pytest_basetemp(temp_repo_root)
    relative = basetemp.relative_to(temp_repo_root).as_posix()
    assert relative.startswith(f"{PYTEST_TEMP_NAMESPACE}/")
    assert "<" not in relative
    assert ">" not in relative
    assert ":" not in relative.split("/")[-1]
    assert "|" not in relative
    assert "?" not in relative
    assert "*" not in relative


def test_apply_invocation_basetemp_preserves_explicit_cli(temp_repo_root: Path) -> None:
    config = pytest.Config.fromdictargs(
        {"basetemp": None},
        ["pytest", "--basetemp=build/explicit-cli"],
    )
    result = apply_invocation_pytest_basetemp(config, temp_repo_root)
    assert result is None
    assert config.option.basetemp == "build/explicit-cli"


def test_apply_invocation_basetemp_assigns_namespace_path(temp_repo_root: Path) -> None:
    config = pytest.Config.fromdictargs({"basetemp": None}, ["pytest", "-q"])
    assigned = apply_invocation_pytest_basetemp(config, temp_repo_root)
    assert assigned is not None
    assert config.option.basetemp == str(assigned)
    assert resolved_pytest_basetemp(config) == assigned
    relative = assigned.relative_to(temp_repo_root).as_posix()
    assert relative.startswith(f"{PYTEST_TEMP_NAMESPACE}/")
    assert LEGACY_SHARED_BASETEMP_MARKER not in relative


def test_parent_and_child_pytest_invocations_use_distinct_basetemp() -> None:
    probe = (
        "from pathlib import Path;"
        "import pytest;"
        "from testing_support.pytest_temp_root import apply_invocation_pytest_basetemp, resolved_pytest_basetemp;"
        "config = pytest.Config.fromdictargs({'basetemp': None}, ['pytest', '-q']);"
        "apply_invocation_pytest_basetemp(config, Path('.'));"
        "print(resolved_pytest_basetemp(config).as_posix())"
    )
    parent = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    child = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    parent_root = parent.stdout.strip()
    child_root = child.stdout.strip()
    assert parent_root != child_root
    assert parent_root.startswith(f"{PYTEST_TEMP_NAMESPACE}/")
    assert child_root.startswith(f"{PYTEST_TEMP_NAMESPACE}/")
    assert LEGACY_SHARED_BASETEMP_MARKER not in parent_root
    assert LEGACY_SHARED_BASETEMP_MARKER not in child_root
