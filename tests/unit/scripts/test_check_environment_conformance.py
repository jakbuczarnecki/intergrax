# © Artur Czarnecki. All rights reserved.

"""Focused tests for ENV-3 environment conformance invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.maintenance.check_environment_conformance import (
    evaluate_base_provenance,
    evaluate_pyvenv_cfg,
    evaluate_pythonpath,
    evaluate_venv_paths,
    profile_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_ci]


@pytest.mark.parametrize("profile", ["local", "ci"])
def test_venv_paths_accept_repository_venv_and_reject_external_paths(
    tmp_path: Path, profile: str
) -> None:
    assert profile_contract(profile).name == profile
    venv_root = tmp_path / ".venv"
    executable = venv_root / "Scripts" / "python.exe"

    assert evaluate_venv_paths(executable, venv_root, venv_root)[0] is True
    assert (
        evaluate_venv_paths(tmp_path / "system" / "python.exe", venv_root, venv_root)[0]
        is False
    )
    assert (
        evaluate_venv_paths(executable, tmp_path / "other-venv", venv_root)[0] is False
    )


@pytest.mark.parametrize("profile", ["local", "ci"])
def test_non_empty_pythonpath_is_rejected(profile: str) -> None:
    assert profile_contract(profile).name == profile
    assert evaluate_pythonpath({})[0] is True
    assert evaluate_pythonpath({"PYTHONPATH": "C:\\external\\injection"})[0] is False


def test_local_profile_requires_managed_base_provenance() -> None:
    assert profile_contract("local").requires_managed_python is True


def test_ci_profile_does_not_require_local_managed_base_provenance() -> None:
    assert profile_contract("ci").requires_managed_python is False


def test_ci_keeps_shared_venv_isolation_and_has_intentional_baseline() -> None:
    local = profile_contract("local")
    ci = profile_contract("ci")

    assert {"PYTEST", "PYDANTIC", "FASTAPI"} <= {
        name for name, *_ in ci.dependencies
    }
    assert "RUFF" in {name for name, *_ in local.dependencies}
    assert "RUFF" not in {name for name, *_ in ci.dependencies}


def test_external_base_interpreter_is_rejected(tmp_path: Path) -> None:
    managed_root = tmp_path / "uv" / "python"
    expected = managed_root / "cpython-3.12.11" / "python.exe"
    external = tmp_path / "system" / "python.exe"

    ok, detail = evaluate_base_provenance(
        external,
        expected.parent,
        expected,
        managed_root,
        {"major": 3, "minor": 12},
    )

    assert ok is False
    assert "expected" in detail


def test_venv_paths_accept_base_runtime_when_prefix_is_active_venv(
    tmp_path: Path,
) -> None:
    venv_root = tmp_path / "repo" / ".venv"
    base_executable = tmp_path / "usr" / "bin" / "python3.12"

    ok, detail = evaluate_venv_paths(
        base_executable,
        venv_root,
        venv_root,
        base_executable=base_executable,
    )

    assert ok is True
    assert "base runtime for active venv" in detail


def test_pyvenv_cfg_home_matches_base_executable_parent(tmp_path: Path) -> None:
    home_dir = tmp_path / "usr" / "bin"
    base_executable = home_dir / "python3.12"
    cfg_path = tmp_path / "pyvenv.cfg"
    cfg_path.write_text(
        f"home = {home_dir}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    ok, detail, _ = evaluate_pyvenv_cfg(cfg_path, base_executable)

    assert ok is True
    assert "isolated=true" in detail


def test_pyvenv_cfg_rejects_home_mismatch(tmp_path: Path) -> None:
    base_executable = tmp_path / "usr" / "bin" / "python3.12"
    cfg_path = tmp_path / "pyvenv.cfg"
    cfg_path.write_text(
        f"home = {tmp_path / 'other' / 'bin'}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    ok, detail, _ = evaluate_pyvenv_cfg(cfg_path, base_executable)

    assert ok is False
    assert "base_executable parent" in detail


def test_conda_base_interpreter_is_rejected(tmp_path: Path) -> None:
    managed_root = tmp_path / "uv" / "python"
    conda = managed_root / "conda-3.12" / "python.exe"

    ok, detail = evaluate_base_provenance(
        conda,
        conda.parent,
        conda,
        managed_root,
        {"major": 3, "minor": 12},
    )

    assert ok is False
    assert "Conda/Anaconda" in detail
