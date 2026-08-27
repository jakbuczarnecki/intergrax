# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_ROOTS = (
    _REPO_ROOT
    / "applications"
    / "local_workspace_application"
    / "scripts"
    / "lkw_tier3_source_roots.py"
)


def _load_source_roots_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lkw_tier3_source_roots",
        _SOURCE_ROOTS,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_roots() -> ModuleType:
    return _load_source_roots_module()


def test_resolve_tier3_source_roots_returns_canonical_directories(
    source_roots: ModuleType,
) -> None:
    applications, agents = source_roots.resolve_tier3_source_roots(_REPO_ROOT)
    assert applications == (_REPO_ROOT / "applications").resolve()
    assert agents == (_REPO_ROOT / "agents").resolve()
    assert applications.is_dir()
    assert agents.is_dir()


def test_format_windows_path_list_preserves_existing_pythonpath(
    source_roots: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "C:\\existing")
    rendered = source_roots.format_windows_path_list(_REPO_ROOT)
    applications, agents = source_roots.resolve_tier3_source_roots(_REPO_ROOT)
    assert rendered == f"{applications}{os.pathsep}{agents}{os.pathsep}C:\\existing"


def test_format_windows_path_list_without_existing_pythonpath(
    source_roots: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    rendered = source_roots.format_windows_path_list(_REPO_ROOT)
    applications, agents = source_roots.resolve_tier3_source_roots(_REPO_ROOT)
    assert rendered == f"{applications}{os.pathsep}{agents}"


def test_main_windows_path_list_stdout(
    source_roots: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    exit_code = source_roots.main(
        [
            "--repo-root",
            str(_REPO_ROOT),
            "--format",
            "windows-path-list",
        ]
    )
    captured = capsys.readouterr()
    applications, agents = source_roots.resolve_tier3_source_roots(_REPO_ROOT)
    assert exit_code == 0
    assert captured.out.strip() == f"{applications}{os.pathsep}{agents}"


def test_ensure_tier3_source_roots_on_sys_path_inserts_repo_roots(
    source_roots: ModuleType,
) -> None:
    original_path = list(sys.path)
    try:
        sys.path[:] = original_path
        applications, agents = source_roots.ensure_tier3_source_roots_on_sys_path(
            _REPO_ROOT
        )
        assert str(applications) in sys.path
        assert str(agents) in sys.path
    finally:
        sys.path[:] = original_path
