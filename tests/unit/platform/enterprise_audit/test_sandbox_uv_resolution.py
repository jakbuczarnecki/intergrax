# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditCategory, AuditMode, AuditSeverity, ManifestCheck
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox, _resolve_uv_executable


def test_resolve_uv_executable_uses_which() -> None:
    with patch("intergrax.platform.enterprise_audit.sandbox.shutil.which", return_value="/bin/uv"):
        assert _resolve_uv_executable() == "/bin/uv"


def test_resolve_uv_executable_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uv = tmp_path / "uv.exe"
    uv.write_text("", encoding="utf-8")
    monkeypatch.setenv("INTERGRAX_UV_EXECUTABLE", str(uv))
    assert _resolve_uv_executable() == str(uv)


def test_resolve_uv_executable_env_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_UV_EXECUTABLE", "C:/nope/uv.exe")
    with pytest.raises(FileNotFoundError, match="INTERGRAX_UV_EXECUTABLE"):
        _resolve_uv_executable()


def test_resolve_uv_executable_uses_uv_env_when_not_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uv = tmp_path / "uv.exe"
    uv.write_text("", encoding="utf-8")
    monkeypatch.delenv("INTERGRAX_UV_EXECUTABLE", raising=False)
    monkeypatch.setenv("UV", str(uv))
    with patch("intergrax.platform.enterprise_audit.sandbox.shutil.which", return_value=None):
        assert _resolve_uv_executable() == str(uv)


def test_resolve_uv_executable_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_UV_EXECUTABLE", raising=False)
    monkeypatch.delenv("UV", raising=False)
    with patch("intergrax.platform.enterprise_audit.sandbox.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            _resolve_uv_executable()


def test_run_check_argv_uses_resolved_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uv = tmp_path / "uv.cmd"
    uv.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setenv("INTERGRAX_UV_EXECUTABLE", str(uv))
    repo = tmp_path / "repo"
    repo.mkdir()
    check = ManifestCheck(
        check_id="UNIT-UV",
        entrypoint_kind="pytest_file",
        entrypoint="tests/conftest.py",
        mutating=False,
        modes=frozenset({AuditMode.FAST}),
        severity_default=AuditSeverity.INFO,
        category=AuditCategory.INTERNAL,
    )
    sandbox = CheckExecutionSandbox(repo)
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "")
    mock_proc.returncode = 0
    with patch("intergrax.platform.enterprise_audit.sandbox.subprocess.Popen", return_value=mock_proc) as popen:
        sandbox.run_check(check)
    argv = popen.call_args[0][0]
    assert argv[0] == str(uv)
    assert argv[1:4] == ["run", "pytest", "tests/conftest.py"]
