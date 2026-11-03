from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from intergrax.platform.enterprise_audit.contracts import ManifestCheck
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox


def _minimal_check() -> ManifestCheck:
    return ManifestCheck(
        check_id="ENT-A-BND-01",
        entrypoint_kind="pytest_nodeid",
        entrypoint="tests/qualification/ent_audit_x/unit/test_repo_identity.py::test_repo_identity_reads_head_and_branch",
    )


def test_sandbox_timeout_returns_124(tmp_path) -> None:
    sandbox = CheckExecutionSandbox(tmp_path, timeout_seconds=1)
    proc = MagicMock()
    proc.pid = 99999
    proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=1)
    proc.returncode = None
    with patch("intergrax.platform.enterprise_audit.sandbox.subprocess.Popen", return_value=proc):
        with patch("intergrax.platform.enterprise_audit.sandbox._kill_process_tree"):
            result = sandbox.run_check(_minimal_check())
    assert result.exit_code == 124
    assert "TIMEOUT" in result.stderr


def test_sandbox_run_id_unique_per_instance(tmp_path) -> None:
    a = CheckExecutionSandbox(tmp_path)
    b = CheckExecutionSandbox(tmp_path)
    assert a.run_id != b.run_id
