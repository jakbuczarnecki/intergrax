from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditCategory, AuditMode, AuditSeverity, ManifestCheck
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox


def test_sandbox_rejects_path_outside_repo(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    sandbox = CheckExecutionSandbox(root)
    check = ManifestCheck(
        check_id="ADV-17",
        entrypoint_kind="pytest_file",
        entrypoint="../../../../../etc/passwd",
        modes=frozenset({AuditMode.FAST}),
        severity_default=AuditSeverity.ERROR,
        category=AuditCategory.ARCHITECTURE,
    )
    with pytest.raises(ValueError, match=r"\.\.|escapes"):
        sandbox.run_check(check)
