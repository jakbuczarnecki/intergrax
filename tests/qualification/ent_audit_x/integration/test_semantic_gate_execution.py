from __future__ import annotations

import os
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.checks.executor import execute_manifest_check
from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox


def test_semantic_gate_bnd_03_script_executes() -> None:
    root = Path(__file__).resolve().parents[4]
    check = next(c for c in load_default_manifest() if c.check_id == "ENT-A-BND-03")
    sandbox = CheckExecutionSandbox(root, timeout_seconds=300)
    outcome = execute_manifest_check(check, sandbox, mode=AuditMode.FAST)
    assert outcome.executed is True
    assert outcome.exit_code in {0, 1, 2, 3}


@pytest.mark.parametrize(
    "check_id",
    [c.check_id for c in load_default_manifest()],
)
def test_semantic_gate_matrix_optional(check_id: str) -> None:
    if os.environ.get("ENT_AUDIT_X_GATE_MATRIX") != "1":
        pytest.skip("full 85 gate matrix requires ENT_AUDIT_X_GATE_MATRIX=1")
    root = Path(__file__).resolve().parents[4]
    check = next(c for c in load_default_manifest() if c.check_id == check_id)
    sandbox = CheckExecutionSandbox(root, timeout_seconds=600)
    outcome = execute_manifest_check(check, sandbox, mode=AuditMode.FULL)
    assert outcome.executed or outcome.skipped
    if outcome.skipped:
        pytest.fail(f"{check_id} skipped: {outcome.skip_reason}")
