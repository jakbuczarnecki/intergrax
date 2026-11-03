# © Artur Czarnecki. All rights reserved.

"""Shared semantic gate oracle for ENT-AUDIT-X acceptance (C033)."""

from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.checks.executor import execute_manifest_check
from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox


def assert_semantic_gate_executes(
    repo_root: Path,
    check_id: str,
    *,
    prefer_mode: AuditMode | None = None,
) -> None:
    checks = load_default_manifest()
    check = next(c for c in checks if c.check_id == check_id)
    mode = prefer_mode or (AuditMode.FAST if AuditMode.FAST in check.modes else AuditMode.FULL)
    sandbox = CheckExecutionSandbox(repo_root)
    outcome = execute_manifest_check(check, sandbox, mode=mode)
    assert outcome.executed, f"{check_id} not executed: {outcome.skip_reason}"
    assert not outcome.skipped, f"{check_id} skipped: {outcome.skip_reason}"
    assert outcome.exit_code == 0, f"{check_id} failed exit={outcome.exit_code} findings={outcome.findings}"
