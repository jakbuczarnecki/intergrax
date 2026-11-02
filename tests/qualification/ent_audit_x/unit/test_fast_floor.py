from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import AuditMode, CheckRunOutcome
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.modes import fast_floor_incomplete


def test_fast_floor_incomplete_when_mandatory_missing() -> None:
    checks = load_default_manifest()
    mandatory = next(c for c in checks if c.fast_mandatory)
    outcomes = tuple(
        CheckRunOutcome(check_id=c.check_id, executed=c.check_id != mandatory.check_id, skipped=False)
        for c in checks
        if c.fast_mandatory
    )
    assert fast_floor_incomplete(AuditMode.FAST, checks, outcomes)
