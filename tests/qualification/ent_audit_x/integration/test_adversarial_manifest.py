from __future__ import annotations

from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_closed_world_manifest_rejects_unknown_id_at_runtime() -> None:
    checks = load_default_manifest()
    known = {c.check_id for c in checks}
    assert "ENT-A-BND-01" in known
    assert "NOT-A-REAL-CHECK" not in known
