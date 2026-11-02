from __future__ import annotations

from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_every_check_maps_to_frz_or_sys_inv() -> None:
    checks = load_default_manifest()
    assert len(checks) == 85
    for check in checks:
        assert check.pg_id or check.check_id.startswith(("TEN-", "VBN-"))
