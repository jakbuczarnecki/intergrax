from __future__ import annotations

from intergrax.platform.enterprise_audit.digests import manifest_digest


def test_manifest_digest_stable() -> None:
    first = manifest_digest()
    second = manifest_digest()
    assert first == second
    assert len(first) == 64
