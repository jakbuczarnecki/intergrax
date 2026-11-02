from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_acceptance_ent_a_ten_02_manifest_entrypoint_exists() -> None:
    checks = load_default_manifest()
    match = next(c for c in checks if c.check_id == "ENT-A-TEN-02")
    assert match.entrypoint_kind != "internal_pending"
    root = Path(__file__).resolve().parents[4]
    if "::" in match.entrypoint:
        rel = match.entrypoint.split("::", 1)[0]
    else:
        rel = match.entrypoint
    assert (root / rel).is_file(), rel
