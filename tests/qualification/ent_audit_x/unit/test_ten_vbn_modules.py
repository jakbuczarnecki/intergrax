from __future__ import annotations

from intergrax.platform.enterprise_audit.vbn import vbn_domain_rollup


def test_ten_and_vbn_module_ids_present_in_manifest() -> None:
    from intergrax.platform.enterprise_audit.manifest import load_default_manifest

    checks = load_default_manifest()
    ids = {c.check_id for c in checks}
    for ten in (
        "TEN-IDENTITY",
        "TEN-FAIL-CLOSED",
    ):
        assert ten in ids
    for vbn in ("VBN-INVENTORY-DRIFT", "VBN-REACHABILITY"):
        assert vbn in ids


def test_vbn_rollup_reports_thirty_one_domains() -> None:
    rollup = vbn_domain_rollup(())
    assert rollup["domains_total"] == 31
