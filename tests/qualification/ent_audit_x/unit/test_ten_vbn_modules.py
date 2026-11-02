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


def test_vbn_rollup_uses_domain_report_when_present() -> None:
    report = {
        "domains_total": 31,
        "domains": [{"status": "QUALIFIED"} for _ in range(31)],
    }
    rollup = vbn_domain_rollup((), domain_report=report)
    assert rollup["domains_accounted"] == 31
