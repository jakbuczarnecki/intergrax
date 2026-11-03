from __future__ import annotations

from intergrax.platform.enterprise_audit.digests import manifest_digest
from intergrax.platform.enterprise_audit.vbn import validate_domain_report


def _base_report(**overrides) -> dict:
    domains = [
        {
            "domain_id": f"D{i:02d}",
            "domain_key": f"k{i}",
            "status": "QUALIFIED" if i <= 28 else "NA_WITH_EVIDENCE",
            "coverage_note": "COVERED" if i <= 28 else "N/A vendor",
            "inventory_error_count": 0,
            "test_nodeid": "tests/x.py",
            "rpl_class": "QUALIFIED",
        }
        for i in range(1, 32)
    ]
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "qualification_sha": "a" * 40,
        "check_manifest_digest": manifest_digest(),
        "domains_total": 31,
        "domains": domains,
        "limitations": [],
    }
    payload.update(overrides)
    return payload


def test_vbn_missing_qualification_sha_rejected() -> None:
    report = _base_report(qualification_sha="")
    errors = validate_domain_report(report, expected_sha="a" * 40)
    assert any("qualification_sha" in e for e in errors)


def test_vbn_stale_manifest_digest_rejected() -> None:
    report = _base_report(check_manifest_digest="b" * 64)
    errors = validate_domain_report(report, expected_manifest_digest=manifest_digest())
    assert any("check_manifest_digest stale" in e for e in errors)


def test_vbn_duplicate_domain_id_rejected() -> None:
    report = _base_report()
    report["domains"][1]["domain_id"] = "D01"
    errors = validate_domain_report(report)
    assert any("duplicate" in e for e in errors)


def test_vbn_unauthorized_na_rejected() -> None:
    report = _base_report()
    report["domains"][0]["status"] = "NA_WITH_EVIDENCE"
    report["domains"][0]["coverage_note"] = "COVERED"
    errors = validate_domain_report(report)
    assert any("unauthorized NA_WITH_EVIDENCE" in e for e in errors)


def test_vbn_gap_and_partial_allowed() -> None:
    report = _base_report()
    report["domains"][0]["status"] = "GAP"
    report["domains"][1]["status"] = "PARTIAL"
    errors = validate_domain_report(report)
    assert errors == ()
